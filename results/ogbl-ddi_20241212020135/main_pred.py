from utils import *
from models import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# python3 -m tensorboard.main --logdir .
## args settings #######################################################################################################
parser = argparse.ArgumentParser(description='ComHG for link prediction')
parser.add_argument('--float', default=np.float32)
parser.add_argument('--dataset', type=str, default='ogbl-ddi')  # ddi collab ppa citation2
parser.add_argument('--result_appendix', type=str, default='', help="if '', time as appendix")
parser.add_argument('--device', type=str, default='0', help="cpu or gpu id")
# basic setting for ogb graph datasets
parser.add_argument('--directed', type=bool, default=False)
parser.add_argument('--coalesce', type=bool, default=True, help="whether to coalesce multiple edges between two nodes")
parser.add_argument('--use_weight', type=bool, default=False, help="whether to use edge weight.")
parser.add_argument('--use_val', type=bool, default=False, help="whether to use edges in validation set for testing")
parser.add_argument('--collab_year', type = int, default = 2010, help="for ogbl-collab.")

# feature setting
parser.add_argument('--use_feature', type=bool, default=False)
parser.add_argument('--use_node_emb', type=bool, default=True)
parser.add_argument('--use_dist', type=bool, default=False, help="whether to use shortest path distance")
parser.add_argument('--use_cn', type=bool, default=False, help="whether to use common neighbor")
parser.add_argument('--use_aa', type=bool, default=False, help="whether to use Adamic/Adar")
parser.add_argument('--use_ja', type=bool, default=False, help="whether to use Jaccard")
parser.add_argument('--use_ra', type=bool, default=False, help="whether to use Resource Allocation")
parser.add_argument('--use_degree', type=bool, default=False)
## max heuristics
parser.add_argument('--max_dist', type=int, default=5)
parser.add_argument('--max_cn', type=int, default=100)
parser.add_argument('--max_aa', type=int, default=100)
parser.add_argument('--max_ja', type=int, default=100)
parser.add_argument('--max_ra', type=int, default=20)
parser.add_argument('--max_degree', type=int, default=1000)
# magnitude for ja aa ra. ja, aa, ra are float. We convert them to int by multiply the magnitude, e.g. mag_ja=100 will be ja*100.
parser.add_argument('--mag_ja', type=int, default=100)
parser.add_argument('--mag_aa', type=int, default=10)
parser.add_argument('--mag_ra', type=int, default=10)
# distance computing setting
parser.add_argument('--heurisctic_reproduce', type=bool, default=False, help="whether to re-produce heuristics, False would speed up training but may decrease the performance")
parser.add_argument('--heurisctic_batch_size', type=int, default=100, help="number of rows in distance computation")
parser.add_argument('--heurisctic_directed', type=bool, default=False, help="use directed graph in distance computation")
parser.add_argument('--heurisctic_reuse', type=bool, default=True)
parser.add_argument('--neg_size', type=float, default=2000, help="size of data_neg_train: neg_size * len(split_edge['train']['edge'])")


# model setting
parser.add_argument('--adj_hop', type=int, default=2, help="hieghest hop of adj for GNN.")
parser.add_argument('--adj_neg', type=float, default=0.0, help="neg samples (global neighbors) used in adj. ")
parser.add_argument('--adj_neg_dist', type=int, default=10, help="distance from global neighbors to the central node.")
parser.add_argument('--adj_weight', type=str, default='same', help="the method for weights in the original adj. options: same or decay")
## attention setting
parser.add_argument('--atten_type', type=str, default='Multiply',
                    help="attention mechnism. Options: Concat, Cosine, Multiply, no_atten, with repsect to GAT, AGNN, Transformer and GCN")
parser.add_argument('--atten_combine', type=str, default='plus',
                    help="type of combining original adj with attention (if use). must be: plus, multiply or only_atten")
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--dim_node_emb', type=int, default=512)
parser.add_argument('--dim_encoding', type=int, default=32, help="dim for encoding heuristics, etc.")
parser.add_argument('--dim_atten', type=int, default=8, help="dim for matrix multiplication. Should be small for large graph")
parser.add_argument('--dim_hidden', type=int, default=None, help="if None, dim_hidden = dim_in")
parser.add_argument('--n_layers', type=int, default=2, help="CNN layers")
parser.add_argument('--n_heads', type=int, default=1, help="multiply head CNN")
parser.add_argument('--n_layers_mlp', type=int, default=5)
parser.add_argument('--residual', type=bool, default=True, help="whether to use residual connection in CNN")
parser.add_argument('--reduce', type=str, default='add', help="combine outputs of multi-head CNN modules. options: concat, add")
parser.add_argument('--negative_slope', type=float, default=0.2, help="negative_slope for leaky_relu")
# learning setting
parser.add_argument('--num_workers', type=int, default=64)
parser.add_argument('--optimizer', type=str, default='Adam', help="'Adam', 'AdamW', 'SGD'")
parser.add_argument('--clip_grad_norm', type=float, default=1.0, help="whether to use clip_grad_norm_ in training")
parser.add_argument('--use_layer_norm', type=bool, default=False, help="whether to use layer norm")
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--dropout_adj', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_mini', type=float, default=0.00001, help="lr stops decreasing at lr_mini")
parser.add_argument('--scheduler_gamma', type=float, default=0.997)
parser.add_argument('--shuffle', type=bool, default=True)
# parser.add_argument('--val_per', type=float, default=1)
# training setting
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100000)
parser.add_argument('--batch_num', type=int, default=1000, help="number of batches trained in an epoch")
parser.add_argument('--world_size', type=int, default=4, help="number of GPUs to use")
parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training")

########################################################################################################################
args = parser.parse_args()
args.max_dist = max(args.max_dist, 3) if args.use_dist else 3 # 3: includes: 2-hop neighbors is 2 and others is 3
########################################################################################################################

# device setting
device = torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
if device == torch.device('cuda'): torch.cuda.set_device(int(args.device))
args.device = device
print(args.device)
# evaluation metrics
metrics = {'ogbl-collab': {'metric': 'Hits@50', 'hitK': [50], 'dense_sparse': 'dense'},
           'ogbl-ddi': {'metric': 'Hits@20', 'hitK': [20], 'dense_sparse': 'dense'},
           'ogbl-ppa': {'metric': 'Hits@100', 'hitK': [100], 'dense_sparse': 'sparse'},
           'ogbl-citation2': {'metric': 'MRR', 'hitK': [20], 'dense_sparse': 'sparse'},
           }
args.eval_metrics, args.hitK, args.dense_sparse = metrics[args.dataset]['metric'], metrics[args.dataset]['hitK'], metrics[args.dataset]['dense_sparse']
# results saving setting
if args.result_appendix == '': args.result_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.dir_result = os.path.join('results/{}{}'.format(args.dataset, args.result_appendix))
print('Results will be saved in ' + args.dir_result)
if not os.path.exists(args.dir_result):  os.makedirs(args.dir_result)
for file in ['main_pred.py', 'utils.py', 'models.py']: copy(file, args.dir_result)
# loggers for result record
loggers = get_loggers(args)
log_file = osp.join(args.dir_result, 'log.log')
# print args
with open(log_file, 'w') as f:
    print(str(args), file=f)
    print(str(args), file=sys.stdout)
########################################################################################################################
## input data ##########################################################################################################
data_pos_train = graph_prepare(args, posneg_split='pos_train')
data_neg_train = graph_prepare(args, posneg_split='neg_train')
data_pos_valid = graph_prepare(args, posneg_split='pos_valid')
data_neg_valid = graph_prepare(args, posneg_split='neg_valid')
data_pos_test = graph_prepare(args, posneg_split='pos_test')
data_neg_test = graph_prepare(args, posneg_split='neg_test')
########################################################################################################################
## args dim in #########################################################################################################
def get_dim_in(args, data):
    if data.x != None:
        x_size = data.x.size(-1)
        if args.dataset == 'ogbl-ppa':
            x_size = args.dim_encoding
    args.dim_in = 0
    if args.use_feature and data.x != None:
        args.dim_in += x_size
    if args.use_node_emb:
        args.dim_in += args.dim_node_emb
    if args.use_degree:
        args.dim_in += args.dim_encoding
    if args.dim_in == 0:
        args.dim_in = 1

    return args
########################################################################################################################
## train ###############################################################################################################

def my_DataLoader(args, data, shuffle=False):
    edges = data.edges
    num_edge = edges.size(0)
    perm_size = int(min(args.batch_num * args.batch_size, num_edge))
    if shuffle:
        if num_edge > 1E8:
            perm = np.array(random.sample(range(num_edge), perm_size))
        else:
            perm = np.random.permutation(num_edge)
            perm = perm[:perm_size]
        edges = edges[perm]
    step, end = args.batch_size, perm_size
    perms = [np.array(range(i, i + step)) if i + step < end else np.array(range(i, end)) for i in range(0, end, step)]

    return edges, perms

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, args, predictor, optimizer, scheduler, data_pos, data_neg):
    predictor.train()
    running_loss = running_examples = 0

    # Wrap model with DDP
    predictor = DDP(predictor, device_ids=[rank])
    
    pos_edges, pos_perms = my_DataLoader(args, data_pos, shuffle=args.shuffle)
    neg_edges, neg_perms = my_DataLoader(args, data_neg, shuffle=args.shuffle)
    
    # Create DistributedSampler for the edges
    pos_sampler = DistributedSampler(pos_perms, num_replicas=args.world_size, rank=rank)
    neg_sampler = DistributedSampler(neg_perms, num_replicas=args.world_size, rank=rank)
    
    leniter = min(len(pos_perms), len(neg_perms))
    for i in range(leniter):
        optimizer.zero_grad()
        
        # Get indices for this GPU
        pos_idx = pos_sampler.dataset[i]
        neg_idx = neg_sampler.dataset[i]
        
        edge_pos = pos_edges[pos_idx].to(rank)
        edge_neg = neg_edges[neg_idx].to(rank)
        
        y_pos = predictor(data_pos, edge_pos)
        y_neg = predictor(data_neg, edge_neg)
        
        if len(y_pos) != len(y_neg):
            break

        pos_loss = (-torch.log(y_pos + 1e-15)).mean()
        neg_loss = (-torch.log(1.0 - y_neg + 1e-15)).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        running_examples += 1
        
    # Synchronize loss across GPUs
    dist.all_reduce(torch.tensor(running_loss).to(rank))
    dist.all_reduce(torch.tensor(running_examples).to(rank))
    
    return running_loss / running_examples


@torch.no_grad()
def test(args, predictor, data_pos, data_neg):
    predictor.eval()

    def get_predictions(args, predictor, data_pred):
        pred = []
        edges = data_pred.edges
        num_edge = edges.size(0)
        step, end = args.batch_size*100, num_edge
        perms = [np.array(range(i, i + step)) if i + step < end else np.array(range(i, end)) for i in range(0, end, step)]
        for perm in perms:
            edge_batch = edges[perm]
            y = predictor(data_pred, edge_batch)
            pred.append(y.detach().cpu().squeeze(1))
        return torch.cat(pred, dim=0)

    pred = torch.cat([get_predictions(args, predictor,data_pos), get_predictions(args, predictor,data_neg)], dim=0)
    true = torch.cat([torch.ones(data_pos.edges.size(0)), torch.zeros(data_neg.edges.size(0))], dim=0)

    return pred, true

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
########################################################################################################################
## start training ######################################################################################################
def main(rank, args, data_pos_train, data_neg_train, data_pos_valid, data_neg_valid, data_pos_test, data_neg_test):
    setup(rank, args.world_size)
    
    # Instead of trying to move the entire graph_prepare object,
    # we need to move its internal tensors/data to the device
    data_pos_train.edges = data_pos_train.edges.to(rank)
    if hasattr(data_pos_train, 'x') and data_pos_train.x is not None:
        data_pos_train.x = data_pos_train.x.to(rank)
    
    # Repeat for other data objects
    data_neg_train.edges = data_neg_train.edges.to(rank)
    data_pos_valid.edges = data_pos_valid.edges.to(rank)
    data_neg_valid.edges = data_neg_valid.edges.to(rank) 
    data_pos_test.edges = data_pos_test.edges.to(rank)
    data_neg_test.edges = data_neg_test.edges.to(rank)

    predictor = Predictor(args).to(rank)
    optimizer = get_optimizer(args, predictor.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    for epoch in range(args.epochs):
        data_neg_train.resample_neg_edges()

        loss = train(rank, args, predictor, optimizer, scheduler, data_pos_train, data_neg_train)
        lr_now = optimizer.param_groups[0]["lr"]
        if lr_now > args.lr_mini:
            scheduler.step()

        if rank == 0 and epoch % args.eval_epoch == args.eval_epoch - 1:
            # with Timing(name='Times of validation: '):
            val_pred, val_true = test(args, predictor.module, data_pos_valid, data_neg_valid)
            test_pred, test_true = test(args, predictor.module, data_pos_test, data_neg_test)
            results = get_eval_result(args, val_pred, val_true, test_pred, test_true)
            for key, result in results.items():
                loggers[key].add_result(run, result)
                valid_res, test_res = result
                to_print = (f'learning rate: {lr_now:.7f}' + '\n'
                            + f': {key}: Run: {run:02d}, Epoch: {epoch:02d}, '
                            + f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, '
                            + f'Test: {100 * test_res:.2f}%')
                # tensorboard_writer.add_scalar('eval/loss', loss, epoch + 1)
                # tensorboard_writer.add_scalar('eval/Valid', valid_res, epoch + 1)
                # tensorboard_writer.add_scalar('eval/Test', test_res, epoch + 1)
                with open(log_file, 'a') as f:
                    print(to_print, file=f)
                    print(to_print)


    for key in loggers.keys():
        to_print = key
        with open(log_file, 'a') as f:
            print(to_print, file=f)
            loggers[key].print_statistics(run, f=f)
            print(to_print)
            loggers[key].print_statistics(run)

    cleanup()

# Launch distributed training
if __name__ == "__main__":
    args = parser.parse_args()
    mp.spawn(main,
             args=(args, data_pos_train, data_neg_train, data_pos_valid, data_neg_valid, data_pos_test, data_neg_test),
             nprocs=args.world_size,
             join=True)

########################################################################################################################
for key in loggers.keys():
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
        print(f'{key}')
        loggers[key].print_statistics()
########################################################################################################################
# end ##################################################################################################################
########################################################################################################################