import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch_sparse
# from horology import Timing
import math

def get_optimizer(args, paras):
    name = args.optimizer
    lr = args.lr
    if name == 'Adam':
        optimizer = torch.optim.Adam(paras, lr=lr)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(paras, lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(paras, lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    return optimizer


def SparseTensor_norm(adj, method: str = 'row_sum'):
    if isinstance(adj, torch_sparse.SparseTensor):
        if method == 'row_sum':
            deg_inv = 1 / (torch_sparse.sum(adj, dim=1) + 1e-15)
            # deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj = torch_sparse.mul(adj, deg_inv.view(-1, 1))
        elif method == 'symmetric':
            deg = torch_sparse.sum(adj, dim=1)
            deg_inv_sqrt = 1/(deg.pow_(0.5) + 1e-15)
            # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
            adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))
    return adj

def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        if self.args.use_node_emb:
            self.emb_node = nn.Embedding(args.num_nodes, args.dim_node_emb)
        if self.args.use_degree:
            self.emb_degree = nn.Embedding(args.max_degree + 1, args.dim_encoding)
        ## embedding huristics
        emb_heuristics = []
        self.num_heuristics = sum([args.use_dist, args.use_cn, args.use_ja, args.use_aa, args.use_ra])
        max_heuristics = max(args.max_dist,args.max_cn,args.max_ja,args.max_aa,args.max_ra)+1
        for i in range(self.num_heuristics):
            emb_heuristics.append(nn.Embedding(max_heuristics, args.dim_encoding))
        self.emb_heuristics = nn.ModuleList(emb_heuristics)
        ## for ppa
        if args.dataset == 'ogbl-ppa':
            self.id_encoder = nn.Embedding(100, args.dim_encoding)
        ## SMGT layer
        ComHGlayers = []
        dim_hidden = args.dim_in if args.dim_hidden is None else args.dim_hidden
        if args.n_layers > 0:
            for i in range(args.n_layers):
                dim_in = args.dim_in if i ==0 else dim_hidden
                ComHGlayers.append(ComHG(dim_in, dim_hidden, n_heads=args.n_heads, bias=args.bias,
                                         residual=args.residual, reduce=args.reduce, atten_type=args.atten_type,
                                         atten_combine=args.atten_combine, dim_atten=args.dim_atten,
                                         negative_slope=args.negative_slope))
        else:
            self.linear_in = nn.Linear(args.dim_in, dim_hidden, bias=args.bias)

        self.ComHGlayers = nn.ModuleList(ComHGlayers)
        self.row_norm = nn.LayerNorm(dim_hidden)
        ## MLP
        dim_hidden += args.dim_encoding * self.num_heuristics
        self.mlp = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden, bias=args.bias) for _ in range(args.n_layers_mlp)])
        self.row_norm_mlp = nn.LayerNorm(dim_hidden)
        self.final_out = nn.ModuleList([nn.Linear(dim_hidden, 64, bias=args.bias), nn.Linear(64, 1, bias=args.bias)])

        self.apply(init_params)

        self.adj_decay = Parameter(torch.Tensor(1))


    def forward(self, graph, edge_batch):
        ## input config #####################################
        edge_batch = edge_batch.to(self.args.device, dtype=torch.int64)
        feats = graph.x
        if self.args.dataset == 'ogbl-ppa':
            feats = self.id_encoder(feats.squeeze().to(self.args.device))

        x = torch.zeros((self.args.num_nodes,1)).to(self.args.device)
        if self.args.use_feature and x != None:
            x = torch.cat([x, feats.to(self.args.device)], dim=1)
        if self.args.use_node_emb:
            node_ids = torch.arange(0, self.args.num_nodes).long().to(self.args.device)
            x = torch.cat([x, self.emb_node(node_ids)], dim=1)
        if self.args.use_degree:
            dg = torch.from_numpy(graph.degree).squeeze().long().to(self.args.device)
            x = torch.cat([x, self.emb_degree(dg)], dim=1)
        if x.size(1) > 1:
            x =x[:,1:]

        ## adj config #####################################
        adj = graph.adj_gnn.to(self.args.device)
        values = adj.storage._value
        if self.args.adj_weight == 'decay':# exponential decay
            self.adj_decay.data = self.adj_decay.clamp(0.1, 1.5)
            values = torch.exp(-1 * values * self.adj_decay)
        elif self.args.adj_weight == 'same':
            values = values.new_zeros(len(values)) + 1
        else:
            raise ValueError((f'adj_weight must be options: same, decay'))
        values = F.dropout(values, p=self.args.dropout_adj, training=self.training)
        adj.storage._value = values
        adj = SparseTensor_norm(adj)
        # ## adj config #####################################
        # adj = graph.adj_gnn.to(self.args.device)
        # adj = SparseTensor_norm(adj)

        ## start learning ##################################
        if len(self.ComHGlayers)>0 and x.size(1) > 1:
            for layer in self.ComHGlayers[:-1]:
                x = layer(x, adj)
                if self.args.use_layer_norm:
                    x = self.row_norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.ComHGlayers[-1](x, adj)
            # x = F.relu(x)
        elif x.size(1) > 1:
            x = self.linear_in(x)
            # x = F.relu(x)
        ## use distance, common neighbors and other heuristics
        x = x[edge_batch[:, 0]] * x[edge_batch[:, 1]]
        for i in range(self.num_heuristics):
            x = torch.cat([x, self.emb_heuristics[i](edge_batch[:, i+2])], dim=1)

        ## link prediction
        for linear in self.mlp:
            x = linear(x)
            if self.args.use_layer_norm:
                x = self.row_norm_mlp(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_out[0](x)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_out[1](x)
        x = torch.sigmoid(x)
        return x

class ComHG_attention(nn.Module):
    def __init__(self, dim_in, dim_out, atten_type, atten_combine, dim_atten: int=2, negative_slope: float=0.2):
        super().__init__()
        self.atten_type = atten_type
        self.atten_combine = atten_combine
        self.negative_slope = negative_slope
        self.square_d = torch.rsqrt(torch.tensor(dim_atten))

        self.linear_row = nn.Linear(dim_in, dim_atten, bias = False)
        self.linear_col = nn.Linear(dim_in, dim_atten, bias=False)
        if self.atten_type == 'Concat':
            self.linear_concat = nn.Linear(dim_atten*2, 1, bias=False)

        self.linear_x = nn.Linear(dim_in, dim_out)

    def forward(self, x, adj):
        if isinstance(adj, torch_sparse.SparseTensor):
            if self.atten_type in ['Concat', 'Cosine', 'Multiply']:
                a_row, a_col = self.linear_row(x), self.linear_col(x)
                row, col = adj.storage._row, adj.storage._col
                if self.atten_type == 'Concat':
                    # Concat based attention mechanism, see more in GAT
                    atten = torch.cat((a_row[row], a_col[col]), dim=1)
                    atten = self.linear_concat(atten)
                elif self.atten_type == 'Cosine':
                    # cosine similarity-based attention mechanism like AGNN
                    atten = F.cosine_similarity(a_row[row], a_col[col], dim=1)
                elif self.atten_type == 'Multiply':
                    # self-attention mechanism like Transformer
                    atten = a_row[row] * a_col[col]
                    atten = atten.sum(dim=1) * self.square_d
                # softmax attention
                atten = F.leaky_relu(atten, self.negative_slope)
                atten = atten.squeeze()
                atten = atten - atten.max()
                atten = atten.exp()
                atten = adj.set_value(atten, layout='coo')
                atten = SparseTensor_norm(atten)
                # combine attention with original adj weight.
                if self.atten_combine == 'plus':
                    atten = (adj.storage._value + atten.storage._value)/2
                    atten = adj.set_value(atten, layout='coo')
                elif self.atten_combine == 'multiply':
                    atten = (adj.storage._value * atten.storage._value)
                    atten = adj.set_value(atten, layout='coo')
                    atten = SparseTensor_norm(atten)
                elif self.atten_combine == 'only_atten':
                    atten = atten
                else:
                    raise ValueError((f'atten_combine must be: plus, multiply or only_atten'))
            elif self.atten_type == 'no_atten':
                atten = adj
            else:
                raise ValueError((f'atten_type must be: Concat, Cosine, Multiply or no_atten'))

            out = torch_sparse.matmul(atten, x, reduce='sum')
            out = self.linear_x(out)
        else:
            raise ValueError((f'adj must be torch_sparse.SparseTensor in this version.'))

        return out


class ComHG(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_heads: int = 1, bias: bool = True,
                 residual: bool = True, reduce: str = 'add', atten_type: str = 'Multiply',
                 atten_combine: str = 'no_use', dim_atten: int=2, negative_slope: float=0.2):
        super().__init__()

        if residual:
            self.linear_residual = nn.Linear(dim_in, dim_out, bias=bias)
        self.multihead_layer = nn.ModuleList([ComHG_attention(dim_in, dim_out, atten_type, atten_combine,
                                                              dim_atten, negative_slope) for _ in range(n_heads)])

        self.residual = residual
        self.reduce = reduce
        self.n_heads = n_heads + 1 if residual else n_heads

        if reduce == 'concat':
            self.linear = nn.Linear(dim_out * self.n_heads, dim_out, bias=bias)
        elif reduce == 'add':
            self.linear = nn.Linear(dim_out, dim_out, bias=bias)
        else:
            raise ValueError('args.reduce is error. must be: concat or add')

    def forward(self, x, adj):
        y = [self.linear_residual(x)] if self.residual else []
        for layer in self.multihead_layer:
            y.append(layer(x, adj))
        if len(y)>1:
            if self.reduce == 'concat':
                y = torch.cat(y, dim=-1)
                y = self.linear(y)
            elif self.reduce == 'add':
                out = y[0]
                for i in range(1,len(y)):
                    out += y[i]
                y = self.linear(out)

        return y


class HierarchicalComHG(Predictor):
    def __init__(self, args):
        super().__init__(args)
        
        # Get the actual dimension being used (from parent class logic)
        dim_hidden = args.dim_in if args.dim_hidden is None else args.dim_hidden
        
        # Additional components for hierarchical model
        self.text_encoder = nn.Linear(args.text_dim, args.dim_encoding)
        self.relation_encoder = nn.Linear(args.relation_dim, args.dim_encoding)
        
        # Local and Global Attention (in addition to existing ComHG layers)
        self.local_attention = nn.ModuleList([
            MultiHeadAttention(args.dim_encoding, args.n_heads) 
            for _ in range(args.n_layers)
        ])
        
        self.global_attention = nn.ModuleList([
            MultiHeadAttention(args.dim_encoding, args.n_heads)
            for _ in range(args.n_layers)
        ])
        
        # Community-based components
        self.community_encoder = nn.Linear(args.num_communities, args.dim_encoding)
        
        # Additional MLP for combining hierarchical features
        self.hierarchical_mlp = nn.Sequential(
            nn.Linear(dim_hidden * 3, dim_hidden * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(dim_hidden * 2, dim_hidden)
        )

    def forward(self, graph, edge_batch):
        # Original ComHG processing
        x_original = super().forward(graph, edge_batch)
        
        # Additional hierarchical processing
        if hasattr(graph, 'text_features') and hasattr(graph, 'communities'):
            # Text embedding processing
            text_emb = self.text_encoder(graph.text_features)
            
            # Local attention
            local_features = text_emb
            for layer in self.local_attention:
                local_features = layer(local_features, graph.adj)
            
            # Global community-based attention
            community_emb = scatter_mean(text_emb, graph.communities, dim=0)
            global_features = text_emb
            for layer in self.global_attention:
                global_features = layer(global_features, community_emb)
            
            # Community encoding
            community_features = self.community_encoder(
                F.one_hot(graph.communities, num_classes=self.args.num_communities).float()
            )
            
            # Combine features
            combined_features = torch.cat([
                x_original,
                local_features[edge_batch[:, 0]] * local_features[edge_batch[:, 1]],
                global_features[edge_batch[:, 0]] * global_features[edge_batch[:, 1]]
            ], dim=-1)
            
            # Final prediction combining original and hierarchical features
            x = self.hierarchical_mlp(combined_features)
            return torch.sigmoid(x)
        
        # Fallback to original prediction if hierarchical features aren't available
        return x_original

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_k = dim_in // n_heads
        
        self.Q = nn.Linear(dim_in, dim_in)
        self.K = nn.Linear(dim_in, dim_in)
        self.V = nn.Linear(dim_in, dim_in)
        
        self.out = nn.Linear(dim_in, dim_in)
        
    def forward(self, x, adj, mask=None):
        B = x.size(0)
        N = x.size(0) if len(x.size()) == 2 else x.size(1)
        
        # Reshape for multi-head attention
        Q = self.Q(x).view(B, N, self.n_heads, self.dim_k)
        K = self.K(x).view(B, N, self.n_heads, self.dim_k)
        V = self.V(x).view(B, N, self.n_heads, self.dim_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply attention
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        
        # Reshape and project output
        out = out.reshape(B, N, -1)
        return self.out(out)

# Add these arguments to your argument parser
def add_hierarchical_args(parser):
    parser.add_argument('--text_dim', type=int, default=768,
                      help='Dimension of text embeddings (e.g., BERT)')
    parser.add_argument('--relation_dim', type=int, default=100,
                      help='Dimension of relation embeddings')
    parser.add_argument('--num_communities', type=int, default=100,
                      help='Number of communities from Leiden algorithm')
    #parser.add_argument('--n_heads', type=int, default=4,
    #                  help='Number of attention heads')
    return parser

