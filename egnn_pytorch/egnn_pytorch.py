import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# types

from typing import Optional, List, Union

# pytorch geometric

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
    PYG_AVAILABLE = True
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    
    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

# helper functions

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
    return x

# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# helper classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.fn = nn.LayerNorm(1)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        phase = self.fn(norm)
        return (phase * normed_coors)

# classes

class EGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        num_nearest_neighbors = 0,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = False,
        norm_coors = False,
        update_feats = True,
        update_coors = True,
        only_sparse_neighbors = False,
        valid_radius = float('inf'),
        m_pool_method = 'sum',
        soft_edges = False,
        coor_weights_clamp_value = None
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        ) if soft_edges else None

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            nn.Linear(dim * 2, dim),
        ) if update_feats else None

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.coor_weights_clamp_value = coor_weights_clamp_value

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(self, feats, coors, edges = None, mask = None, adj_mat = None):
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = *feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        use_nearest = num_nearest > 0 or only_sparse_neighbors

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        i = j = n

        if use_nearest:
            ranking = rel_dist[..., 0].clone()

            if exists(mask):
                rank_mask = mask[:, :, None] * mask[:, None, :]
                ranking.masked_fill_(~rank_mask, 1e5)

            if exists(adj_mat):
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

                if only_sparse_neighbors:
                    num_nearest = int(adj_mat.float().sum(dim = -1).max().item())
                    valid_radius = 0

                self_mask = rearrange(torch.eye(n, device = device, dtype = torch.bool), 'i j -> () i j')

                adj_mat = adj_mat.masked_fill(self_mask, False)
                ranking.masked_fill_(self_mask, -1.)
                ranking.masked_fill_(adj_mat, 0.)

            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim = -1, largest = False)

            nbhd_mask = nbhd_ranking <= valid_radius

            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim = 2)

            if exists(edges):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            j = num_nearest

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        if use_nearest:
            feats_j = batched_index_select(feats, nbhd_indices, dim = 1)
        else:
            feats_j = rearrange(feats, 'b j d -> b () j d')

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        if exists(self.edge_gate):
            m_ij = m_ij * self.edge_gate(m_ij)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
                mask = (mask_i * mask_j) & nbhd_mask
            else:
                mask_j = rearrange(mask, 'b j -> b () j')
                mask = mask_i * mask_j

        if exists(self.coors_mlp):
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)

            if exists(mask):
                coor_weights.masked_fill_(~mask, 0.)

            if exists(self.coor_weights_clamp_value):
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max = clamp_value)

            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors
        else:
            coors_out = coors

        if exists(self.node_mlp):
            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

            if self.m_pool_method == 'mean':
                if exists(mask):
                    # masked mean
                    mask_sum = m_ij_mask.sum(dim = -2)
                    m_i = safe_div(m_ij.sum(dim = -2), mask_sum)
                else:
                    m_i = m_ij.mean(dim = -2)

            elif self.m_pool_method == 'sum':
                m_i = m_ij.sum(dim = -2)

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim = -1)
            node_out = self.node_mlp(node_mlp_input) + feats
        else:
            node_out = feats

        return node_out, coors_out

class EGNN_Network(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        num_tokens = None,
        num_edge_tokens = None,
        num_positions = None,
        edge_dim = 0,
        num_adj_degrees = None,
        adj_dim = 0,
        **kwargs
    ):
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.num_positions = num_positions

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = edge_dim > 0

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = edge_dim if self.has_edges else 0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EGNN(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs))

    def forward(self, feats, coors, adj_mat = None, edges = None, mask = None):
        b, device = feats.shape[0], feats.device

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.pos_emb):
            n = feats.shape[1]
            assert n <= self.num_positions, f'given sequence length {n} must be less than the number of positions {self.num_positions} set at init'
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        if exists(edges) and exists(self.edge_emb):
            edges = self.edge_emb(edges)

        # create N-degrees adjacent matrix from 1st degree connections
        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        for layer in self.layers:
            feats, coors = layer(feats, coors, adj_mat = adj_mat, edges = edges, mask = mask)

        return feats, coors

class EGNN_sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        soft_edge = 0,
        norm_feats = False,
        norm_coors = False,
        update_feats = True,
        update_coors = True, 
        dropout = 0.,
        coor_weights_clamp_value = None, 
        aggr = "add",
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                         nn.Sigmoid()
        ) if soft_edge else None

        # NODES
        self.node_norm = nn.LayerNorm(feats_dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None, 
                angle_data: List = None,  size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (n_edges, 2)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                           coors=coors, rel_coors=rel_coors)
        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp( torch.cat([x_i, x_j, edge_attr], dim=-1) )
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max = clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"])
            hidden_out = self.node_mlp( torch.cat([hidden_feats, m_i], dim = -1) )
            hidden_out = kwargs["x"] + hidden_out
        else: 
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__) 


class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """
    def __init__(self, n_layers, feats_dim, pos_dim = 3,
                       edge_attr_dim = 0, m_dim = 16,
                       fourier_features = 0,
                       embedding_nums=[], embedding_dims=[],
                       edge_embedding_nums=[], edge_embedding_dims=[],
                       update_coors=True, update_feats=True, 
                       norm_feats=True, norm_coors=False, 
                       coor_weights_clamp_value=None, recalc=0 ):
        super().__init__()

        self.n_layers         = n_layers 

        # Embeddings? solve here
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers      = nn.ModuleList()
        self.feats_dim        = feats_dim
        self.pos_dim          = pos_dim
        self.edge_attr_dim    = edge_attr_dim
        self.m_dim            = m_dim
        self.fourier_features = fourier_features
        self.norm_feats       = norm_feats
        self.update_feats     = update_feats
        self.update_coors     = update_coors
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc           = recalc
        
        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_sparse(feats_dim = feats_dim,
                                pos_dim = pos_dim,
                                edge_attr_dim = edge_attr_dim,
                                m_dim = m_dim,
                                fourier_features = fourier_features, 
                                norm_feats = norm_feats,
                                norm_coors = norm_coors,
                                update_feats = update_feats,
                                update_coors = update_coors, 
                                coor_weights_clamp_value = coor_weights_clamp_value)
            self.mpnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        # regulates wether to embedd edges each layer
        edges_need_embedding = True  
        for i,layer in enumerate(self.mpnn_layers):
            
            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            # pass layers
            x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = True
            
        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))
