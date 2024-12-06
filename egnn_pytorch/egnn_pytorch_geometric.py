import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# types

from typing import Optional, List, Union

# pytorch geometric

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    
    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

from .egnn_pytorch import *

# global linear attention

class Attention_Sparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None: 
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1: 
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze() # get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi,n_idxs in zip(*batch_uniques):
                x_list.append( 
                    self.sparse_forward(
                        x[aux_count:aux_count+n_i], 
                        context[aux_count:aux_count+n_idxs],
                        batch_uniques = (bi.unsqueeze(-1), n_idxs.unsqueeze(-1)) 
                    ) 
                )
            return torch.cat(x_list, dim=0)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = torch_geomtric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geomtric.nn.norm.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim, heads, dim_head)
        self.attn2 = Attention_Sparse(dim, heads, dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geomtric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask = mask)
        out     = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x =  out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries


# define pytorch-geometric equivalents

class EGNN_Sparse(MessagePassing):
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
        norm_coors_scale_init = 1e-2,
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
        super(EGNN_Sparse, self).__init__(**kwargs)
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

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1. 
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

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
            * edge_index: (2, n_edges)
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
                                                           coors=coors, rel_coors=rel_coors, 
                                                           batch=batch)
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
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        
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

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
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
    def __init__(self, n_layers, feats_dim, 
                 pos_dim = 3,
                 edge_attr_dim = 0, 
                 m_dim = 16,
                 fourier_features = 0, 
                 soft_edge = 0,
                 embedding_nums=[], 
                 embedding_dims=[],
                 edge_embedding_nums=[], 
                 edge_embedding_dims=[],
                 update_coors=True, 
                 update_feats=True, 
                 norm_feats=True, 
                 norm_coors=False,
                 norm_coors_scale_init = 1e-2, 
                 dropout=0.,
                 coor_weights_clamp_value=None, 
                 aggr="add",
                 global_linear_attn_every = 0,
                 global_linear_attn_heads = 8,
                 global_linear_attn_dim_head = 64,
                 num_global_tokens = 4,
                 recalc=0 ,):
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
        self.soft_edge        = soft_edge
        self.norm_feats       = norm_feats
        self.norm_coors       = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats     = update_feats
        self.update_coors     = update_coors
        self.dropout          = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc           = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))
        
        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_Sparse(feats_dim = feats_dim,
                                pos_dim = pos_dim,
                                edge_attr_dim = edge_attr_dim,
                                m_dim = m_dim,
                                fourier_features = fourier_features, 
                                soft_edge = soft_edge, 
                                norm_feats = norm_feats,
                                norm_coors = norm_coors,
                                norm_coors_scale_init = norm_coors_scale_init, 
                                update_feats = update_feats,
                                update_coors = update_coors, 
                                dropout = dropout, 
                                coor_weights_clamp_value = coor_weights_clamp_value)

            # global attention case
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention(dim = self.feats_dim, 
                                                   heads = global_linear_attn_heads, 
                                                   dim_head = global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            # normal case
            else: 
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

            # attn tokens
            global_tokens = None
            if exists(self.global_tokens):
                unique, amounts = torch.unique(batch, return_counts)
                num_idxs = torch.cat([torch.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            # pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else: 
                # only pass feats to the attn layer
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                # merge attn-ed feats and coords
                x = torch.cat( (x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = True
            
        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))
