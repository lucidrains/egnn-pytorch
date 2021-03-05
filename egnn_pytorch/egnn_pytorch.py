import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch_geometric.nn import MessagePassing
# types
from typing import Optional, List, Union
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

# helper functions

def exists(val):
    return val is not None

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# classes

class EGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0
    ):
        super().__init__()
        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, feats, coors, edges = None):
        b, n, d, fourier_features = *feats.shape, self.fourier_features

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        feats_i = repeat(feats, 'b i d -> b i n d', n = n)
        feats_j = repeat(feats, 'b j d -> b n j d', n = n)
        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors

        m_i = m_ij.sum(dim = -2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors_out


class EGNN_sparse(MessagePassing):
    def __init__(
        self,
        feats_dim,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0
    ):
        super().__init__()
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim

        edge_input_dim = (fourier_features * 2) + (feats_dim * 2) + edge_attr_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
        """
        x, coors = x[:, :self.feats_dim], x[:, self.feats_dim:]
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                                           coors=coors, rel_coors=rel_coors)
        return torch.cat([hidden_out, coors_out], dim=-1)


    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp( torch.cat([x_i, x_j, edge_attr], dim=-1) )
        coor_w = self.coors_mlp(m_ij)
        return m_ij, coor_w

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
        m_ij, coor_wij = self.message(**msg_kwargs)
        # aggregate them
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        m_i     = self.aggregate(m_ij, **aggr_kwargs)
        coor_wi = self.aggregate(coor_wij, **aggr_kwargs)
        coor_ri = self.aggregate(kwargs["rel_coors"], **aggr_kwargs)
        # return tuple
        update_kwargs = self.inspector.distribute('update', coll_dict)

        node, coors = kwargs["x"], kwargs["coors"]
        coors_out  = coors + ( coor_wi * coor_ri )

        hidden_out = self.node_mlp( torch.cat([node, m_i], dim = -1) )
        hidden_out = hidden_out + node

        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(dict_print) 


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
        * recalc: bool. Whether to recalculate edge features between MPNN layers.
        * verbose: bool. verbosity level.
    """
    def __init__(self, n_layers, feats_dim, pos_dim = 3,
                       edge_attr_dim = 0, m_dim = 16,
                       fourier_features = 0,
                       embedding_nums=[], embedding_dims=[],
                       edge_embedding_nums=[], edge_embedding_dims=[],
                       recalc=True, verbose=False):
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
            feats_x_in += embedding_dims[i] - 1
            feats_x_out += embedding_dims[i] - 1

        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            feats_edge_in += edge_embedding_dims[i] - 1
            feats_edge_out += edge_embedding_dims[i] - 1

        # rest
        self.mpnn_layers      = nn.ModuleList()
        self.feats_dim        = feats_dim
        self.pos_dim          = pos_dim
        self.edge_attr_dim    = edge_attr_dim
        self.m_dim            = m_dim
        self.fourier_features = fourier_features
        self.recalc           = recalc
        self.verbose          = verbose
        
        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_sparse(feats_dim = feats_dim,
                                pos_dim = pos_dim,
                                edge_attr_dim = edge_attr_dim,
                                m_dim = m_dim,
                                fourier_features = fourier_features)
            self.mpnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Embedding of inputs when necessary, then pass layers.
            Recalculate edge features every time with the
            `recalc_edge` function if self.recalc_edge is set.
        """
        # pick to embedd. embedd sequentially and add to input - points:
        to_embedd = x[:, -len(self.embedding_dims):].long()
        for i,emb_layer in enumerate(self.emb_layers):
            # the portion corresponding to `to_embedd` part gets dropped
            # at first iter
            stop_concat = -len(self.embedding_dims) if i == 0 else x.shape[-1]
            x = torch.cat([ x[:, :stop_concat], 
                            emb_layer( to_embedd[:, i] ) 
                          ], dim=-1)
        # pass layers
        for i,layer in enumerate(self.mpnn_layers):
            # embedd edge items (needed everytime since edge_attr and idxs
            # are recalculated every pass)
            to_embedd = edge_attr[:, -len(self.edge_embedding_dims):].long()
            for i,edge_emb_layer in enumerate(self.edge_emb_layers):
                # the portion corresponding to `to_embedd` part gets dropped
                # at first iter
                stop_concat = -len(self.edge_embedding_dims) if i == 0 else x.shape[-1]
                edge_attr = torch.cat([ edge_attr[:, :-len(self.edge_embedding_dims) + i], 
                                        edge_emb_layer( to_embedd[:, i] ) 
                              ], dim=-1)
            # pass layers
            x = layer(x, edge_index, edge_attr, size=bsize)

            # recalculate edge info - not needed if last layer
            if i < len(self.mpnn_layers)-1 and self.recalc:
                edge_attr, edge_index, _ = recalc_edge(x.detach()) # returns attr, idx, embedd_info
            
            if verbose:
                print("========")
                print(i, "layer, nlinks:", edge_attr.shape)
            
        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))
