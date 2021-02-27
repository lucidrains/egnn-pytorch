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
            nn.ReLU(),
            nn.Linear(edge_input_dim * 2, m_dim)
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1)
        )

        self.hidden_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, feats, coors, edges = None):
        b, n, d, fourier_features = *feats.shape, self.fourier_features

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(dim = -1, keepdim = True)

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

        hidden_mlp_input = torch.cat((feats, m_i), dim = -1)
        hidden_out = self.hidden_mlp(hidden_mlp_input)

        return hidden_out, coors_out


class EGNN_sparse(MessagePassing):
    def __init__(
        self,
        feats_dim,
        pos_dim = 3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0
    ):
        super().__init__()
        self.fourier_features = fourier_features
        self.pos_dim = pos_dim

        edge_input_dim = (fourier_features * 2) + (feats_dim * 2) + edge_attr_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_input_dim * 2, m_dim)
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1)
        )

        self.hidden_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            nn.ReLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
        """
        coors, x = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = torch.norm(rel_coors, dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)

        if edge_attr is None:
            edge_attr = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr = rel_dist

        coors_out, hidden_out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                                           coors=coors, rel_coors=rel_coors)
        return torch.cat([coors_out, hidden_out], dim=-1)


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
        coors_out  = kwargs["coors"] +  coor_wi + coor_ri
        hidden_out = self.hidden_mlp( torch.cat([kwargs["x"], m_i], dim = -1) )

        return self.update((hidden_out, coors_out), **update_kwargs)
        
    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(dict_print) 

# attention version

class EGAT(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        heads = 4,
        dim_head = 64,
        fourier_features = 0
    ):
        super().__init__()
        self.fourier_features = fourier_features

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        edge_input_dim = (fourier_features * 2) + (dim_head * 2) + edge_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_input_dim * 2, m_dim)
        )

        self.to_attn_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1),
            Rearrange('... () -> ...')
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1),
            Rearrange('... () -> ...')
        )

    def forward(self, feats, coors, edges = None):
        b, n, d, h, fourier_features, device = *feats.shape, self.heads, self.fourier_features, feats.device

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(dim = -1, keepdim = True)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        rel_dist = repeat(rel_dist, 'b i j d -> b h i j d', h = h)

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # expand queries and keys for concatting

        q = repeat(q, 'b h i d -> b h i n d', n = n)
        k = repeat(k, 'b h j d -> b h n j d', n = n)

        edge_input = torch.cat((q, k, rel_dist), dim = -1)

        if exists(edges):
            edges = repeat(edges, 'b i j d -> b h i j d', h = h)
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        mask_value = -torch.finfo(coor_weights.dtype).max

        mask = rearrange(torch.eye(n).bool(), 'i j -> () () i j')
        coor_weights.masked_fill_(mask, mask_value)
        coors_attn = coor_weights.softmax(dim = -1)

        coors_out = einsum('b h i j, b i j c -> b i c', coors_attn, rel_coors) + coors

        # derive attention

        sim = self.to_attn_mlp(m_ij)
        attn = sim.softmax(dim = -1)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out
