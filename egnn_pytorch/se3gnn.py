import numba
import numpy as np

import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

# types

from typing import Optional, List, Union
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

# own module
from egnn_pytorch.egnn_pytorch import *



def get_angle(c1, c2, c3):
    """ Returns the angle in radians.
        Inputs: 
        * ci: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1), 
                        (u1*u2).sum(dim=-1) )

def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * ci: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )

@numba.jit(nopython=True)
def get_sparse_adj_paths(idxs, n=3):
    """ Returns the adjacency matrices of order 1...N. 
        Many simplifications - only works for a max of n=3 levels
        Inputs: 
        * idxs: (2, N) indices of adjacent positions in the sparse tensor.
        Outputs: 
        * paths: (n, path_length, Ni): list with adjacency paths at each level.
                path_length = degree-1 + 2 ( connection + extremes(ij) ) 
        * uniques: (n-1, Ni). long tensor - corresponding unique ij combination
    """
    paths = [ idxs ] # adj paths. [0],[-1] are the bonded ones in nth degree
    uniques = [  ] # correspondence of path to unique ij pair
    rows, cols = idxs
    size = int(rows.max())+1
    
    for deg in range(2, n+1):
        res = []
        uni = []
        unique = 0 # unique ij pair
        for i in range(size):
            row_idxs = paths[-1][0][paths[-1][-1] == i] # latest degree connections
            for j in range(size):
                col_idxs = cols[rows == j] # basic adjacency
                common = [r for r in row_idxs for c in col_idxs if i != r == c != j]
                # info about degree paths
                new_unique = 0
                for com in common: 
                    his = [com]
                    # elongate previous path with new info
                    if deg > 2: 
                        ext = np.argwhere((paths[-1][0]==i) * \
                                          (paths[-1][-1]==com) * 
                                          ((paths[-1][1:-1]==com).sum(axis=0)==0) ).flatten()
                        his = [ [ paths[-1][i][ex] for i in range(1, deg-1) ] + his for ex in ext]
                    else: 
                        his = [his]
                    # check no repeats: length = deg-1 + 2 (bridge + ij)
                    his = [[i]+his_i+[j] for his_i in his]
                    his = [his_i for his_i in his if len(set(his_i)) == deg+1]
                    res.extend(his)
                    uni.extend([unique]*len(his))
                    new_unique += len(his)
                # update counter of unique ij combinations
                if new_unique:
                    unique += 1
                 
        # avoid getitem of empty array which throws error in numba               
        paths.append(np.array(res).T if unique > 0 else np.empty((deg+1, 0), dtype=np.int64))
        uniques.append(np.array(uni))
        
    return paths, uniques

###############
# SE3 wrapper #
###############

class SE3GNN_sparse(EGNN_sparse):
    """ Adds ability to distinguish mirrored point clouds. """
    def __init__(self, angles = 3, **kwargs):
        super(SE3GNN_sparse, self).__init__(**kwargs)

        self.angles = angles

        # add angles number bc bond type is passed as 1-hot encoding
        self.edge_input_dim += int(self.angles)

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, self.m_dim),
            SiLU()
        )

        self.angles_lin = nn.Linear(int(self.angles),  
                                    self.edge_input_dim-(self.feats_dim * 2)) if angles else None

        self.apply(self.init_)

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

        # get angle features if we need to
        if angle_data is not None and self.angles:
            angle_data, uniques = angle_data # () | ()
            # aggr edges 
            adj_type = torch.zeros(self.angles, coors.shape[0], coors.shape[0],  device = feats.device)
            for level in range(self.angles):
                adj_type[level, angle_data[level][0], angle_data[level][-1]] = 1.
            # get idxs
            edge_index = torch.nonzero( adj_type.sum(dim=0) > 0 ).t() # (2, N)
            # get attrs specific to each bond degree
            bond_type = adj_type[:, edge_index[0], edge_index[1]].t() # (N, self.angles)
            angles    = torch.zeros_like(torch.stack([bond_type[..., -1]]*3, dim=-1))
            padded_edge_attr = torch.zeros(angles.shape[0], edge_attr.shape[-1], device = feats.device)
            for level in range(self.angles): 
                edge_mask = bond_type.t()[level]>0.
                # add attrs to true bonds
                if level == 0: 
                    padded_edge_attr[edge_mask] = edge_attr 
                # encode angles
                if level == 1: 
                    p1, p2, p3, = angle_data[level]
                    aux_angle = get_angle(coors[p1], coors[p2], coors[p3])
                    aux_angle = scatter_mean(aux_angle, uniques[-2])
                    angles[edge_mask, 0] = torch.cos( aux_angle )
                # encode dihedrals
                if level == 2: 
                    p1, p2, p3, p4 = angle_data[level]
                    # angle between p1-p4 with vertex in the mean of p2-p3
                    aux_angle = get_angle(coors[p1], 0.5*(coors[p2]+coors[p3]), coors[p4])
                    aux_angle = scatter_mean(aux_angle, uniques[-1])
                    angles[edge_mask, 0] = torch.cos( aux_angle )
                    # dihedral (p1-p2-p3 and p2-p3-p4)
                    aux_angle = get_dihedral(coors[p1], coors[p2], coors[p3], coors[p4])
                    aux_angle = scatter_mean(aux_angle, uniques[-1])
                    sin, cos = torch.sin(aux_angle), torch.cos(aux_angle)
                    angles[edge_mask, 1] = sin
                    angles[edge_mask, 2] = cos

            edge_attr = torch.cat([padded_edge_attr, bond_type], dim=-1) # (N, dims)

        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        # add angles to edges
        if angle_data is not None and self.angles: 
            edge_attr_feats += self.angles_lin(angles)

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                           coors=coors, rel_coors=rel_coors)
        return torch.cat([coors_out, hidden_out], dim=-1)


    def __repr__(self):
        dict_print = {}
        return "SE3 + E(n)-GNN Layer for Graphs " + str(self.__dict__) 