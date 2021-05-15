import os
import sys
sys.path.append("../egnn_pytorch")

import torch

from egnn_pytorch.utils import rot
from egnn_pytorch import EGNN, EGNN_sparse, get_sparse_adj_paths, SE3GNN_sparse


def test_geom_angles():
    layer = SE3GNN_sparse(feats_dim=128,
                          edge_attr_dim = 4,
                          m_dim = 32,
                          fourier_features = 4, 
                          angles = 3)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)
    apply_action = lambda t: (t @ R + T).squeeze()

    # example with aspirin
    edge_idxs = torch.tensor([[0, 2, 1, 2, 3, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 4, 9, 10, 10, 11, 10, 12], 
                              [2, 0, 2, 1, 2, 3, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 4, 9, 10, 9, 11, 10, 12, 10]])

    feats = torch.randn(13, 128)
    coors = torch.randn(13, 3)

    edges_attrs = torch.randn(13, 13, 4)
    edges_attrs = edges_attrs[edge_idxs[0], edge_idxs[1]] 

    angle_data = get_sparse_adj_paths(edge_idxs.numpy())
    angle_data = [[torch.tensor(x).long() for x in angle_data[0]], 
                  [torch.tensor(x).long() for x in angle_data[1]]]

    x1 = torch.cat([coors, feats], dim=-1)
    x2 = torch.cat([apply_action(coors), feats], dim=-1)
    x3 = torch.cat([coors*torch.tensor([[-1., 1., 1.]]), feats], dim=-1)

    batch = torch.zeros(x1.shape[0]).long()

    out1 = layer(x=x1, edge_index=edge_idxs, edge_attr=edges_attrs, batch=batch, angle_data=angle_data)
    out2 = layer(x=x2, edge_index=edge_idxs, edge_attr=edges_attrs, batch=batch, angle_data=angle_data)
    out3 = layer(x=x3, edge_index=edge_idxs, edge_attr=edges_attrs, batch=batch, angle_data=angle_data)

    feats1, coors1 = out1[:, 3:], out1[:, :3]
    feats2, coors2 = out2[:, 3:], out2[:, :3]
    feats3, coors3 = out3[:, 3:], out3[:, :3]

    print("1-2 feats",  feats1 - feats2, "maxdiff", (feats1 - feats2).abs().max())
    print("1-3 feats",  feats1 - feats3, "maxdiff", (feats1 - feats2).abs().max() )
    print("\n////\n")
    print( "1-2", apply_action(coors1) -  coors2)
    print( "1-3", (coors1 - coors3*torch.tensor([[-1., 1., 1.]])).t() )
    assert torch.allclose(feats1, feats2, atol=1e-4), 'features must be invariant'
    assert torch.allclose(apply_action(coors1), coors2, atol=1e-4), 'coordinates must be equivariant'

    assert not torch.allclose( feats1, feats3, atol=1e-4 ), 'features must NOT be invariant'
    assert not torch.allclose( coors1, coors3*torch.tensor([[-1., 1., 1.]]), atol=1e-4 ), 'coordinates must NOT be equivariant'


