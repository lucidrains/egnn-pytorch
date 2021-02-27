import torch
from egnn_pytorch.utils import rot
from egnn_pytorch import EGNN, EGNN_sparse, EGAT

def test_egnn_equivariance():
    layer = EGNN(dim = 512, edge_dim = 4)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = layer(feats, coors @ R + T, edges)
    feats2, coors2 = layer(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_geom_equivalence():
    layer = EGNN_sparse(feats_dim=128,
                    pos_dim = 3,
                    edge_attr_dim = 0,
                    m_dim = 16,
                    fourier_features = 0)

    feats = torch.randn(16, 128)
    coors = torch.randn(16, 3)
    x     = torch.cat([coors, feats], dim=-1)
    edge_idxs   = (torch.rand(2, 20) * 16 ).long()
    edges_attrs = torch.randn(16, 16, 4)
    edges_attrs = edges_attrs[edge_idxs[0], edge_idxs[1]] 

    assert layer.forward(x, edge_idxs, edge_attr=edges_attrs).shape == x.shape

def test_egat_equivariance():
    layer = EGAT(dim = 512, edge_dim = 4)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = layer(feats, coors @ R + T, edges)
    feats2, coors2 = layer(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'
