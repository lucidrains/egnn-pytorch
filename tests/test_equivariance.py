import torch

from egnn_pytorch import EGNN, EGNN_Sparse
from egnn_pytorch.utils import rot

torch.set_default_dtype(torch.float64)

def test_egnn_equivariance():
    layer = EGNN(dim=512, edge_dim=4)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)
    mask = torch.ones(1, 16).bool()

    # Cache first two nodes' features
    node1 = feats[:, 0, :]
    node2 = feats[:, 1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = feats.clone().detach()
    feats_permuted_row_wise[:, 0, :] = node2
    feats_permuted_row_wise[:, 1, :] = node1

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)
    feats2, coors2 = layer(feats, coors, edges, mask=mask)
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'

def test_higher_dimension():
    layer = EGNN(dim=512, edge_dim=4)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 5)
    edges = torch.randn(1, 16, 16, 4)
    mask = torch.ones(1, 16).bool()

    feats, coors = layer(feats, coors, edges, mask = mask)
    assert True

def test_egnn_equivariance_with_nearest_neighbors():
    layer = EGNN(dim=512, edge_dim=1, num_nearest_neighbors=8)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 256, 512)
    coors = torch.randn(1, 256, 3)
    edges = torch.randn(1, 256, 256, 1)
    mask = torch.ones(1, 256).bool()

    # Cache first two nodes' features
    node1 = feats[:, 0, :]
    node2 = feats[:, 1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = feats.clone().detach()
    feats_permuted_row_wise[:, 0, :] = node2
    feats_permuted_row_wise[:, 1, :] = node1

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)
    feats2, coors2 = layer(feats, coors, edges, mask=mask)
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'


def test_egnn_equivariance_with_coord_norm():
    layer = EGNN(dim=512, edge_dim=1, num_nearest_neighbors=8, norm_coors=True)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 256, 512)
    coors = torch.randn(1, 256, 3)
    edges = torch.randn(1, 256, 256, 1)
    mask = torch.ones(1, 256).bool()

    # Cache first two nodes' features
    node1 = feats[:, 0, :]
    node2 = feats[:, 1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = feats.clone().detach()
    feats_permuted_row_wise[:, 0, :] = node2
    feats_permuted_row_wise[:, 1, :] = node1

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)
    feats2, coors2 = layer(feats, coors, edges, mask=mask)
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'


def test_egnn_sparse_equivariance():
    layer = EGNN_Sparse(feats_dim=1,
                        m_dim=16,
                        fourier_features=4)

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)
    apply_action = lambda t: (t @ R + T).squeeze()

    feats = torch.randn(16, 1)
    coors = torch.randn(16, 3)
    edge_idxs = (torch.rand(2, 20) * 16).long()

    # Cache first two nodes' features
    node1 = feats[0, :]
    node2 = feats[1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = feats.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    x1 = torch.cat([coors, feats], dim=-1)
    x2 = torch.cat([apply_action(coors), feats], dim=-1)
    x3 = torch.cat([apply_action(coors), feats_permuted_row_wise], dim=-1)

    out1 = layer(x=x1, edge_index=edge_idxs)
    out2 = layer(x=x2, edge_index=edge_idxs)
    out3 = layer(x=x3, edge_index=edge_idxs)

    feats1, coors1 = out1[:, 3:], out1[:, :3]
    feats2, coors2 = out2[:, 3:], out2[:, :3]
    feats3, coors3 = out3[:, 3:], out3[:, :3]

    print(feats1 - feats2)
    print(apply_action(coors1) - coors2)
    assert torch.allclose(feats1, feats2), 'features must be invariant'
    assert torch.allclose(apply_action(coors1), coors2), 'coordinates must be equivariant'
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'


def test_geom_equivalence():
    layer = EGNN_Sparse(feats_dim=128,
                        edge_attr_dim=4,
                        m_dim=16,
                        fourier_features=4)

    feats = torch.randn(16, 128)
    coors = torch.randn(16, 3)
    x = torch.cat([coors, feats], dim=-1)
    edge_idxs = (torch.rand(2, 20) * 16).long()
    edges_attrs = torch.randn(16, 16, 4)
    edges_attrs = edges_attrs[edge_idxs[0], edge_idxs[1]]

    assert layer.forward(x, edge_idxs, edge_attr=edges_attrs).shape == x.shape
