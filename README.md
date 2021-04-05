<img src="./egnn.png" width="600px"></img>

## EGNN - Pytorch

Implementation of <a href="https://arxiv.org/abs/2102.09844v1">E(n)-Equivariant Graph Neural Networks</a>, in Pytorch. May be eventually used for Alphafold2 replication. This technique went for simple invariant features, and ended up beating all previous methods (including SE3 Transformer and Lie Conv) in both accuracy and performance. SOTA in dynamical system models, molecular activity prediction tasks, etc.

## Install

```bash
$ pip install egnn-pytorch
```

## Usage

```python
import torch
from egnn_pytorch import EGNN

layer1 = EGNN(dim = 512)
layer2 = EGNN(dim = 512)

feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3)

feats, coors = layer1(feats, coors)
feats, coors = layer2(feats, coors) # (1, 16, 512), (1, 16, 3)
```

With edges

```python
import torch
from egnn_pytorch import EGNN

layer1 = EGNN(dim = 512, edge_dim = 4)
layer2 = EGNN(dim = 512, edge_dim = 4)

feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3)
edges = torch.randn(1, 16, 16, 4)

feats, coors = layer1(feats, coors, edges)
feats, coors = layer2(feats, coors, edges) # (1, 16, 512), (1, 16, 3)
```

A full EGNN network

```python
import torch
from egnn_pytorch.egnn_pytorch import EGNN_Network

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8
)

feats = torch.randint(0, 21, (1, 1024)) # (1, 1024)
coors = torch.randn(1, 1024, 3)         # (1, 1024, 3)
mask = torch.ones_like(feats).bool()    # (1, 1024)

feats_out, coors_out = net(feats, coors, mask = mask) # (1, 1024, 32), (1, 1024, 3)
```

Only attend to sparse neighbors, given to the network as an adjacency matrix.

```python
import torch
from egnn_pytorch.egnn_pytorch import EGNN_Network

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    only_sparse_neighbors = True
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat) # (1, 1024, 32), (1, 1024, 3)
```

You can also have the network automatically determine the Nth-order neighbors, and pass in an adjacency embedding (depending on the order) to be used as an edge, with two extra keyword arguments

```python
import torch
from egnn_pytorch.egnn_pytorch import EGNN_Network

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    num_adj_degrees = 3,           # fetch up to 3rd degree neighbors
    adj_dim = 8,                   # pass an adjacency degree embedding to the EGNN layer, to be used in the edge MLP
    only_sparse_neighbors = True
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, mask = mask, adj_mat = adj_mat) # (1, 1024, 32), (1, 1024, 3)
```

## Edges

If you need to pass in continuous edges

```python
import torch
from egnn_pytorch.egnn_pytorch import EGNN_Network

net = EGNN_Network(
    num_tokens = 21,
    dim = 32,
    depth = 3,
    edge_dim = 4,
    num_nearest_neighbors = 3
)

feats = torch.randint(0, 21, (1, 1024))
coors = torch.randn(1, 1024, 3)
mask = torch.ones_like(feats).bool()

continuous_edges = torch.randn(1, 1024, 1024, 4)

# naive adjacency matrix
# assuming the sequence is connected as a chain, with at most 2 neighbors - (1024, 1024)
i = torch.arange(1024)
adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

feats_out, coors_out = net(feats, coors, edges = continuous_edges, mask = mask, adj_mat = adj_mat) # (1, 1024, 32), (1, 1024, 3)
```

## Examples

To run the protein backbone denoising example, first install `sidechainnet`

```bash
$ pip install sidechainnet
```

Then

```bash
$ python denoise_sparse.py
```

## Citations

```bibtex
@misc{satorras2021en,
    title 	= {E(n) Equivariant Graph Neural Networks}, 
    author 	= {Victor Garcia Satorras and Emiel Hoogeboom and Max Welling},
    year 	= {2021},
    eprint 	= {2102.09844},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
