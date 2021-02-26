<img src="./egnn.png" width="600px"></img>

## EGNN - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2102.09844v1">E(n)-Equivariant Graph Neural Networks</a>, in Pytorch. May be eventually used for Alphafold2 replication.

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

## Todo

- [ ] add integration with pytorch geometric
- [ ] add tests for se3 equivariance
- [ ] add an EGAT (attention flavored variant)


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
