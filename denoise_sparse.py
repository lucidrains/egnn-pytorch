import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from einops import rearrange, repeat

import sidechainnet as scn
from egnn_pytorch.egnn_pytorch import EGNN_Network

torch.set_default_dtype(torch.float64)

BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

net = EGNN_Network(
    num_tokens = 21,
    num_positions = 200 * 3,   # maximum number of positions - absolute positional embedding since there is inherent order in the sequence
    depth = 5,
    dim = 8,
    num_nearest_neighbors = 16,
    fourier_features = 2,
    norm_coors = True,
    coor_weights_clamp_value = 2.
).cuda()

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

dl = cycle(data['train'])
optim = Adam(net.parameters(), lr=1e-3)

for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda().type(torch.float64)
        masks = masks.cuda().bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # Keeping only the backbone coordinates

        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        i = torch.arange(seq.shape[-1], device = seq.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

        noised_coords = coords + torch.randn_like(coords)

        feats, denoised_coords = net(seq, noised_coords, adj_mat = adj_mat, mask = masks)

        loss = F.mse_loss(denoised_coords[masks], coords[masks])

        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print('loss:', loss.item())
    optim.step()
    optim.zero_grad()
