import torch
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

class EGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16
    ):
        super().__init__()
        input_dim = 2 * dim + edge_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, m_dim)
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
        b, n, d = feats.shape

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(dim = -1, keepdim = True)

        feats_i = repeat(feats, 'b i d -> b i n d', n = n)
        feats_j = repeat(feats, 'b j d -> b n j d', n = n)
        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors)

        m_i = m_ij.sum(dim = -2)

        hidden_mlp_input = torch.cat((feats, m_i), dim = -1)
        hidden_out = self.hidden_mlp(hidden_mlp_input)

        return hidden_out, coors_out
