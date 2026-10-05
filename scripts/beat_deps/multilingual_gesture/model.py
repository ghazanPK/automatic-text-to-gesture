"""GestureCLR: 2D-pose and 3D-motion Transformer encoders trained with NT-Xent."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def padding_mask(lengths, frames, device=None):
    """``True`` marks padded frames (PyTorch ``src_key_padding_mask`` convention)."""
    if lengths is None:
        return None
    lengths = torch.as_tensor(lengths, device=device).long().clamp(1, frames)
    return torch.arange(frames, device=lengths.device)[None, :] >= lengths[:, None]


class Encoder(nn.Module):
    """Positional encoding, 3-layer/5-head Transformer (width 150, FF 512),
    masked mean pooling, then Linear -> BatchNorm -> LeakyReLU to a 10-D latent."""

    def __init__(self, input_dim, latent=10, width=150, max_frames=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, width)
        layer = nn.TransformerEncoderLayer(width, 5, 512, batch_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        pos = torch.arange(max_frames).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
        pe = torch.zeros(max_frames, width); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.out = nn.Sequential(nn.Linear(width, latent), nn.BatchNorm1d(latent), nn.LeakyReLU(.1))

    def forward(self, x, lengths=None):
        if x.shape[1] > len(self.pe):
            raise ValueError("sequence exceeds positional encoding capacity")
        mask = padding_mask(lengths, x.shape[1], x.device)
        h = self.net(self.proj(x) + self.pe[:x.shape[1]], src_key_padding_mask=mask)
        if mask is None:
            pooled = h.mean(1)
        else:
            keep = (~mask).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return F.normalize(self.out(pooled), dim=-1)


class GestureCLR(nn.Module):
    def __init__(self, d2, d3):
        super().__init__(); self.pose2d = Encoder(d2); self.motion3d = Encoder(d3)

    def forward(self, a, b, lengths_a=None, lengths_b=None):
        return self.pose2d(a, lengths_a), self.motion3d(b, lengths_b)


def ntxent(a, b, t=.07):
    """Symmetric NT-Xent over in-batch pairs (cosine similarity / temperature)."""
    logits = a @ b.T / t; labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
