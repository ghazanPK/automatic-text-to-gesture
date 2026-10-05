from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, width: int, max_frames: int = 256):
        super().__init__()
        pos = torch.arange(max_frames).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe = torch.zeros(max_frames, width); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class MotionEncoder(nn.Module):
    """Input projection to 150-D, sinusoidal positions, 3-layer/5-head Transformer (FF 512),
    masked temporal mean pooling, then Linear -> BatchNorm -> LeakyReLU and L2 normalisation."""
    def __init__(self, input_dim: int, model_dim: int = 150, latent_dim: int = 10):
        super().__init__()
        self.input = nn.Linear(input_dim, model_dim)
        self.position = PositionalEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, 5, 512, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(model_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.LeakyReLU(.1))
    def forward(self, x, mask=None):
        """``mask`` is a bool [N, F] tensor, True on real frames; padded frames are ignored by attention and pooling."""
        if mask is not None:
            mask = mask.bool()
        h = self.transformer(self.position(self.input(x)), src_key_padding_mask=(~mask if mask is not None else None))
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1) if mask is not None else h.mean(1)
        return F.normalize(self.head(pooled), dim=-1)


class GestureCLR(nn.Module):
    def __init__(self, dim_2d: int, dim_3d: int, latent_dim: int = 10):
        super().__init__(); self.pose2d = MotionEncoder(dim_2d, latent_dim=latent_dim); self.motion3d = MotionEncoder(dim_3d, latent_dim=latent_dim)
    def forward(self, pose2d, motion3d, mask=None, mask2d=None, mask3d=None):
        """``mask`` applies to both inputs; ``mask2d``/``mask3d`` override it per modality."""
        return self.pose2d(pose2d, mask2d if mask2d is not None else mask), self.motion3d(motion3d, mask3d if mask3d is not None else mask)


def ntxent(z2, z3, temperature: float = .07):
    """NT-Xent over the 2D-by-3D cosine matrix (eq. 5.1-5.3); latents are already L2-normalised.

    Row-wise cross-entropy matches each 2D sequence to its own 3D unit against all other units in the
    batch; the column-wise term does the same from the 3D side, and the two are averaged.
    """
    logits = z2 @ z3.T / temperature
    labels = torch.arange(len(z2), device=z2.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
