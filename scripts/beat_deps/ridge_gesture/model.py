"""RIDGE dual encoder.

``RidgeModel`` follows the paper: a Sentence-BERT text branch with a
feed-forward projection, and a GestureCLR-architecture motion encoder
(positional encoding, 3-layer/5-head Transformer of width 150 with FF 512,
masked mean pooling, Linear -> BatchNorm -> LeakyReLU) into a shared 10-D space.
``TextMotionModel`` is the earlier compact variant kept for existing callers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPosition(nn.Module):
    def __init__(self, width, max_length=2048):
        super().__init__(); pos = torch.arange(max_length).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
        pe = torch.zeros(max_length, width); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if x.shape[1] > len(self.pe): raise ValueError("sequence exceeds positional encoding capacity")
        return x + self.pe[:x.shape[1]]


class TextMotionModel(nn.Module):
    """Compact variant: text MLP plus a 2-layer temporal Transformer."""

    def __init__(self, text_dim, motion_dim, latent=10, width=128):
        super().__init__(); self.text = nn.Sequential(nn.Linear(text_dim, width), nn.GELU(), nn.Linear(width, latent))
        self.motion_in = nn.Linear(motion_dim, width); self.position = SinusoidalPosition(width)
        layer = nn.TransformerEncoderLayer(width, 4, 256, batch_first=True, activation="gelu")
        self.motion = nn.TransformerEncoder(layer, 2); self.motion_out = nn.Linear(width, latent)

    def forward(self, text, motion):
        if motion.ndim != 3 or text.ndim != 2 or len(text) != len(motion):
            raise ValueError("expected paired text [B,D] and motion [B,F,D]")
        return F.normalize(self.text(text), dim=-1), \
            F.normalize(self.motion_out(self.motion(self.position(self.motion_in(motion))).mean(1)), dim=-1)


def padding_mask(lengths, frames, device=None):
    if lengths is None:
        return None
    lengths = torch.as_tensor(lengths, device=device).long().clamp(1, frames)
    return torch.arange(frames, device=lengths.device)[None, :] >= lengths[:, None]


class GestureEncoder(nn.Module):
    """GestureCLR motion encoder; parameter names match ``GestureCLR.motion3d``."""

    def __init__(self, input_dim, latent=10, width=150, max_frames=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, width)
        layer = nn.TransformerEncoderLayer(width, 5, 512, batch_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        pos = torch.arange(max_frames).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
        pe = torch.zeros(max_frames, width); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.out = nn.Sequential(nn.Linear(width, latent), nn.BatchNorm1d(latent), nn.LeakyReLU(.1))

    def forward(self, x, lengths=None):
        if x.shape[1] > len(self.pe):
            raise ValueError("sequence exceeds positional encoding capacity")
        mask = padding_mask(lengths, x.shape[1], x.device)
        h = self.net(self.proj(x) + self.pe[:x.shape[1]], src_key_padding_mask=mask)
        if mask is None:
            pooled = h.mean(1)
        else:
            keep = (~mask).unsqueeze(-1).float(); pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return F.normalize(self.out(pooled), dim=-1)


class RidgeModel(nn.Module):
    ARCHITECTURE = "ridge-gestureclr-v1"

    def __init__(self, text_dim, motion_dim, latent=10, text_width=256):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(text_dim, text_width), nn.GELU(), nn.Linear(text_width, latent))
        self.motion = GestureEncoder(motion_dim, latent)

    def encode_text(self, text):
        return F.normalize(self.text(text), dim=-1)

    def encode_motion(self, motion, lengths=None):
        return self.motion(motion, lengths)

    def forward(self, text, motion, lengths=None):
        if motion.ndim != 3 or text.ndim != 2 or len(text) != len(motion):
            raise ValueError("expected paired text [B,D] and motion [B,F,D]")
        return self.encode_text(text), self.encode_motion(motion, lengths)


def contrastive(text, motion, temperature=.07):
    """Symmetric in-batch InfoNCE: every other pair in the batch is a negative."""
    logits = text @ motion.T / temperature; labels = torch.arange(len(text), device=text.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def gesture_encoder_state(checkpoint):
    """Extract GestureEncoder weights from a GestureCLR-style checkpoint.

    Accepted formats (``torch.save`` dictionaries):
    * ``{"state": GestureCLR.state_dict(), "d3": D}`` as written by the
      ``multigesture``/``wild-pose`` ``train`` commands; ``motion3d.*`` keys are used;
    * ``{"motion_encoder": GestureEncoder.state_dict()}``;
    * a bare ``GestureEncoder.state_dict()``.
    """
    state = checkpoint.get("motion_encoder", checkpoint.get("state", checkpoint))
    if any(k.startswith("motion3d.") for k in state):
        state = {k[len("motion3d."):]: v for k, v in state.items() if k.startswith("motion3d.")}
    state = {k: v for k, v in state.items() if k != "pe"}
    if "proj.weight" not in state or "out.0.weight" not in state:
        raise ValueError("checkpoint has no GestureCLR motion encoder weights")
    return state


def load_gesture_init(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = gesture_encoder_state(checkpoint)
    expected = model.motion.proj.weight.shape
    if tuple(state["proj.weight"].shape) != tuple(expected):
        raise ValueError(f"GestureCLR motion input {tuple(state['proj.weight'].shape)} does not match {tuple(expected)}")
    if tuple(state["out.0.weight"].shape) != tuple(model.motion.out[0].weight.shape):
        raise ValueError("GestureCLR latent size differs from the RIDGE latent size")
    model.motion.load_state_dict(state)
    return model


def load_checkpoint(path):
    """Return ``(model, checkpoint)`` for RIDGE checkpoints of either architecture."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("architecture") == RidgeModel.ARCHITECTURE:
        model = RidgeModel(ck["text_dim"], ck["motion_dim"], ck.get("latent", 10))
    else:
        model = TextMotionModel(ck["text_dim"], ck["motion_dim"])
    model.load_state_dict(ck["state"]); model.eval()
    return model, ck


def encode_text(model, text):
    return model.encode_text(text) if isinstance(model, RidgeModel) else F.normalize(model.text(text), dim=-1)
