"""GestureCLR training: paper augmentation, AdamW + cosine annealing, validation split and best checkpoint."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .pipeline import PAPER_NOISE_VARIANCES, augment_projected
from .units import Normalization

CHECKPOINT_FORMAT = "gestureclr/2"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    latent_dim: int = 10
    noise_variances: tuple = PAPER_NOISE_VARIANCES
    shift_prob: float = 0.5
    crop_frames: int = 30
    max_offset: int = 15
    fills: tuple = ("mean", "zero")
    val_fraction: float = 0.1
    max_steps: int | None = None
    seed: int = 0


# Paper: AdamW, lr 5e-4, weight decay 1e-4, cosine annealing, up to 1000 epochs, batch 512 (thesis 5.2.3).
# Demo: the same optimiser and augmentation at a step count that converges on a few hundred units.
PRESETS = {
    "paper": TrainConfig(epochs=1000, batch_size=512),
    "demo": TrainConfig(epochs=300, batch_size=64),
}


def config_from(preset: str = "demo", path: str | Path | None = None, **overrides) -> TrainConfig:
    """Preset, then an optional JSON config file, then explicit (non-None) overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose {sorted(PRESETS)}")
    values = asdict(PRESETS[preset])
    if path:
        extra = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(extra) - {f.name for f in fields(TrainConfig)}
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        values.update(extra)
    values.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("noise_variances", "fills"):
        values[key] = tuple(values[key])
    return TrainConfig(**values)


def encode(encoder, x: np.ndarray, mask: np.ndarray | None = None, batch: int = 512) -> np.ndarray:
    """Eval-mode latents for one modality encoder in batches."""
    import torch
    encoder.eval()
    mask = np.ones(x.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    rows = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            rows.append(encoder(torch.from_numpy(np.ascontiguousarray(x[i:i + batch], np.float32)), torch.from_numpy(mask[i:i + batch])).numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)


def top1(z2: np.ndarray, z3: np.ndarray) -> float:
    """Fraction of 2D sequences whose nearest 3D latent is their own unit."""
    if not len(z2):
        return float("nan")
    return float(((z2 @ z3.T).argmax(1) == np.arange(len(z2))).mean())


def _views(x2, m2, idx, rng, cfg):
    pairs = [augment_projected(x2[i], rng, m2[i], cfg.noise_variances, cfg.shift_prob, cfg.fills, cfg.crop_frames, cfg.max_offset) for i in idx]
    return np.stack([p[0] for p in pairs]).astype(np.float32), np.stack([p[1] for p in pairs])


def train_gestureclr(pose2d: np.ndarray, motion3d: np.ndarray, mask: np.ndarray | None = None, cfg: TrainConfig = PRESETS["demo"], log=print):
    """Train on scale-normalised, padded ``[N, F, D]`` arrays. Returns (model with the best weights, history dict)."""
    import torch
    from .model import GestureCLR, ntxent

    x2 = np.asarray(pose2d, np.float32); x3 = np.asarray(motion3d, np.float32)
    if len(x2) != len(x3) or len(x2) < 2:
        raise ValueError("training requires at least two aligned 2D/3D pairs")
    if cfg.epochs < 1 or cfg.batch_size < 2:
        raise ValueError("epochs must be positive and batch size must be at least two")
    m = np.ones(x2.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x2))
    n_val = max(2, int(round(len(x2) * cfg.val_fraction))) if cfg.val_fraction > 0 and len(x2) >= 6 else 0
    val, tr = order[:n_val], order[n_val:]
    if len(tr) < 2:
        raise ValueError("training split needs at least two pairs")
    batch = min(cfg.batch_size, len(tr))
    steps_per_epoch = max(1, len(tr) // batch)  # drop the ragged last batch (BatchNorm needs >1 sample)
    epochs = cfg.epochs if cfg.max_steps is None else max(1, math.ceil(cfg.max_steps / steps_per_epoch))
    total_steps = epochs * steps_per_epoch
    model = GestureCLR(x2.shape[-1], x3.shape[-1], cfg.latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    val_rng = np.random.default_rng(cfg.seed + 1)
    if n_val:
        v2, vm2 = _views(x2, m, val, val_rng, cfg)  # fixed augmented validation views
        v3, vm3 = torch.from_numpy(x3[val]), torch.from_numpy(m[val])
        v2t, vm2t = torch.from_numpy(v2), torch.from_numpy(vm2)
    history = {"config": asdict(cfg), "train_pairs": int(len(tr)), "val_pairs": int(n_val), "epochs": []}
    best_loss, best_state, best_epoch, step = math.inf, None, 0, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        total = 0.0; seen = 0
        for b in range(steps_per_epoch):
            ids = perm[b * batch:(b + 1) * batch]
            a2, am2 = _views(x2, m, ids, rng, cfg)
            z2, z3 = model(torch.from_numpy(a2), torch.from_numpy(x3[ids]), mask2d=torch.from_numpy(am2), mask3d=torch.from_numpy(m[ids]))
            loss = ntxent(z2, z3, cfg.temperature)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            total += float(loss.detach()) * len(ids); seen += len(ids)
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        row = {"epoch": epoch + 1, "step": step, "loss": total / max(seen, 1), "lr": float(sched.get_last_lr()[0])}
        if n_val:
            model.eval()
            with torch.no_grad():
                z2v, z3v = model(v2t, v3, mask2d=vm2t, mask3d=vm3)
                row["val_loss"] = float(ntxent(z2v, z3v, cfg.temperature))
                row["val_top1"] = top1(z2v.numpy(), z3v.numpy())
        monitor = row.get("val_loss", row["loss"])
        if monitor < best_loss:
            best_loss, best_epoch, best_state = monitor, epoch + 1, copy.deepcopy(model.state_dict())
            row["best"] = True
        history["epochs"].append(row)
        if log and (epoch == 0 or (epoch + 1) % max(1, epochs // 20) == 0 or epoch + 1 == epochs):
            log(json.dumps(row))
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_monitor=best_loss, monitor="val_loss" if n_val else "train_loss")
    # Eval-mode own-pair top-1 with one fixed augmented view per training pair (BatchNorm running statistics).
    t2, tm2 = _views(x2, m, tr, np.random.default_rng(cfg.seed + 2), cfg)
    history["train_top1_eval"] = top1(encode(model.pose2d, t2, tm2), encode(model.motion3d, x3[tr], m[tr]))
    if n_val:
        history["val_top1_best"] = top1(encode(model.pose2d, v2, vm2), encode(model.motion3d, x3[val], m[val]))
    history["val_ids"] = [int(i) for i in val]
    return model, history


def save_checkpoint(path: str | Path, model, dim2: int, dim3: int, cfg: TrainConfig, norm: Normalization, history: dict | None = None) -> None:
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CHECKPOINT_FORMAT, "state": model.state_dict(), "dim2": int(dim2), "dim3": int(dim3), "latent_dim": int(cfg.latent_dim),
                "normalization": norm.to_dict(), "config": json.dumps(asdict(cfg)),
                "best_epoch": int(history.get("best_epoch", 0)) if history else 0}, path)


def load_checkpoint(path: str | Path, dim2: int | None = None, dim3: int | None = None):
    """Return (model in eval mode, Normalization). Plain state-dict checkpoints from earlier versions load without normalisation."""
    import torch
    from .model import GestureCLR
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and data.get("format") == CHECKPOINT_FORMAT:
        model = GestureCLR(data["dim2"], data["dim3"], data["latent_dim"])
        model.load_state_dict(data["state"])
        norm = Normalization.from_dict(data["normalization"])
    else:
        if dim2 is None or dim3 is None:
            raise ValueError("legacy checkpoint needs explicit input dimensions")
        latent = next(v.shape[0] for k, v in data.items() if k.endswith("head.0.weight"))
        model = GestureCLR(dim2, dim3, latent)
        model.load_state_dict(data)
        norm = Normalization(enabled=False)
    model.eval()
    return model, norm
