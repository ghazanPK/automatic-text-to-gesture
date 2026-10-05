"""Pose scale normalisation and Algorithm 3 gesture-unit extraction (Ali, thesis section 5.2.1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Normalization:
    """Neck-centred, shoulder-width-scaled poses. ``shoulders=None`` scales by RMS joint distance from the neck.

    Default indices follow the upper-body order written by ``scripts/prepare_public_data.py``:
    Neck=1, LeftArm=4, RightArm=8 (BVH ``*Arm`` joints sit at the shoulders).
    """
    neck: int = 1
    shoulders: tuple | None = (4, 8)
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Normalization":
        if not data:
            return cls(enabled=False)
        shoulders = data.get("shoulders")
        return cls(int(data.get("neck", 1)), tuple(shoulders) if shoulders is not None else None, bool(data.get("enabled", True)))

    def apply(self, x: np.ndarray, coords: int, mask: np.ndarray | None = None) -> np.ndarray:
        """Normalise ``[F, J*c]``, ``[N, F, J*c]`` or ``[..., F, J, c]`` sequences; padded frames stay zero."""
        x = np.asarray(x, np.float32)
        if not self.enabled:
            return x.copy()
        flat = x.shape[-1] != coords
        seq = x.reshape(*x.shape[:-1], -1, coords) if flat else x
        single = seq.ndim == 3
        if single:
            seq = seq[None]
        m = np.ones(seq.shape[:2], bool) if mask is None else np.asarray(mask, bool).reshape(seq.shape[:2])
        joints = seq.shape[2]
        if not 0 <= self.neck < joints or (self.shoulders is not None and not all(0 <= j < joints for j in self.shoulders)):
            raise ValueError(f"normalisation joints neck={self.neck} shoulders={self.shoulders} fall outside a {joints}-joint skeleton")
        out = seq - seq[:, :, self.neck : self.neck + 1]
        if self.shoulders is not None:
            width = np.linalg.norm(out[:, :, self.shoulders[0]] - out[:, :, self.shoulders[1]], axis=-1)
        else:
            width = np.sqrt((out ** 2).sum(-1).mean(-1))
        scale = np.array([np.median(w[v]) if v.any() else 1.0 for w, v in zip(width, m)], np.float32)
        scale = np.where(scale > 1e-6, scale, 1.0)
        out = out / scale[:, None, None, None]
        out[~m] = 0
        if single:
            out = out[0]
        return out.reshape(x.shape).astype(np.float32)


def _features(motion: np.ndarray, coords: int, norm: Normalization) -> np.ndarray:
    x = norm.apply(motion, coords)
    return x.reshape(len(x), -1)


def _window_stats(x: np.ndarray, lengths: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Closure (start-end distance) and mean per-dimension variance for every (start, length)."""
    frames = len(x)
    c1 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0, dtype=np.float64)])
    c2 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x.astype(np.float64) ** 2, 0)])
    closure = np.full((frames, len(lengths)), np.inf)
    variance = np.full((frames, len(lengths)), -np.inf)
    for k, n in enumerate(lengths):
        if n > frames:
            continue
        s = np.arange(frames - n + 1)
        mean = (c1[s + n] - c1[s]) / n
        variance[s, k] = np.maximum(((c2[s + n] - c2[s]) / n - mean ** 2).mean(1), 0)
        closure[s, k] = np.linalg.norm(x[s] - x[s + n - 1], axis=1)
    return closure, variance


def window_variances(motions: Iterable[np.ndarray], coords: int = 3, length: int = 37, norm: Normalization = Normalization()) -> np.ndarray:
    """Variance of every ``length``-frame window (stride 1) of scale-normalised motion."""
    values = []
    for motion in motions:
        x = _features(motion, coords, norm)
        if len(x) >= length:
            values.append(_window_stats(x, [length])[1][: len(x) - length + 1, 0])
    if not values:
        raise ValueError("motion is shorter than one unit")
    return np.concatenate(values)


def suggest_variance_threshold(variances: np.ndarray, method: str = "percentile", percentile: float = 25.0) -> float:
    """Pick the expert-dependent variance cut-off: a percentile of window variances, or the elbow of their sorted curve."""
    v = np.sort(np.asarray(variances, np.float64))
    if not len(v):
        raise ValueError("no variances")
    if method == "percentile":
        return float(np.percentile(v, percentile))
    if method != "elbow":
        raise ValueError("method must be percentile or elbow")
    if v[-1] - v[0] < 1e-12:
        return float(v[0])
    x = np.linspace(0, 1, len(v))
    y = (v - v[0]) / (v[-1] - v[0])
    return float(v[int(np.argmax(x - y))])  # farthest point below the chord of the convex sorted curve


def extract_units(motion: np.ndarray, fps: float = 15, min_seconds: float = 2.0, max_seconds: float = 3.0, variance_threshold: float = 0.0,
                  max_closure: float | None = None, coords: int = 3, norm: Normalization = Normalization()) -> list[tuple[int, int]]:
    """Algorithm 3: repeatedly take the 2-3 s clip whose start and end poses are closest, if its variance passes.

    Every candidate (start, end) with ``min..max`` frames lies inside a still-unused part of the sequence.
    The globally minimal start-end distance clip that meets ``variance_threshold`` (and ``max_closure``)
    is extracted and removed, splitting the remaining motion, until no candidate is left. Distances and
    variances use neck-centred, shoulder-width-scaled poses. Returns sorted ``(start, end)`` frame spans.
    """
    lo, hi = int(round(min_seconds * fps)), int(round(max_seconds * fps))
    if lo < 2 or hi < lo:
        raise ValueError("unit length range must satisfy 2 <= min <= max frames")
    x = _features(motion, coords, norm)
    frames = len(x)
    lengths = list(range(lo, hi + 1))
    closure, variance = _window_stats(x, lengths)
    allowed = variance >= variance_threshold
    if max_closure is not None:
        allowed &= closure <= max_closure
    score = np.where(allowed, closure, np.inf)
    starts = np.arange(frames)[:, None]
    ln = np.asarray(lengths)[None, :]
    ends = np.minimum(starts + ln, frames)
    used = np.zeros(frames + 1, np.int64)
    spans = []
    while True:
        occupied = np.concatenate([[0], np.cumsum(used[:frames])])
        free = (occupied[ends] - occupied[starts] == 0) & (starts + ln <= frames)
        masked = np.where(free, score, np.inf)
        flat = int(np.argmin(masked))
        if not np.isfinite(masked.flat[flat]):
            break
        s, k = divmod(flat, len(lengths))
        spans.append((s, s + lengths[k]))
        used[s : s + lengths[k]] = 1
    return sorted(spans)


def dense_windows(frames: int, fps: float = 15, min_seconds: float = 2.0, max_seconds: float = 3.0, stride: int = 5, seed: int = 0) -> list[tuple[int, int]]:
    """Overlapping 2-3 s training sequences (random length per start) for GestureCLR pairs."""
    lo, hi = int(round(min_seconds * fps)), int(round(max_seconds * fps))
    rng = np.random.default_rng(seed)
    return [(s, s + n) for s in range(0, frames - lo + 1, max(1, stride)) for n in [int(rng.integers(lo, hi + 1))] if s + n <= frames]


def project_xy(motion: np.ndarray) -> np.ndarray:
    """Orthographic camera looking down -Z: keep X and Y of ``[..., J, 3]``."""
    return np.asarray(motion, np.float32)[..., :2]


def pack_units(motions: dict[str, np.ndarray], spans: dict[str, list[tuple[int, int]]], max_frames: int) -> dict[str, np.ndarray]:
    """Pad variable-length units at the end to ``max_frames`` and emit 3D, projected 2D, masks, lengths and unique ids."""
    rows3, rows2, masks, lengths, ids, takes, starts, ends = [], [], [], [], [], [], [], []
    for take, take_spans in spans.items():
        motion = np.asarray(motions[take], np.float32)
        for s, e in take_spans:
            n = e - s
            if n > max_frames:
                raise ValueError("unit longer than max_frames")
            clip = motion[s:e]
            pad = ((0, max_frames - n), (0, 0), (0, 0))
            rows3.append(np.pad(clip, pad).reshape(max_frames, -1))
            rows2.append(np.pad(project_xy(clip), pad).reshape(max_frames, -1))
            mask = np.zeros(max_frames, bool); mask[:n] = True
            masks.append(mask); lengths.append(n); ids.append(f"{take}:{s}-{e}"); takes.append(take); starts.append(s); ends.append(e)
    if not ids:
        raise ValueError("no units were extracted; lower the variance threshold or check the motion length")
    return {"motion3d": np.stack(rows3).astype(np.float32), "pose2d": np.stack(rows2).astype(np.float32), "mask": np.stack(masks),
            "lengths": np.asarray(lengths, np.int32), "ids": np.asarray(ids), "takes": np.asarray(takes),
            "starts": np.asarray(starts, np.int32), "ends": np.asarray(ends, np.int32), "dim2": np.asarray(rows2[0].shape[-1])}
