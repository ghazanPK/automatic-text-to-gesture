"""Shared BEAT ingest: processed OmniMo collections or raw BVH/TextGrid folders.

Every loader returns plain per-take records::

    {"id": take, "speaker": "1", "take": "1_wayne_0_1_1", "fps": 30,
     "joint_names": [...], "positions": float32 [T, J, 3] metres,
     "words": [(word, start_frame, end_frame), ...], "source": {...}}

Positions use one basis for both routes: metres, Y up, +Z forward and the
subject's left on +X (the raw BEAT BVH convention). Processed OmniMo Unity data
is left-negative-X and is mirrored on load; ``source.axis_signs`` records the
original basis. Only numpy is required. Recordings, prepared arrays and fitted
weights belong in ignored folders and are never committed. See BEAT_INGEST.md.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "paperreach.beat-ingest.v1"
HF_BASE = ("https://huggingface.co/datasets/H-Liu1997/BEAT/resolve/main/"
           "beat_english_v0.2.1/beat_english_v0.2.1/")
BASIS = "metres; Y up; +Z forward; subject left on +X (raw BEAT BVH basis)"

# Canonical upper-body joints of the four gesture-method data contracts, named
# as in the raw BEAT BVH. Processed Unity names map onto them through ALIASES.
UPPER_BODY = ("Hips", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
              "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand")
ALIASES = {"LeftUpperArm": "LeftArm", "LeftLowerArm": "LeftForeArm",
           "RightUpperArm": "RightArm", "RightLowerArm": "RightForeArm",
           "LeftUpperLeg": "LeftUpLeg", "LeftLowerLeg": "LeftLeg",
           "RightUpperLeg": "RightUpLeg", "RightLowerLeg": "RightLeg",
           "LeftToes": "LeftToeBase", "RightToes": "RightToeBase"}

DEFAULT_ROLES = {
    "automatic": {"library": 0.5, "video": "rest"},
    "wild": {"library": 1 / 3, "train": 1 / 3, "wild": "rest"},
    "multilingual": {"library": 1 / 3, "train": 1 / 3, "wild": "rest"},
    "ridge": {"library": 1 / 3, "train": 1 / 3, "heldout": "rest"},
}


# ----------------------------------------------------------------------------
# Rotation and forward kinematics (vectorised over frames)

def sixd_to_matrix(values):
    """6D rotation (two Gram-Schmidt columns) to matrices, any leading shape."""
    a = np.asarray(values, dtype=np.float64)
    shape = a.shape[:-1]
    a = a.reshape(-1, 6)
    x = a[:, :3] / np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-12)
    y = a[:, 3:] - np.sum(x * a[:, 3:], axis=1, keepdims=True) * x
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1).reshape(*shape, 3, 3)


def euler_matrices(axis, degrees):
    """Batch rotation matrices about one axis; degrees has shape [T]."""
    angle = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    c, s = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(c), np.zeros_like(c)
    if axis == "X":
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "Y":
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    elif axis == "Z":
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    else:
        raise ValueError(f"Unknown rotation axis {axis}")
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def forward_kinematics(local, root, parents, offsets):
    """World joint positions [T,J,3] from local rotations [T,J,3,3].

    ``offsets`` may be [J,3] or per-frame [T,J,3]; joints must be ordered so a
    parent precedes its children.
    """
    local = np.asarray(local, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    frames, count = local.shape[:2]
    world = np.empty_like(local)
    positions = np.empty((frames, count, 3), dtype=np.float64)
    for joint, parent in enumerate(parents):
        offset = offsets[..., joint, :]
        if parent < 0:
            world[:, joint] = local[:, joint]
            positions[:, joint] = np.asarray(root, dtype=np.float64) + offset
        else:
            if parent >= joint:
                raise ValueError("Skeleton joints must follow their parents")
            world[:, joint] = world[:, parent] @ local[:, joint]
            positions[:, joint] = positions[:, parent] + np.einsum(
                "tij,tj->ti", world[:, parent], np.broadcast_to(offset, (frames, 3)))
    return positions


def _frame_rows(source_fps, fps, total, start=0, max_frames=None):
    """Fractional source rows for target frames start..start+max_frames."""
    if fps <= 0 or source_fps <= 0:
        raise ValueError("fps must be positive")
    duration = (total - 1) / source_fps
    count = int(math.floor(duration * fps + 1e-6)) + 1
    stop = count if max_frames is None else min(count, start + int(max_frames))
    if start < 0 or start >= stop:
        raise ValueError("Invalid frame selection")
    return np.arange(start, stop) * (source_fps / fps)


def _sample(array, rows):
    """Linear interpolation of array[T,...] at fractional rows (exact on integers)."""
    low = np.floor(rows + 1e-6).astype(int).clip(0, len(array) - 1)
    high = (low + 1).clip(0, len(array) - 1)
    weight = (rows - low).clip(0, 1).reshape(-1, *([1] * (array.ndim - 1)))
    if np.all(weight < 1e-6):
        return np.asarray(array[low])
    return (1 - weight) * array[low] + weight * array[high]


# ----------------------------------------------------------------------------
# Name helpers

def canonical_name(name):
    return ALIASES.get(str(name), str(name))


def _scale_words(words, factor, offset=0, frames=None):
    out = []
    for word, start, end in words:
        s = int(round(start * factor)) - offset
        e = max(s + 1, int(round(end * factor)) - offset)
        if frames is not None:
            if e <= 0 or s >= frames:
                continue
            s, e = max(0, s), min(frames, e)
        out.append((str(word), s, e))
    return out


def _matches(value, patterns):
    if not patterns:
        return True
    return any(fnmatch.fnmatch(str(value).casefold(), str(p).casefold()) for p in patterns)


def _split(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v) for item in value for v in str(item).split(",")]
    else:
        items = str(value).split(",")
    return [v.strip() for v in items if v.strip()] or None


# ----------------------------------------------------------------------------
# Processed OmniMo collection

def _npz_member(path, name):
    """Memory-map an uncompressed NPZ member; fall back to a normal load."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            with archive.open(info) as stream:
                return np.lib.format.read_array(stream)
        with open(path, "rb") as stream:
            stream.seek(info.header_offset + 26)
            name_size = int.from_bytes(stream.read(2), "little")
            extra_size = int.from_bytes(stream.read(2), "little")
            stream.seek(info.header_offset + 30 + name_size + extra_size)
            version = np.lib.format.read_magic(stream)
            reader = (np.lib.format.read_array_header_2_0 if version >= (2, 0)
                      else np.lib.format.read_array_header_1_0)
            shape, fortran, dtype = reader(stream)
            if fortran:
                raise ValueError("Fortran ordered arrays are unsupported")
            offset = stream.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def detect_kind(source):
    source = Path(source)
    if (source / "meta.json").is_file() or any(source.glob("*/meta.json")):
        return "processed"
    if any(source.glob("*.bvh")) or any(source.glob("*/*.bvh")):
        return "raw"
    raise FileNotFoundError(f"{source}: no processed meta.json or raw .bvh takes found")


def _speaker_dirs(source):
    source = Path(source)
    if (source / "meta.json").is_file():
        return [source]
    return sorted((p.parent for p in source.glob("*/meta.json")),
                  key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name))


def _speaker_match(speaker_id, name, speakers):
    return not speakers or any(s.casefold() in {str(speaker_id).casefold(), str(name).casefold()}
                               for s in speakers)


def list_takes(source, *, kind="auto", speakers=None, takes=None, max_takes_per_speaker=None):
    """Return lightweight take descriptors without loading motion."""
    kind = detect_kind(source) if kind == "auto" else kind
    speakers, takes = _split(speakers), _split(takes)
    out = []
    if kind == "processed":
        for folder in _speaker_dirs(source):
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            speaker = folder.name
            name = (meta.get("speaker") or {}).get("name", "") if isinstance(meta.get("speaker"), dict) else ""
            if not _speaker_match(speaker, name, speakers):
                continue
            chosen = [t for t in meta["takes"] if _matches(t["sequence_id"], takes)]
            for record in chosen[:max_takes_per_speaker]:
                out.append({"kind": "processed", "speaker": speaker, "speaker_name": name,
                            "take": record["sequence_id"], "folder": str(folder),
                            "frames": int(record["frame_end"] - record["frame_start"]),
                            "fps": float(meta["fps"]), "words": len(record.get("words", []))})
    elif kind == "raw":
        source = Path(source)
        files = sorted(set(source.glob("*.bvh")) | set(source.glob("*/*.bvh")))
        counts = {}
        for bvh in files:
            take = bvh.stem
            parts = take.split("_")
            speaker = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            if not _speaker_match(speaker, name, speakers) or not _matches(take, takes):
                continue
            counts[speaker] = counts.get(speaker, 0) + 1
            if max_takes_per_speaker is not None and counts[speaker] > max_takes_per_speaker:
                continue
            grid = bvh.with_suffix(".TextGrid")
            out.append({"kind": "raw", "speaker": speaker, "speaker_name": name, "take": take,
                        "bvh": str(bvh), "textgrid": str(grid) if grid.is_file() else None})
    else:
        raise ValueError(f"Unknown source kind {kind}")
    return out


def load_processed(folder, take, *, fps=30, start_frame=0, max_frames=None, joints=None,
                   mirror_x=True):
    """Load one take from an OmniMo speaker folder (meta.json + motion.npz)."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    record = next((t for t in meta["takes"] if t["sequence_id"] == take), None)
    if record is None:
        raise KeyError(f"{folder}: no take {take}")
    source_fps = float(meta["fps"])
    total = int(record["frame_end"] - record["frame_start"])
    rows = _frame_rows(source_fps, fps, total, start_frame, max_frames)
    first, last = int(math.floor(rows[0])), min(total - 1, int(math.ceil(rows[-1])))
    motion = folder / "motion.npz"
    base = int(record["frame_start"])
    rotations = np.asarray(_npz_member(motion, "rotations")[base + first: base + last + 1], np.float64)
    root = np.asarray(_npz_member(motion, "root")[base + first: base + last + 1], np.float64)
    parents = [int(p) for p in np.asarray(_npz_member(motion, "parents"))]
    offsets = np.asarray(_npz_member(motion, "offsets"), np.float64)
    count = len(parents)
    local = sixd_to_matrix(rotations.reshape(len(rotations), count, 6))
    positions = forward_kinematics(local, root, parents, offsets)
    positions = _sample(positions, rows - first)
    if mirror_x:
        positions[..., 0] *= -1
    names = [canonical_name(n) for n in meta["joints"]]
    frames = len(positions)
    words = [(w.get("text", w.get("word", "")), int(w["frame_start"]), int(w["frame_end"]))
             for w in record.get("words", []) if str(w.get("text", w.get("word", ""))).strip()]
    words = _scale_words(words, fps / source_fps, offset=int(start_frame), frames=frames)
    speaker_meta = meta.get("speaker") if isinstance(meta.get("speaker"), dict) else {}
    result = {"id": take, "speaker": folder.name, "take": take, "fps": float(fps),
              "joint_names": names, "positions": positions.astype(np.float32), "words": words,
              "source": {"kind": "processed", "dataset": meta.get("dataset", "beat"),
                         "version": meta.get("version"), "folder": str(folder),
                         "speaker_name": speaker_meta.get("name"),
                         "take_frame_start": int(start_frame), "source_fps": source_fps,
                         "axis_signs": [-1, 1, 1], "mirrored_x": bool(mirror_x),
                         "original_joint_names": list(meta["joints"]),
                         "motion_url": HF_BASE + f"{folder.name}/{take}.bvh",
                         "alignment_url": HF_BASE + f"{folder.name}/{take}.TextGrid"},
              "basis": BASIS}
    return select_joints(result, joints) if joints else result


# ----------------------------------------------------------------------------
# Raw BEAT BVH + TextGrid

def parse_bvh(path):
    """Return header (names, parents, offsets in file units, channels), frame time and values."""
    text = Path(path).read_text(encoding="utf-8")
    head, sep, body = text.partition("MOTION")
    if not sep:
        raise ValueError(f"{path}: BVH is missing MOTION")
    names, parents, offsets, channels = [], [], [], []
    stack, pending, end_site = [], None, False
    for line in head.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            pending = len(names)
            names.append(parts[1]); parents.append(stack[-1] if stack else -1)
            offsets.append([0., 0., 0.]); channels.append([])
        elif parts[:2] == ["End", "Site"]:
            end_site = True
        elif parts[0] == "{":
            stack.append(-2 if end_site else pending); pending = None
        elif parts[0] == "}":
            stack.pop(); end_site = False
        elif parts[0] == "OFFSET" and stack and stack[-1] >= 0:
            offsets[stack[-1]] = [float(x) for x in parts[1:4]]
        elif parts[0] == "CHANNELS" and stack and stack[-1] >= 0:
            channels[stack[-1]] = parts[2:]
    lines = body.strip().split("\n", 2)
    frame_count = int(lines[0].split(":", 1)[1])
    frame_time = float(lines[1].split(":", 1)[1])
    if frame_time <= 0:
        raise ValueError("BVH Frame Time must be positive")
    width = sum(map(len, channels))
    values = np.array(lines[2].split(), dtype=np.float64) if len(lines) > 2 else np.zeros(0)
    if width == 0 or values.size % width:
        raise ValueError(f"{path}: BVH channel count mismatch")
    values = values.reshape(-1, width)[:frame_count]
    return {"names": names, "parents": parents, "offsets": np.asarray(offsets), "channels": channels}, frame_time, values


def bvh_positions(header, values, unit_scale=0.01):
    """FK for selected BVH rows; returns metres [T,J,3] (BEAT BVH is centimetres)."""
    frames = len(values)
    local = np.empty((frames, len(header["names"]), 3, 3))
    translation = np.broadcast_to(header["offsets"], (frames, len(header["names"]), 3)).copy()
    cursor = 0
    for joint, joint_channels in enumerate(header["channels"]):
        matrix = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        for channel in joint_channels:
            column = values[:, cursor]; cursor += 1
            if channel.endswith("position"):
                translation[:, joint, "XYZ".index(channel[0])] += column
            elif channel.endswith("rotation"):
                matrix = matrix @ euler_matrices(channel[0], column)
            else:
                raise ValueError(f"Unsupported BVH channel {channel}")
        local[:, joint] = matrix
    root = np.zeros((frames, 3))
    return forward_kinematics(local, root, header["parents"], translation) * unit_scale


def parse_textgrid(path, fps=30.0, tier="words"):
    """Word intervals from a Praat long-format TextGrid as (word, start, end) frames."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    block = next((x for x in re.split(r"\bitem \[\d+\]:", text) if re.search(rf'name = "{tier}"', x)), None)
    if block is None:
        raise ValueError(f"{path}: TextGrid has no {tier} tier")
    values = re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "((?:[^"]|"")*)"', block)
    words = []
    for start, end, word in values:
        word = word.replace('""', '"').strip()
        if not word:
            continue
        s = int(round(float(start) * fps))
        e = max(s + 1, int(round(float(end) * fps)))
        words.append((word, s, e))
    return words


def load_bvh(bvh, textgrid=None, *, fps=30, start_frame=0, max_frames=None, joints=None,
             speaker=None):
    """Load one raw BEAT take; positions in metres, words from its TextGrid."""
    bvh = Path(bvh)
    header, frame_time, values = parse_bvh(bvh)
    source_fps = 1.0 / frame_time
    if abs(source_fps - round(source_fps)) < 0.01:  # "0.008333" means exactly 120 Hz
        source_fps = float(round(source_fps))
    rows = _frame_rows(source_fps, fps, len(values), start_frame, max_frames)
    if np.allclose(rows, np.round(rows), atol=1e-4):
        positions = bvh_positions(header, values[np.round(rows).astype(int)])
    else:
        first, last = int(math.floor(rows[0])), min(len(values) - 1, int(math.ceil(rows[-1])))
        positions = _sample(bvh_positions(header, values[first:last + 1]), rows - first)
    grid = Path(textgrid) if textgrid else bvh.with_suffix(".TextGrid")
    words = []
    if grid.is_file():
        words = parse_textgrid(grid, fps)
        words = _scale_words(words, 1.0, offset=int(start_frame), frames=len(positions))
    take = bvh.stem
    speaker = speaker or take.split("_")[0]
    result = {"id": take, "speaker": str(speaker), "take": take, "fps": float(fps),
              "joint_names": [canonical_name(n) for n in header["names"]],
              "positions": positions.astype(np.float32), "words": words,
              "source": {"kind": "raw", "dataset": "beat", "version": "english_v0.2.1",
                         "bvh": str(bvh), "textgrid": str(grid) if grid.is_file() else None,
                         "take_frame_start": int(start_frame), "source_fps": round(source_fps, 6),
                         "axis_signs": [1, 1, 1], "mirrored_x": False,
                         "motion_url": HF_BASE + f"{speaker}/{take}.bvh",
                         "alignment_url": HF_BASE + f"{speaker}/{take}.TextGrid"},
              "basis": BASIS}
    return select_joints(result, joints) if joints else result


def load_take(descriptor, **options):
    if descriptor["kind"] == "processed":
        return load_processed(descriptor["folder"], descriptor["take"], **options)
    return load_bvh(descriptor["bvh"], descriptor.get("textgrid"), speaker=descriptor["speaker"], **options)


def iter_takes(source, *, kind="auto", speakers=None, takes=None, max_takes_per_speaker=None, **options):
    """Yield records for every matching take (loaded lazily, one at a time)."""
    for descriptor in list_takes(source, kind=kind, speakers=speakers, takes=takes,
                                 max_takes_per_speaker=max_takes_per_speaker):
        yield load_take(descriptor, **options)


# ----------------------------------------------------------------------------
# Record transforms

def resample(record, fps):
    """Linearly resample a record to ``fps`` (e.g. 30 -> 15) and rescale word frames."""
    if float(fps) == float(record["fps"]):
        return record
    positions = np.asarray(record["positions"], np.float32)
    rows = _frame_rows(float(record["fps"]), float(fps), len(positions))
    out = dict(record, fps=float(fps), positions=_sample(positions, rows).astype(np.float32))
    out["words"] = _scale_words(record["words"], float(fps) / float(record["fps"]), frames=len(rows))
    return out


def joint_indices(joint_names, names=UPPER_BODY):
    lookup = {canonical_name(n): i for i, n in enumerate(joint_names)}
    missing = [n for n in names if canonical_name(n) not in lookup]
    if missing:
        raise ValueError(f"Missing joints: {missing}")
    return [lookup[canonical_name(n)] for n in names]


def select_joints(record, names=UPPER_BODY):
    """Keep (and order) the named joints; Unity and BVH aliases are accepted."""
    index = joint_indices(record["joint_names"], names)
    return dict(record, joint_names=[canonical_name(n) for n in names],
                positions=np.asarray(record["positions"])[:, index])


def center_positions(positions, joint_names, joint="Neck", per_frame=True, axes=(0, 1, 2)):
    """Subtract a joint (Neck by default; Hips for root centring) per frame or by first frame."""
    positions = np.array(positions, dtype=np.float32, copy=True)
    index = joint_indices(joint_names, [joint])[0]
    anchor = positions[:, index:index + 1] if per_frame else positions[:1, index:index + 1]
    axes = list(axes)
    positions[..., axes] -= anchor[..., axes]
    return positions


def center(record, joint="Neck", per_frame=True, axes=(0, 1, 2)):
    return dict(record, positions=center_positions(record["positions"], record["joint_names"],
                                                   joint, per_frame, axes))


def motion_energy(positions, joint_names, fps, joints=("LeftHand", "RightHand")):
    """Mean wrist speed (units/s) after neck centring; near-static clips score low."""
    positions = np.asarray(positions, np.float32)
    if len(positions) < 2:
        return 0.0
    names = [canonical_name(n) for n in joint_names]
    neck = names.index("Neck") if "Neck" in names else None
    rel = positions - positions[:, neck:neck + 1] if neck is not None else positions
    index = [names.index(j) for j in joints if j in names] or list(range(len(names)))
    speed = np.linalg.norm(np.diff(rel[:, index], axis=0), axis=-1) * float(fps)
    return float(speed.mean())


# ----------------------------------------------------------------------------
# Speaker / take roles

def assign_roles(items, spec, seed=0):
    """Deterministically assign items (speakers or takes) to named roles.

    ``spec`` maps role -> explicit list of items, a fraction (0-1], an integer
    count, or ``"rest"``. Explicit lists are honoured first; remaining items are
    shuffled with ``seed`` and allocated in spec order. Returns {item: role};
    items left over are omitted. Roles never share an item.
    """
    items = sorted({str(i) for i in items}, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
    assigned = {}
    for role, value in spec.items():
        if isinstance(value, (list, tuple, set)):
            for item in map(str, value):
                if item not in items:
                    raise ValueError(f"Role {role}: unknown item {item}")
                if item in assigned:
                    raise ValueError(f"Item {item} assigned to both {assigned[item]} and {role}")
                assigned[item] = role
    remaining = [i for i in items if i not in assigned]
    order = np.random.default_rng(seed).permutation(len(remaining))
    remaining = [remaining[i] for i in order]
    pending = [(r, v) for r, v in spec.items() if not isinstance(v, (list, tuple, set))]
    total = len(remaining)
    for position, (role, value) in enumerate(pending):
        later = len(pending) - position - 1
        if value == "rest":
            count = len(remaining) - sum(1 for r, v in pending[position + 1:] if v != "rest")
        elif isinstance(value, float) and 0 < value <= 1:
            count = max(1, int(round(value * total)))
        else:
            count = int(value)
        count = max(0, min(count, len(remaining) - later if len(remaining) > later else len(remaining)))
        for item in remaining[:count]:
            assigned[item] = role
        remaining = remaining[count:]
    return assigned


def parse_role_spec(entries):
    """Parse CLI entries like ``library=1,2`` ``train=0.5`` ``wild=rest`` ``train=3``."""
    spec = {}
    for entry in entries or []:
        role, _, value = entry.partition("=")
        if not role or not value:
            raise ValueError(f"Role entry must be role=value: {entry}")
        value = value.strip()
        if value == "rest":
            spec[role] = "rest"
        elif re.fullmatch(r"0?\.\d+|1\.0+", value):
            spec[role] = float(value)
        elif value.startswith("n") and value[1:].isdigit():
            spec[role] = int(value[1:])
        else:
            spec[role] = [v for v in value.split(",") if v]
    return spec


def role_assignment(descriptors, spec, seed=0, unit="auto"):
    """Assign take descriptors to roles by speaker (preferred) or by take."""
    speakers = sorted({d["speaker"] for d in descriptors})
    explicit = [v for v in spec.values() if isinstance(v, (list, tuple, set))]
    if unit == "auto":
        if explicit:
            unit = "take" if any(str(x) not in speakers for v in explicit for x in v) else "speaker"
        else:
            unit = "speaker" if len(speakers) >= len(spec) else "take"
    key = (lambda d: d["speaker"]) if unit == "speaker" else (lambda d: d["take"])
    roles = assign_roles([key(d) for d in descriptors], spec, seed)
    return {d["take"]: roles.get(key(d)) for d in descriptors}, unit


# ----------------------------------------------------------------------------
# Windows, IDs and projection

def window_id(take, start, end):
    return f"{take}:{int(start)}-{int(end)}"


def windows(record, length, stride=None, *, min_words=1, word_rule="midpoint", offset=0):
    """Fixed windows over one record; windows with fewer than ``min_words`` are skipped.

    ``word_rule='midpoint'`` assigns each word to the window containing its
    midpoint, so adjacent windows never share a boundary word; ``'overlap'``
    keeps every overlapping word (the legacy contract behaviour).
    """
    stride = int(stride or length)
    positions = np.asarray(record["positions"])
    out = []
    for start in range(int(offset), len(positions) - int(length) + 1, stride):
        end = start + int(length)
        selected = []
        for word, s, e in record["words"]:
            inside = (start <= (s + e) / 2 < end) if word_rule == "midpoint" else (e > start and s < end)
            if inside:
                selected.append((word, max(0, s - start), min(int(length), e - start)))
        if len(selected) < min_words:
            continue
        out.append({"id": window_id(record["take"], start, end), "speaker": record["speaker"],
                    "take": record["take"], "start": start, "end": end, "fps": record["fps"],
                    "joint_names": record["joint_names"], "positions": positions[start:end],
                    "words": selected, "text": " ".join(w for w, _, _ in selected)})
    return out


def camera_matrix(yaw=0.0, pitch=0.0):
    """Rotation that orbits the camera by yaw (about Y) then pitch (about X), degrees."""
    return (euler_matrices("X", [pitch])[0] @ euler_matrices("Y", [yaw])[0]).astype(np.float64)


def project_2d(positions, yaw=0.0, pitch=0.0, focal=1.0, distance=3.0, pivot=None):
    """Project [...,T,J,3] through an explicit pinhole camera to [...,T,J,2].

    The camera sits ``distance`` units in front of the subject (+Z) looking
    back at the pivot (per-sequence mean joint position by default), orbited by
    ``yaw``/``pitch`` degrees. Image x grows to the camera's right (the
    subject's left), image y grows upward. Coordinates are rescaled by
    ``focal * distance / depth`` so that, at the pivot depth, one image unit is
    one world unit. ``distance=None`` (or inf) gives an orthographic camera.
    """
    p = np.asarray(positions, dtype=np.float64)
    if pivot is None:
        pivot = p.reshape(*p.shape[:-3], -1, 3).mean(axis=-2)[..., None, None, :]
    rotated = (p - pivot) @ camera_matrix(yaw, pitch).T
    if distance is None or not np.isfinite(distance):
        scale = focal
    else:
        depth = np.maximum(float(distance) - rotated[..., 2:3], 1e-3)
        scale = focal * float(distance) / depth
    return (rotated[..., :2] * scale).astype(np.float32)


def corrupt_2d(pose2d, seed=0, *, noise=0.02, jitter=1, dropout=0.05, fill="zero"):
    """OpenPose-like corruption of [T,J,2] (or [N,T,J,2]) 2D pose.

    ``noise`` is a Gaussian standard deviation relative to the sequence's RMS
    body extent; ``jitter`` resamples each frame from up to +/-jitter frames;
    ``dropout`` zeroes joints with that probability (confidence 0), or holds the
    previous value (``fill='hold'``) or writes NaN (``fill='nan'``).
    Returns (pose, confidence).
    """
    rng = np.random.default_rng(seed)
    pose = np.array(pose2d, dtype=np.float32, copy=True)
    batched = pose.ndim == 4
    if not batched:
        pose = pose[None]
    confidence = np.ones(pose.shape[:-1], np.float32)
    for n in range(len(pose)):
        seq = pose[n]
        frames = len(seq)
        if jitter and frames > 1:
            index = (np.arange(frames) + rng.integers(-int(jitter), int(jitter) + 1, frames)).clip(0, frames - 1)
            seq = seq[index]
        extent = float(np.sqrt(np.mean((seq - seq.reshape(-1, 2).mean(0)) ** 2))) or 1.0
        seq = seq + rng.normal(0, noise * extent, seq.shape).astype(np.float32)
        drop = rng.random(seq.shape[:2]) < dropout
        if fill == "hold":
            for t in range(1, frames):
                seq[t][drop[t]] = seq[t - 1][drop[t]]
        else:
            seq[drop] = np.nan if fill == "nan" else 0.0
        confidence[n][drop] = 0.0
        pose[n] = seq
    return (pose, confidence) if batched else (pose[0], confidence[0])


# ----------------------------------------------------------------------------
# Export helpers

def write_words_jsonl(words, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"word": w, "start_frame": int(s), "end_frame": int(e)}) + "\n"
                            for w, s, e in words), encoding="utf-8")
    return path


def read_words_jsonl(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return [(r["word"], int(r["start_frame"]), int(r["end_frame"])) for r in rows]


def save_record(record, folder):
    """Write <take>.npz (positions, joint_names, fps, speaker) and <take>.words.jsonl."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npz = folder / f"{record['take']}.npz"
    np.savez(npz, positions=np.asarray(record["positions"], np.float32),
             joint_names=np.asarray(record["joint_names"]), fps=np.asarray(record["fps"]),
             speaker=np.asarray(record["speaker"]), take=np.asarray(record["take"]),
             source_json=np.asarray(json.dumps(record["source"])))
    write_words_jsonl(record["words"], folder / f"{record['take']}.words.jsonl")
    return npz


def load_record(npz):
    npz = Path(npz)
    data = np.load(npz, allow_pickle=False)
    take = str(data["take"])
    words_path = npz.with_name(f"{take}.words.jsonl")
    return {"id": take, "speaker": str(data["speaker"]), "take": take, "fps": float(data["fps"]),
            "joint_names": [str(n) for n in data["joint_names"]], "positions": data["positions"],
            "words": read_words_jsonl(words_path) if words_path.is_file() else [],
            "source": json.loads(str(data["source_json"])), "basis": BASIS}


def _contract_records(source, roles_needed, *, kind="auto", speakers=None, takes=None,
                      max_takes_per_speaker=None, role_spec=None, role_unit="auto", seed=0,
                      fps=15, max_frames=None, joints=UPPER_BODY):
    descriptors = list_takes(source, kind=kind, speakers=speakers, takes=takes,
                             max_takes_per_speaker=max_takes_per_speaker)
    if not descriptors:
        raise ValueError("No BEAT takes matched the source and filters")
    spec = role_spec or {r: v for r, v in roles_needed.items()}
    roles, unit = role_assignment(descriptors, spec, seed, role_unit)
    grouped = {role: [] for role in spec}
    for d in descriptors:
        role = roles.get(d["take"])
        if role is None:
            continue
        record = load_take(d, fps=fps, max_frames=max_frames, joints=joints)
        grouped[role].append(record)
    for role in roles_needed:
        if not grouped.get(role):
            raise ValueError(f"Role {role} received no takes; add speakers/takes or adjust --role")
    assignment = {"unit": unit, "seed": seed,
                  "roles": {role: sorted({(r["speaker"] if unit == "speaker" else r["take"]) for r in recs})
                            for role, recs in grouped.items()},
                  "takes": {role: [r["take"] for r in recs] for role, recs in grouped.items()}}
    return grouped, assignment


def _to_units(positions, units):
    return np.asarray(positions, np.float32) * (100.0 if units == "cm" else 1.0)


def _neck_windows(records, length, units, min_words=1):
    out, skipped = [], 0
    for record in records:
        centred = center(record, "Neck")
        found = windows(centred, length, min_words=min_words)
        skipped += max(0, (len(record["positions"]) // length) - len(found))
        for w in found:
            w["positions"] = _to_units(w["positions"], units)
        out.extend(found)
    return out, skipped


def _joint_order(path, fps, units, camera, assignment, extra=None):
    payload = {"schema": SCHEMA, "joints": list(UPPER_BODY), "fps": fps, "units": units,
               "projection": camera, "centering": "per-frame Neck", "roles": assignment,
               "basis": BASIS.replace("metres", units if units != "m" else "metres"),
               "note": "Projected BEAT motion stands in for video pose; it is not paper data."}
    payload.update(extra or {})
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _camera(options):
    return {"yaw": float(options.get("yaw", 20.0)), "pitch": float(options.get("pitch", 5.0)),
            "focal": float(options.get("focal", 1.0)), "distance": options.get("distance", 3.0)}


def _corrupt(options):
    return {"noise": float(options.get("noise", 0.02)), "jitter": int(options.get("jitter", 1)),
            "dropout": float(options.get("dropout", 0.05))}


def _project_windows(items, camera, corruption=None, seed=0, units="cm"):
    """Project windows; positions are already in ``units``. Distance is metres."""
    scale = 100.0 if units == "cm" else 1.0
    distance = camera["distance"]
    distance = None if distance is None else float(distance) * scale
    poses, confidences = [], []
    for i, w in enumerate(items):
        yaw = camera["yaw"](i) if callable(camera["yaw"]) else camera["yaw"]
        pose = project_2d(w["positions"], yaw, camera["pitch"], camera["focal"], distance)
        conf = np.ones(pose.shape[:-1], np.float32)
        if corruption:
            pose, conf = corrupt_2d(pose, seed + i, **corruption)
        poses.append(pose); confidences.append(conf)
    return poses, confidences


def export_automatic(source, output_dir, *, fps=15, unit_seconds=3.0, units="cm", seed=0,
                     camera=None, corruption=None, **select):
    """Automatic Text-to-Gesture contract: bank.npz (library) + video.npz (held-out video role)."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    grouped, assignment = _contract_records(source, DEFAULT_ROLES["automatic"], seed=seed, fps=fps, **select)
    length = int(round(fps * unit_seconds))
    bank, bank_skipped = _neck_windows(grouped["library"], length, units)
    if not bank:
        raise ValueError("Library role produced no worded windows")
    frontal = {"yaw": 0.0, "pitch": 0.0, "focal": 1.0, "distance": None}
    bank_pose, _ = _project_windows(bank, frontal, None, seed, units)
    np.savez(output / "bank.npz", **{w["id"]: p for w, p in zip(bank, bank_pose)})
    cam = _camera(camera or {}); bad = _corrupt(corruption or {})
    streams, confs, words, segments, cursor = [], [], [], [], 0
    for i, record in enumerate(grouped["video"]):
        centred = center(record, "Neck")
        positions = _to_units(centred["positions"], units)
        distance = None if cam["distance"] is None else float(cam["distance"]) * (100.0 if units == "cm" else 1.0)
        pose = project_2d(positions, cam["yaw"], cam["pitch"], cam["focal"], distance)
        pose, conf = corrupt_2d(pose, seed + 1000 + i, **bad)
        streams.append(pose); confs.append(conf)
        words.extend({"word": w, "start_frame": s + cursor, "end_frame": e + cursor} for w, s, e in record["words"])
        segments.append({"take": record["take"], "speaker": record["speaker"], "start_frame": cursor,
                         "end_frame": cursor + len(pose)})
        cursor += len(pose)
    np.savez(output / "video.npz", pose=np.concatenate(streams), confidence=np.concatenate(confs),
             words_json=np.asarray(json.dumps(words)), segments_json=np.asarray(json.dumps(segments)))
    _joint_order(output / "joint_order.json", fps, units,
                 {"bank": "orthographic frontal (yaw 0)", "video": {**cam, **bad, "distance_m": cam["distance"]}},
                 assignment, {"bank_ids": [w["id"] for w in bank]})
    return {"contract": "automatic", "bank_units": len(bank), "skipped_wordless_windows": bank_skipped,
            "video_frames": cursor, "video_words": len(words), "roles": assignment["roles"]}


def _pose_contract(source, output, mode, *, fps, unit_seconds, units, seed, camera, corruption,
                   train_yaw=30.0, **select):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    grouped, assignment = _contract_records(source, DEFAULT_ROLES[mode], seed=seed, fps=fps, **select)
    length = int(round(fps * unit_seconds))
    library, skip_library = _neck_windows(grouped["library"], length, units)
    train, skip_train = _neck_windows(grouped["train"], length, units)
    wild, skip_wild = _neck_windows(grouped["wild"], length, units)
    if not library or not train or not wild:
        raise ValueError("Every role needs at least one worded window")
    rng = np.random.default_rng(seed)
    yaws = rng.uniform(-train_yaw, train_yaw, len(train))
    train_pose, _ = _project_windows(train, {"yaw": lambda i: float(yaws[i]), "pitch": 0.0, "focal": 1.0,
                                             "distance": (camera or {}).get("distance", 3.0)}, None, seed, units)
    cam = _camera(camera or {}); bad = _corrupt(corruption or {})
    wild_pose, wild_conf = _project_windows(wild, cam, bad, seed + 1000, units)
    flat3 = lambda items: np.stack([w["positions"].reshape(length, -1) for w in items])
    flat2 = lambda poses: np.stack([p.reshape(length, -1) for p in poses])
    dim2 = len(UPPER_BODY) * 2
    np.savez(output / "pairs.npz", pose2d=flat2(train_pose), motion3d=flat3(train),
             ids=np.asarray([w["id"] for w in train]), texts=np.asarray([w["text"] for w in train]),
             yaw=yaws.astype(np.float32))
    np.savez(output / "units.npz", motion3d=flat3(library), ids=np.asarray([w["id"] for w in library]),
             dim2=np.asarray(dim2), texts=np.asarray([w["text"] for w in library]),
             speakers=np.asarray([w["speaker"] for w in library]))
    np.savez(output / "wild.npz", pose2d=flat2(wild_pose), texts=np.asarray([w["text"] for w in wild]),
             ids=np.asarray([w["id"] for w in wild]), confidence=np.stack(wild_conf))
    summary = {"contract": mode, "library_units": len(library), "train_pairs": len(train),
               "wild_windows": len(wild), "skipped_wordless_windows": skip_library + skip_train + skip_wild,
               "roles": assignment["roles"]}
    if mode == "multilingual":
        first = grouped["library"][0]["speaker"]
        motion = np.concatenate([_to_units(center(r, "Neck")["positions"], units).reshape(len(r["positions"]), -1)
                                 for r in grouped["library"] if r["speaker"] == first])
        np.save(output / "speaker_motion.npy", motion)
        summary["speaker_motion_frames"] = len(motion)
    _joint_order(output / "joint_order.json", fps, units,
                 {"train": f"pinhole, yaw uniform in +/-{train_yaw} deg, clean", "wild": {**cam, **bad, "distance_m": cam["distance"]},
                  "units": "orthographic-equivalent scale at pivot depth"}, assignment)
    return summary


def export_wild(source, output_dir, *, fps=15, unit_seconds=3.0, units="cm", seed=0, camera=None,
                corruption=None, **select):
    """Wild Pose Matching contract: units.npz (library), pairs.npz (train), wild.npz (held-out, corrupted 2D)."""
    return _pose_contract(source, output_dir, "wild", fps=fps, unit_seconds=unit_seconds, units=units,
                          seed=seed, camera=camera, corruption=corruption, **select)


def export_multilingual(source, output_dir, *, fps=15, unit_seconds=3.0, units="cm", seed=0, camera=None,
                        corruption=None, **select):
    """Multilingual Gesture contract: wild contract plus speaker_motion.npy from the library speaker."""
    return _pose_contract(source, output_dir, "multilingual", fps=fps, unit_seconds=unit_seconds, units=units,
                          seed=seed, camera=camera, corruption=corruption, **select)


def export_ridge(source, output_dir, *, fps=15, unit_seconds=3.0, units="cm", seed=0, sbert=None, **select):
    """RIDGE contract: train_pairs.npz (train), heldout_pairs.npz, speaker_motion.npy, transcripts.jsonl (library)."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    grouped, assignment = _contract_records(source, DEFAULT_ROLES["ridge"], seed=seed, fps=fps, **select)
    length = int(round(fps * unit_seconds))
    train, skip_train = _neck_windows(grouped["train"], length, units)
    heldout, skip_held = _neck_windows(grouped["heldout"], length, units)
    if len(train) < 2:
        raise ValueError("RIDGE training needs at least two worded windows")
    encoder = None
    if sbert:
        from sentence_transformers import SentenceTransformer  # explicit, user-selected model path/name
        encoder = SentenceTransformer(str(sbert))
    def pairs(path, items):
        payload = {"motion": np.stack([w["positions"].reshape(length, -1) for w in items]),
                   "ids": np.asarray([w["id"] for w in items]), "texts": np.asarray([w["text"] for w in items]),
                   "speakers": np.asarray([w["speaker"] for w in items])}
        if encoder is not None:
            payload["text_embeddings"] = encoder.encode([w["text"] for w in items],
                                                        normalize_embeddings=True).astype(np.float32)
        np.savez(path, **payload)
    pairs(output / "train_pairs.npz", train)
    if heldout:
        pairs(output / "heldout_pairs.npz", heldout)
    library = grouped["library"]
    first = library[0]["speaker"]
    motion = np.concatenate([_to_units(center(r, "Neck")["positions"], units).reshape(len(r["positions"]), -1)
                             for r in library if r["speaker"] == first])
    np.save(output / "speaker_motion.npy", motion)
    with (output / "transcripts.jsonl").open("w", encoding="utf-8") as stream:
        for r in library:
            stream.write(json.dumps({"record_id": r["take"], "speaker": r["speaker"],
                                     "text": " ".join(w for w, _, _ in r["words"]),
                                     "words": [{"word": w, "start_frame": s, "end_frame": e} for w, s, e in r["words"]]}) + "\n")
    _joint_order(output / "joint_order.json", fps, units, "none (3D motion)", assignment,
                 {"text_embeddings": f"sentence-transformers:{sbert}" if sbert else None})
    return {"contract": "ridge", "train_pairs": len(train), "heldout_pairs": len(heldout),
            "transcript_records": len(library), "skipped_wordless_windows": skip_train + skip_held,
            "text_embeddings": bool(encoder), "roles": assignment["roles"]}


EXPORTS = {"automatic": export_automatic, "wild": export_wild,
           "multilingual": export_multilingual, "ridge": export_ridge}


# ----------------------------------------------------------------------------
# CLI

def _selection_args(parser):
    parser.add_argument("--source", type=Path, required=True,
                        help="Processed OmniMo BEAT root (<speaker>/meta.json) or raw beat_english_v0.2.1 folder")
    parser.add_argument("--kind", choices=["auto", "processed", "raw"], default="auto")
    parser.add_argument("--speakers", help="Comma list of speaker ids or names (e.g. 1,2,wayne)")
    parser.add_argument("--takes", help="Comma list of take ids or glob patterns (e.g. '*_0_1_1')")
    parser.add_argument("--max-takes-per-speaker", type=int)
    parser.add_argument("--fps", type=float, default=15, help="Output frame rate (30 or 15)")
    parser.add_argument("--max-frames", type=int, help="Cap frames per take (quick tests)")


def _selection(args):
    return {"kind": args.kind, "speakers": args.speakers, "takes": args.takes,
            "max_takes_per_speaker": args.max_takes_per_speaker}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List matching takes")
    _selection_args(listing)
    ingest = sub.add_parser("ingest", help="Write per-take NPZ + words JSONL records")
    _selection_args(ingest)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--upper-body", action="store_true", help="Keep only the 11 canonical upper-body joints")
    ingest.add_argument("--center", choices=["none", "neck", "root"], default="none")
    roles = sub.add_parser("roles", help="Print the deterministic role assignment")
    _selection_args(roles)
    roles.add_argument("--role", action="append", help="role=ids|fraction|nCOUNT|rest; repeatable")
    roles.add_argument("--role-unit", choices=["auto", "speaker", "take"], default="auto")
    roles.add_argument("--seed", type=int, default=0)
    for name in EXPORTS:
        export = sub.add_parser(f"export-{name}", help=f"Write the {name} repository data contract")
        _selection_args(export)
        export.add_argument("--output-dir", type=Path, required=True)
        export.add_argument("--unit-seconds", type=float, default=3.0)
        export.add_argument("--units", choices=["cm", "m"], default="cm",
                            help="Contract position units (cm matches the existing BVH contracts)")
        export.add_argument("--role", action="append",
                            help=f"Override roles ({', '.join(DEFAULT_ROLES[name])}): role=ids|fraction|nCOUNT|rest")
        export.add_argument("--role-unit", choices=["auto", "speaker", "take"], default="auto")
        export.add_argument("--seed", type=int, default=0)
        if name == "ridge":
            export.add_argument("--sbert", help="Local path or name of a Sentence-BERT model for text_embeddings "
                                               "(omitted: texts only, no implicit download)")
        else:
            export.add_argument("--yaw", type=float, default=20.0)
            export.add_argument("--pitch", type=float, default=5.0)
            export.add_argument("--focal", type=float, default=1.0)
            export.add_argument("--distance", type=float, default=3.0, help="Camera distance in metres")
            export.add_argument("--noise", type=float, default=0.02)
            export.add_argument("--jitter", type=int, default=1)
            export.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.command == "list":
        rows = list_takes(args.source, **_selection(args))
        print(json.dumps({"takes": len(rows), "speakers": sorted({r["speaker"] for r in rows}),
                          "items": rows}, indent=1))
        return
    if args.command == "roles":
        rows = list_takes(args.source, **_selection(args))
        mapping, unit = role_assignment(rows, parse_role_spec(args.role) or DEFAULT_ROLES["wild"], args.seed, args.role_unit)
        print(json.dumps({"unit": unit, "takes": mapping}, indent=1))
        return
    if args.command == "ingest":
        manifest = []
        for record in iter_takes(args.source, fps=args.fps, max_frames=args.max_frames,
                                 joints=UPPER_BODY if args.upper_body else None, **_selection(args)):
            if args.center != "none":
                record = center(record, "Neck" if args.center == "neck" else "Hips",
                                axes=(0, 1, 2) if args.center == "neck" else (0, 2))
            path = save_record(record, args.output)
            manifest.append({"take": record["take"], "speaker": record["speaker"], "frames": len(record["positions"]),
                             "words": len(record["words"]), "file": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (args.output / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
        print(json.dumps({"schema": SCHEMA, "output": str(args.output), "takes": len(manifest)}))
        return
    name = args.command.split("-", 1)[1]
    options = dict(fps=args.fps, unit_seconds=args.unit_seconds, units=args.units, seed=args.seed,
                   role_spec=parse_role_spec(args.role) or None, role_unit=args.role_unit,
                   max_frames=args.max_frames, **_selection(args))
    if name == "ridge":
        options["sbert"] = args.sbert
    else:
        options["camera"] = {"yaw": args.yaw, "pitch": args.pitch, "focal": args.focal, "distance": args.distance}
        options["corruption"] = {"noise": args.noise, "jitter": args.jitter, "dropout": args.dropout}
    print(json.dumps(EXPORTS[name](args.source, args.output_dir, **options)))


if __name__ == "__main__":
    main()
