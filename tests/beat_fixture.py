"""Tiny synthetic BEAT-shaped fixtures for the paper-method preparation tests (no downloads).

``make_processed`` writes an OmniMo-style collection (<speaker>/meta.json +
motion.npz with 6D local rotations); ``make_raw`` writes the same takes as raw
BEAT BVH + TextGrid files. Words are tied to arm-motion topics so the learned
pipelines see a real (if trivial) text-motion association.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JOINTS = ["Hips", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
PARENTS = [-1, 0, 1, 1, 3, 4, 5, 1, 7, 8, 9]
OFFSETS = np.array([[0, 0, 0], [0, .5, 0], [0, .15, 0], [.05, 0, 0], [.12, 0, 0], [.28, 0, 0], [.25, 0, 0],
                    [-.05, 0, 0], [-.12, 0, 0], [-.28, 0, 0], [-.25, 0, 0]], float)
NAMES = {"1": "wayne", "2": "scott", "3": "solomon", "4": "lawrence", "5": "stewart", "6": "carla"}
TOPICS = [["open", "hands", "welcome", "everyone", "together"], ["point", "there", "result", "look", "right"],
          ["think", "idea", "important", "maybe", "consider"], ["stop", "never", "wait", "again", "enough"]]
FILLER = ["the", "and", "so", "we", "can", "really", "now"]
VOCABULARY = sorted({w for t in TOPICS for w in t} | set(FILLER))
# Per-topic (left arm Z, right arm Z, left forearm X, right forearm X) base angles and oscillation.
POSES = [(-30, 30, 10, 10), (-75, -10, 0, 60), (-70, 70, 80, 80), (-20, 20, -40, -40)]


def _angles(fps, seconds, seed):
    """Per-frame Euler angles (degrees) per joint and the timed words of one take."""
    rng = np.random.default_rng(seed)
    frames = int(fps * seconds)
    seg = int(fps * 1.5)
    z = np.zeros((frames, len(JOINTS))); x = np.zeros_like(z)
    words = []
    for s in range(0, frames, seg):
        topic = int(rng.integers(len(TOPICS)))
        t = np.arange(min(seg, frames - s)) / fps
        lz, rz, lx, rx = POSES[topic]
        wave = 15 * np.sin(2 * np.pi * (0.8 + 0.4 * topic) * t + rng.uniform(0, 3))
        z[s:s + len(t), 4] = lz + wave
        z[s:s + len(t), 8] = rz - wave
        x[s:s + len(t), 5] = lx + 0.5 * wave
        x[s:s + len(t), 9] = rx + 0.5 * wave
        chosen = [TOPICS[topic][int(i)] for i in rng.choice(5, 2, replace=False)] + list(rng.choice(FILLER, 2))
        rng.shuffle(chosen)
        step = len(t) // 5
        for k, word in enumerate(chosen):
            words.append((str(word), s + 2 + k * step, s + 2 + k * step + step - 1))
    return z, x, words


def _matrix(z, x):
    from_z = np.deg2rad(z); from_x = np.deg2rad(x)
    cz, sz, cx, sx = np.cos(from_z), np.sin(from_z), np.cos(from_x), np.sin(from_x)
    one, zero = np.ones_like(cz), np.zeros_like(cz)
    rz = np.stack([np.stack([cz, -sz, zero], -1), np.stack([sz, cz, zero], -1), np.stack([zero, zero, one], -1)], -2)
    rx = np.stack([np.stack([one, zero, zero], -1), np.stack([zero, cx, -sx], -1), np.stack([zero, sx, cx], -1)], -2)
    return rz @ rx


def make_processed(root, speakers=("1", "2", "3", "4", "5", "6"), takes=1, seconds=30, fps=30):
    root = Path(root)
    for index, spk in enumerate(speakers):
        folder = root / spk; folder.mkdir(parents=True, exist_ok=True)
        rotations, meta_takes, cursor = [], [], 0
        for k in range(1, takes + 1):
            z, x, words = _angles(fps, seconds, seed=100 * index + k)
            matrix = _matrix(z, x)
            rotations.append(np.concatenate([matrix[..., 0], matrix[..., 1]], -1))
            n = len(z)
            meta_takes.append({"sequence_id": f"{spk}_{NAMES[spk]}_0_{k}_{k}", "frame_start": cursor, "frame_end": cursor + n,
                               "words": [{"text": w, "frame_start": s, "frame_end": e} for w, s, e in words]})
            cursor += n
        rot = np.concatenate(rotations).astype(np.float32)
        root_pos = np.tile(np.array([0, 1.0, 0], np.float32), (len(rot), 1))
        np.savez(folder / "motion.npz", rotations=rot.reshape(len(rot), -1), root=root_pos,
                 parents=np.asarray(PARENTS), offsets=OFFSETS.astype(np.float32))
        meta = {"dataset": "beat", "version": "synthetic-fixture", "speaker": {"id": spk, "name": NAMES[spk]},
                "fps": fps, "joints": JOINTS, "total_frames": cursor, "takes": meta_takes}
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def _textgrid(words, fps, duration):
    rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration}",
            "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
            '        name = "words"', "        xmin = 0", f"        xmax = {duration}",
            f"        intervals: size = {len(words)}"]
    for i, (w, s, e) in enumerate(words, 1):
        rows += [f"        intervals [{i}]:", f"            xmin = {s / fps:.4f}", f"            xmax = {e / fps:.4f}",
                 f'            text = "{w}"']
    return "\n".join(rows) + "\n"


def make_raw(root, speakers=("1", "2", "3", "4", "5", "6"), takes=1, seconds=30, fps=30):
    """Raw BEAT layout: <root>/<speaker>/<take>.bvh + .TextGrid (BVH in centimetres)."""
    root = Path(root)
    for index, spk in enumerate(speakers):
        folder = root / spk; folder.mkdir(parents=True, exist_ok=True)
        for k in range(1, takes + 1):
            take = f"{spk}_{NAMES[spk]}_0_{k}_{k}"
            z, x, words = _angles(fps, seconds, seed=100 * index + k)
            lines = ["HIERARCHY"]
            def joint(j, depth):
                pad = "  " * depth
                kind = "ROOT" if PARENTS[j] < 0 else "JOINT"
                off = " ".join(f"{v * 100:.4f}" for v in OFFSETS[j])
                channels = ("CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if PARENTS[j] < 0
                            else "CHANNELS 3 Zrotation Xrotation Yrotation")
                lines.extend([f"{pad}{kind} {JOINTS[j]}", f"{pad}{{", f"{pad}  OFFSET {off}", f"{pad}  {channels}"])
                children = [c for c, p in enumerate(PARENTS) if p == j]
                for c in children:
                    joint(c, depth + 1)
                if not children:
                    lines.extend([f"{pad}  End Site", f"{pad}  {{", f"{pad}    OFFSET 0 5 0", f"{pad}  }}"])
                lines.append(f"{pad}}}")
            joint(0, 0)
            lines += ["MOTION", f"Frames: {len(z)}", f"Frame Time: {1 / fps:.6f}"]
            for t in range(len(z)):
                values = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
                for j in range(1, len(JOINTS)):
                    values += [z[t, j], x[t, j], 0.0]
                lines.append(" ".join(f"{v:.4f}" for v in values))
            (folder / f"{take}.bvh").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (folder / f"{take}.TextGrid").write_text(_textgrid(words, fps, len(z) / fps), encoding="utf-8")
    return root


def make_sbert(path, vocabulary=VOCABULARY, width=32):
    """A tiny local bag-of-words Sentence-Transformers model (no download)."""
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize
    except ImportError:  # older sentence-transformers
        from sentence_transformers.models import BoW, Dense, Normalize
    torch.manual_seed(5)
    SentenceTransformer(modules=[BoW(list(vocabulary)), Dense(len(vocabulary), width), Normalize()]).save_pretrained(str(path))
    return Path(path)


def make_glove(path, vocabulary=VOCABULARY, dim=12, seed=3):
    """GloVe-format text vectors: topic words share an axis, fillers are small noise."""
    rng = np.random.default_rng(seed)
    lines = []
    for word in vocabulary:
        vector = rng.normal(0, 0.05, dim)
        for t, words in enumerate(TOPICS):
            if word in words:
                vector[t] += 1.0
        lines.append(word + " " + " ".join(f"{v:.5f}" for v in vector))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)
