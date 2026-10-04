from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Rule:
    phrase: str
    gesture_id: str
    score: float
    start_frame: int
    end_frame: int
    source: str = "automatic"


def normalize_pose(pose: np.ndarray, neck_joint: int = 1) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[-1] != 2:
        raise ValueError("pose must have shape [frames, joints, 2]")
    if not 0 <= neck_joint < pose.shape[1]:
        raise ValueError("neck_joint is outside the skeleton")
    return pose - pose[:, neck_joint : neck_joint + 1]


def center_pad(pose: np.ndarray, frames: int) -> np.ndarray:
    if frames < 1 or len(pose) < 1:
        raise ValueError("poses and target length must be non-empty")
    if len(pose) == frames:
        return pose
    if len(pose) > frames:
        raise ValueError("gesture is longer than the common mining window; resample FPS before mining")
    missing=frames-len(pose); before=missing//2; after=missing-before
    return np.pad(pose,((before,after),(0,0),(0,0)),mode="constant").astype(np.float32)


def average_frame_cosine(a: np.ndarray, b: np.ndarray) -> float:
    b = center_pad(b, len(a))
    av, bv = a.reshape(len(a), -1), b.reshape(len(a), -1)
    denom = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    sims = np.divide((av * bv).sum(1), denom, out=np.zeros_like(denom), where=denom > 1e-8)
    return float(sims.mean())


def aligned_phrase(words: list[dict], start: int, end: int, max_words: int = 5) -> str:
    selected = [w["word"] for w in words if int(w["end_frame"]) > start and int(w["start_frame"]) < end]
    if len(selected) > max_words:
        trim = len(selected) - max_words
        selected = selected[trim // 2 : trim // 2 + max_words]
    return " ".join(selected)


def mine_rules(video_pose: np.ndarray, words: list[dict], bank: Mapping[str, np.ndarray], threshold: float = .92, seed: int = 0, source: str = "clip") -> list[Rule]:
    if not bank:
        raise ValueError("gesture bank is empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    rng = np.random.default_rng(seed)
    norm_video = normalize_pose(video_pose)
    stride = max(len(v) for v in bank.values())
    normalized_bank = {k: normalize_pose(v) for k, v in bank.items()}
    if any(p.shape[1] != norm_video.shape[1] for p in normalized_bank.values()):
        raise ValueError("video and bank must use the same joint count")
    last_end = 0
    for word in words:
        start, end = int(word["start_frame"]), int(word["end_frame"])
        if start < last_end or end <= start or end > len(norm_video):
            raise ValueError("word frame intervals must be ordered, positive and inside the video")
        last_end = end
    rules: list[Rule] = []
    for start in range(0, max(0, len(norm_video) - stride + 1), stride):
        window = norm_video[start : start + stride]
        candidates = [(gid, average_frame_cosine(window, pose)) for gid, pose in normalized_bank.items()]
        passing = [(gid, score) for gid, score in candidates if score >= threshold]
        phrase = aligned_phrase(words, start, start + stride)
        if passing and phrase:
            gid, score = passing[int(rng.integers(len(passing)))]
            rules.append(Rule(phrase, gid, score, start, start + stride, source))
    return rules


TOKEN = re.compile(r"[\w']+")


def phrase_vector(text: str, vectors: Mapping[str, np.ndarray], dim: int | None = None) -> np.ndarray:
    if dim is None:
        dim = len(next(iter(vectors.values()))) if vectors else 300
    out = np.zeros(dim, np.float32)
    for token in TOKEN.findall(text.lower()):
        if token in vectors:
            out += vectors[token]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else -1.0


def chunks(text: str, size: int = 5) -> list[str]:
    tokens = TOKEN.findall(text)
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1, size)]


def retrieve(text: str, rules: list[Rule], vectors: Mapping[str, np.ndarray], audio_seconds: float | None = None, manual_first: bool = True) -> list[dict]:
    if not rules:
        raise ValueError("rule map is empty")
    if audio_seconds is not None and audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    result = []
    segments = chunks(text)
    slot = audio_seconds / len(segments) if audio_seconds is not None and segments else None
    rule_vecs = [phrase_vector(r.phrase, vectors) for r in rules]
    for index, segment in enumerate(segments):
        exact = [i for i, r in enumerate(rules) if r.source == "manual" and r.phrase.casefold() == segment.casefold()]
        scores = [cosine(phrase_vector(segment, vectors), rv) for rv in rule_vecs]
        if not exact and max(scores) < 0:
            raise ValueError(f"no GloVe vocabulary overlap for text chunk: {segment}")
        chosen = exact[0] if manual_first and exact else int(np.argmax(scores))
        result.append({"text": segment, "gesture_id": rules[chosen].gesture_id, "similarity": scores[chosen], "start_seconds": index * slot if slot else None, "duration_seconds": slot})
    return result


def load_glove(path: str | Path, required_words: set[str] | None = None) -> dict[str, np.ndarray]:
    vectors = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            word, *values = line.rstrip().split()
            if required_words is None or word in required_words:
                vectors[word] = np.asarray(values, dtype=np.float32)
    dims = {len(v) for v in vectors.values()}
    if len(dims) != 1:
        raise ValueError("inconsistent embedding dimensions")
    return vectors


def write_rules(path: str | Path, rules: Iterable[Rule]) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rule in rules:
            fh.write(json.dumps(asdict(rule)) + "\n")


def read_rules(path: str | Path) -> list[Rule]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Rule(**json.loads(line)) for line in fh if line.strip()]
