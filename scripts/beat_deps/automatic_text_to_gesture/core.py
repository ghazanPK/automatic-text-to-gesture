from __future__ import annotations

import csv
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# Paper defaults (Ali, Lee and Hwang 2020). Every value is overridable from the CLI or a JSON config.
DEFAULTS: dict = {
    "threshold": 0.92,      # frame-cosine cut-off (Algorithm 1)
    "phrase_words": 5,      # maximum aligned phrase length, trimmed from both sides
    "chunk_words": 5,       # runtime text chunk size (Algorithm 2)
    "neck_joint": 1,        # pose normalisation centre
    "seed": 0,              # random selection among passing gestures
    "oov": "idle",          # chunk without vocabulary: idle | skip | error
    "idle_id": "idle",      # gesture id emitted for idle slots
}


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


def center_pad(pose: np.ndarray, frames: int, return_mask: bool = False):
    """Zero-pad a gesture on both sides to ``frames``; optionally return the real-frame mask."""
    if frames < 1 or len(pose) < 1:
        raise ValueError("poses and target length must be non-empty")
    if len(pose) > frames:
        raise ValueError("gesture is longer than the common mining window; resample FPS before mining")
    missing = frames - len(pose)
    before = missing // 2
    padded = np.pad(pose, ((before, missing - before), (0, 0), (0, 0)), mode="constant").astype(np.float32)
    if not return_mask:
        return padded
    mask = np.zeros(frames, bool)
    mask[before : before + len(pose)] = True
    return padded, mask


def _real_frames(padded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gesture frames that carry data: inside the centre-pad mask and not all-zero (pre-padded banks)."""
    flat = padded.reshape(len(padded), -1)
    return mask & (np.linalg.norm(flat, axis=1) > 1e-8)


def average_frame_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean frame-by-frame cosine between window ``a`` and gesture ``b`` over the gesture's real frames.

    ``b`` is centre-padded to ``len(a)``. Padding frames (added here, or all-zero frames of a
    bank that was already padded to a common length) are excluded from the mean, so a short
    gesture embedded in a longer window can still score 1.0.
    """
    b, mask = center_pad(b, len(a), return_mask=True)
    valid = _real_frames(b, mask)
    if not valid.any():
        return 0.0
    av, bv = a.reshape(len(a), -1), b.reshape(len(a), -1)
    denom = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    sims = np.divide((av * bv).sum(1), denom, out=np.zeros_like(denom), where=denom > 1e-8)
    return float(sims[valid].mean())


class GestureBank:
    """Neck-normalised gesture bank padded to the longest gesture, with per-gesture real-frame masks."""

    def __init__(self, bank: Mapping[str, np.ndarray], neck_joint: int = 1):
        if not bank:
            raise ValueError("gesture bank is empty")
        self.ids = [str(k) for k in bank]
        poses = [normalize_pose(bank[k], neck_joint) for k in bank]
        joints = {p.shape[1] for p in poses}
        if len(joints) != 1:
            raise ValueError("all bank gestures must use the same joint count")
        self.joints = joints.pop()
        self.stride = max(len(p) for p in poses)
        padded, valid = [], []
        for p in poses:
            q, m = center_pad(p, self.stride, return_mask=True)
            padded.append(q)
            valid.append(_real_frames(q, m))
        flat = np.stack(padded).reshape(len(poses), self.stride, -1)
        norms = np.linalg.norm(flat, axis=2, keepdims=True)
        self.unit = np.divide(flat, norms, out=np.zeros_like(flat), where=norms > 1e-8)
        self.valid = np.stack(valid).astype(np.float32)
        self.count = self.valid.sum(1)

    def scores(self, window: np.ndarray) -> np.ndarray:
        """Padding-aware mean frame cosine of one normalised window against every gesture."""
        flat = window.reshape(len(window), -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        unit = np.divide(flat, norms, out=np.zeros_like(flat), where=norms > 1e-8)
        frame = np.einsum("sd,gsd->gs", unit, self.unit)
        return np.divide((frame * self.valid).sum(1), self.count, out=np.zeros(len(self.ids), np.float32), where=self.count > 0)


def aligned_phrase(words: list[dict], start: int, end: int, max_words: int = 5) -> str:
    selected = [w["word"] for w in words if int(w["end_frame"]) > start and int(w["start_frame"]) < end]
    if len(selected) > max_words:
        trim = len(selected) - max_words
        selected = selected[trim // 2 : trim // 2 + max_words]
    return " ".join(selected)


def _check_words(words: list[dict], frames: int) -> None:
    last_end = 0
    for word in words:
        start, end = int(word["start_frame"]), int(word["end_frame"])
        if start < last_end or end <= start or end > frames:
            raise ValueError("word frame intervals must be ordered, positive and inside the video")
        last_end = end


def window_scores(video_pose: np.ndarray, bank: GestureBank | Mapping[str, np.ndarray], neck_joint: int = 1) -> tuple[list[int], np.ndarray]:
    """Slide the bank over a clip with stride = bank length; return window starts and a [W, G] score matrix."""
    gb = bank if isinstance(bank, GestureBank) else GestureBank(bank, neck_joint)
    video = normalize_pose(video_pose, neck_joint)
    if video.shape[1] != gb.joints:
        raise ValueError("video and bank must use the same joint count")
    starts = list(range(0, max(0, len(video) - gb.stride + 1), gb.stride))
    scores = np.stack([gb.scores(video[s : s + gb.stride]) for s in starts]) if starts else np.zeros((0, len(gb.ids)), np.float32)
    return starts, scores


def mine_rules(video_pose: np.ndarray, words: list[dict], bank: Mapping[str, np.ndarray] | GestureBank, threshold: float = .92, seed: int = 0, source: str = "clip", *,
               phrase_words: int = 5, neck_joint: int = 1, rng: np.random.Generator | None = None, scores: tuple[list[int], np.ndarray] | None = None) -> list[Rule]:
    """Algorithm 1 inner loop for one clip: windows at stride = gesture length, random pick above threshold."""
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be a cosine value in [-1, 1]")
    gb = bank if isinstance(bank, GestureBank) else GestureBank(bank, neck_joint)
    _check_words(words, len(video_pose))
    rng = rng if rng is not None else np.random.default_rng(seed)
    starts, matrix = scores if scores is not None else window_scores(video_pose, gb, neck_joint)
    rules: list[Rule] = []
    for start, row in zip(starts, matrix):
        passing = np.flatnonzero(row >= threshold)
        phrase = aligned_phrase(words, start, start + gb.stride, phrase_words)
        if len(passing) and phrase:
            j = int(passing[int(rng.integers(len(passing)))])
            rules.append(Rule(phrase, gb.ids[j], float(row[j]), start, start + gb.stride, source))
    return rules


@dataclass(frozen=True)
class Clip:
    source: str
    pose: np.ndarray
    words: list


def threshold_from_percentile(score_matrices: Sequence[np.ndarray], percentile: float) -> float:
    """Threshold at a percentile of all window-by-gesture scores (e.g. 95 keeps about 5% of the bank per window)."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    values = np.concatenate([m.ravel() for m in score_matrices if m.size])
    if not values.size:
        raise ValueError("no mining windows: clips are shorter than the longest gesture")
    return float(np.percentile(values, percentile))


def calibration_report(score_matrices: Sequence[np.ndarray], threshold: float, sources: Sequence[str] | None = None,
                       grid: Iterable[float] | None = None, percentiles: Iterable[float] = (50, 75, 90, 95, 99)) -> dict:
    """Per-window pass rates over a threshold grid, score percentiles and a degeneracy warning."""
    mats = [np.asarray(m, np.float32) for m in score_matrices]
    full = np.concatenate([m for m in mats if m.size]) if any(m.size for m in mats) else np.zeros((0, 0), np.float32)
    report: dict = {"windows": int(len(full)), "gestures": int(full.shape[1]) if full.ndim == 2 else 0, "threshold": float(threshold)}
    if not full.size:
        report["warning"] = "no mining windows: clips are shorter than the longest gesture"
        return report
    pct = [float(p) for p in percentiles]
    report["score_percentiles"] = {f"p{p:g}": float(np.percentile(full, p)) for p in pct}
    report["best_score_percentiles"] = {f"p{p:g}": float(np.percentile(full.max(1), p)) for p in pct}
    grid = sorted({round(float(t), 4) for t in (grid if grid is not None else np.arange(0.80, 0.995, 0.01))} | {round(float(threshold), 4)})
    rows = []
    for t in grid:
        fraction = (full >= t).mean(1)
        rows.append({"threshold": t, "window_pass_rate": float((fraction > 0).mean()),
                     "mean_bank_pass_fraction": float(fraction.mean()), "median_bank_pass_fraction": float(np.median(fraction))})
    report["grid"] = rows
    used = (full >= threshold).mean(1)
    report["at_threshold"] = {"window_pass_rate": float((used > 0).mean()), "median_bank_pass_fraction": float(np.median(used)),
                              "mean_passing_gestures": float((full >= threshold).sum(1).mean())}
    report["percentile_thresholds"] = {f"p{p:g}": float(np.percentile(full, p)) for p in pct}
    if sources is not None:
        report["clips"] = [{"source": s, "windows": int(len(m)), "window_pass_rate": float(((m >= threshold).any(1)).mean()) if len(m) else 0.0}
                           for s, m in zip(sources, mats)]
    if np.median(used) > 0.5:
        report["warning"] = ("degenerate threshold: more than half of the bank passes the median window, so random selection makes rules arbitrary; "
                             "raise --threshold or use --threshold-percentile")
    elif (used > 0).mean() == 0:
        report["warning"] = "no window passes this threshold; lower --threshold or use --threshold-percentile"
    return report


def mine_clips(clips: Iterable[Clip], bank: Mapping[str, np.ndarray], threshold: float | None = .92, seed: int = 0, *, phrase_words: int = 5,
               neck_joint: int = 1, threshold_percentile: float | None = None) -> tuple[list[Rule], dict]:
    """Algorithm 1 outer loop: mine every clip against one bank and return rules plus a calibration report."""
    clips = list(clips)
    if not clips:
        raise ValueError("no video clips to mine")
    gb = GestureBank(bank, neck_joint)
    scored = []
    for clip in clips:
        _check_words(clip.words, len(clip.pose))
        scored.append(window_scores(clip.pose, gb, neck_joint))
    if threshold_percentile is not None:
        threshold = threshold_from_percentile([m for _, m in scored], threshold_percentile)
    if threshold is None:
        threshold = DEFAULTS["threshold"]
    rng = np.random.default_rng(seed)
    rules: list[Rule] = []
    counts = []
    for clip, sc in zip(clips, scored):
        mined = mine_rules(clip.pose, clip.words, gb, threshold, source=clip.source, phrase_words=phrase_words, neck_joint=neck_joint, rng=rng, scores=sc)
        rules.extend(mined)
        counts.append(len(mined))
    report = calibration_report([m for _, m in scored], threshold, [c.source for c in clips])
    if threshold_percentile is not None:
        report["threshold_percentile"] = float(threshold_percentile)
    report["rules"] = len(rules)
    for row, count in zip(report.get("clips", []), counts):
        row["rules"] = count
    return rules, report


# --------------------------------------------------------------------------- retrieval

TOKEN = re.compile(r"[\w']+")
SBERT_NAME = "all-MiniLM-L6-v2"
SBERT_ENV = ("BEAT_SBERT_MODEL", "SBERT_MODEL")


def find_sentence_model(explicit: str | Path | None = None, roots: Iterable[str | Path] = (".",)) -> str | None:
    """Sentence-BERT model: ``explicit``, ``BEAT_SBERT_MODEL``, ``SBERT_MODEL``, then ``<root>/models/all-MiniLM-L6-v2``.

    The same lookup order as the repository's demo scripts. Returns None when nothing is configured; a
    model *name* (not a folder) is returned as given and sentence-transformers resolves it.
    """
    if explicit:
        return str(explicit)
    for name in SBERT_ENV:
        if os.environ.get(name):
            return os.environ[name]
    for root in roots:
        folder = Path(root) / "models" / SBERT_NAME
        if folder.is_dir():
            return str(folder.resolve())
    return None


class SentenceEncoder:
    """Sentence-BERT phrase vectors for Algorithm 2: the all-MiniLM-L6-v2 substitution for summed GloVe.

    Each five-word chunk and each mined rule phrase is encoded as a whole (L2-normalised), and the
    chunk takes the rule with the highest cosine, as in Algorithm 2. A text gets a vector only when one
    of its tokens is a whole word of the model's tokenizer vocabulary (``stopwords`` excluded), so a
    chunk of gibberish or an unsupported script stays out of vocabulary and idles, as a chunk without
    GloVe words does. ``model`` is a loaded sentence-transformers model or a folder/name to load.
    """

    kind = "sentence-bert"

    def __init__(self, model, name: str | None = None, stopwords: Iterable[str] = (), device: str = "cpu", cache_size: int = 50000):
        if isinstance(model, (str, Path)):
            from sentence_transformers import SentenceTransformer
            name = name or Path(str(model)).name
            model = SentenceTransformer(str(model), device=device)
        self.model = model
        self.name = name or "sentence-transformer"
        self.stopwords = frozenset(w.lower() for w in stopwords)
        self.cache_size = int(cache_size)
        self._vocab: frozenset | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def label(self) -> str:
        return f"sentence-bert ({self.name})"

    def vocabulary(self) -> frozenset:
        if self._vocab is None:
            tokenizer = getattr(self.model, "tokenizer", None)
            getter = getattr(tokenizer, "get_vocab", None)
            vocab = getter() if callable(getter) else getattr(tokenizer, "vocab", None)
            self._vocab = frozenset(vocab or ())
        return self._vocab

    def in_vocabulary(self, text: str) -> bool:
        """True when a token (or a part around an apostrophe) is a whole vocabulary word, not a stopword."""
        vocab = self.vocabulary()
        for token in TOKEN.findall(str(text).lower()):
            parts = [p for p in token.split("'") if len(p) > 1 or p.isdigit()] or [token]
            if any((p in vocab if vocab else True) and p not in self.stopwords for p in parts):
                return True
        return False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        with self._lock:  # one forward pass at a time; demo servers share the encoder between threads
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
            if missing and len(self._cache) + len(missing) > self.cache_size:
                self._cache.clear()
                missing = list(dict.fromkeys(texts))
            if missing:
                rows = np.asarray(self.model.encode(missing, normalize_embeddings=True), np.float32).reshape(len(missing), -1)
                self._cache.update(zip(missing, rows))
            if not texts:
                return np.zeros((0, 0), np.float32)
            return np.stack([self._cache[t] for t in texts])

    def phrase_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Encoded texts; texts without vocabulary get a zero row (they never match)."""
        texts = list(texts)
        matrix = self.encode(texts)
        known = np.array([self.in_vocabulary(t) for t in texts], bool)
        matrix[~known] = 0.0
        return matrix


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
    """Paper chunking: fixed ``size``-word chunks, a remainder shorter than ``size`` is dropped; text under ``size`` words is one slot."""
    if size < 1:
        raise ValueError("chunk size must be positive")
    tokens = TOKEN.findall(text)
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1, size)]


@dataclass(frozen=True)
class ManualRule:
    """NVBG-style manual rule: any of several keyword phrases triggers one of several gestures."""
    keyword: str
    patterns: tuple
    gestures: tuple
    priority: int = 0


MANUAL_FORMAT = "attg-manual-map/1"


def _manual_rule(keyword: str, patterns: Iterable[str], gestures: Iterable[str], priority=0) -> ManualRule:
    pats = tuple(" ".join(TOKEN.findall(str(p).lower())) for p in patterns)
    pats = tuple(p for p in pats if p)
    gids = tuple(str(g).strip() for g in gestures if str(g).strip())
    if not pats or not gids:
        raise ValueError(f"manual rule {keyword!r} needs at least one pattern and one gesture")
    return ManualRule(str(keyword), pats, gids, int(priority))


def load_manual_map(path: str | Path) -> list[ManualRule]:
    """Read the JSON manual map: {"format": "attg-manual-map/1", "rules": [{keyword, patterns, gestures, priority}]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("manual map needs a non-empty 'rules' list")
    return [_manual_rule(r.get("keyword", (r.get("patterns") or [""])[0]), r.get("patterns", []), r.get("gestures", []), r.get("priority", 0)) for r in rows]


def write_manual_map(path: str | Path, rules: Iterable[ManualRule]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"format": MANUAL_FORMAT, "rules": [{"keyword": r.keyword, "patterns": list(r.patterns), "gestures": list(r.gestures), "priority": r.priority} for r in rules]}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def import_manual_map(path: str | Path) -> list[ManualRule]:
    """Import an NVBG-like XML rule file, a CSV table or the JSON format.

    XML: ``<rule keyword="negation" priority="5"><pattern>no</pattern><animation>shake</animation></rule>``
    (``<animation name="..."/>`` and ``<gesture>`` are also accepted).
    CSV: columns ``keyword,patterns,gestures,priority`` with ``|`` separating several patterns or gestures.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_manual_map(path)
    rules = []
    if suffix == ".xml":
        root = ET.parse(path).getroot()
        for node in root.iter("rule"):
            patterns = [(p.text or "").strip() for p in node.iter("pattern")]
            gestures = [(g.get("name") or g.text or "").strip() for tag in ("animation", "gesture") for g in node.iter(tag)]
            keyword = node.get("keyword") or (patterns[0] if patterns else "")
            rules.append(_manual_rule(keyword, patterns, gestures, node.get("priority", 0) or 0))
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rules.append(_manual_rule(row["keyword"], (row.get("patterns") or row["keyword"]).split("|"), row["gestures"].split("|"), row.get("priority") or 0))
    else:
        raise ValueError("manual map import supports .xml, .csv and .json")
    if not rules:
        raise ValueError("no manual rules found")
    return rules


def match_manual(chunk: str, manual: Sequence[ManualRule]) -> tuple[ManualRule, str] | None:
    """Keyword-containment match: a pattern's words must appear contiguously in the chunk.

    Ties resolve by priority, then longer pattern, then earlier position, then map order.
    """
    tokens = TOKEN.findall(chunk.lower())
    best, best_key = None, None
    for order, rule in enumerate(manual):
        for pattern in rule.patterns:
            ptoks = pattern.split()
            for pos in range(len(tokens) - len(ptoks) + 1):
                if tokens[pos : pos + len(ptoks)] == ptoks:
                    key = (-rule.priority, -len(ptoks), pos, order)
                    if best_key is None or key < best_key:
                        best, best_key = (rule, pattern), key
                    break
    return best


def retrieve(text: str, rules: list[Rule], vectors: Mapping[str, np.ndarray], audio_seconds: float | None = None, manual_first: bool = True, *,
             mode: str | None = None, manual: Sequence[ManualRule] | None = None, chunk_words: int = 5, seed: int = 0,
             oov: str = "idle", idle_id: str | None = "idle", min_similarity: float | None = None) -> list[dict]:
    """Algorithm 2 with the paper's three maps.

    ``vectors`` is either a word-vector mapping (the paper's GloVe; a phrase vector is the sum of its
    word vectors) or a phrase encoder such as ``SentenceEncoder`` (all-MiniLM-L6-v2 phrase vectors, the
    default substitution). Chunking, the argmax cosine and the audio-divided timing are the same.
    ``auto``: cosine against mined rule phrases. ``manual``: keyword containment only;
    unmatched chunks go idle. ``hybrid``: manual match first (higher priority), vectors otherwise.
    Rules whose ``source`` is ``manual`` are treated as single-pattern manual entries.
    A chunk with no vocabulary overlap (or below ``min_similarity``) goes idle, is skipped, or raises (``oov``).
    """
    if audio_seconds is not None and audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    if oov not in ("idle", "skip", "error"):
        raise ValueError("oov must be idle, skip or error")
    manual_rules = list(manual or [])
    if manual_first:
        manual_rules += [_manual_rule(r.phrase, [r.phrase], [r.gesture_id]) for r in rules if r.source == "manual"]
    if mode is None:
        mode = "hybrid" if manual_rules else "auto"
    if mode not in ("manual", "auto", "hybrid"):
        raise ValueError("map mode must be manual, auto or hybrid")
    if mode in ("manual", "hybrid") and not manual_rules:
        raise ValueError(f"{mode} map needs manual rules")
    if mode in ("auto", "hybrid") and not rules:
        raise ValueError("rule map is empty")
    rng = np.random.default_rng(seed)
    segments = chunks(text, chunk_words)
    slot = audio_seconds / len(segments) if audio_seconds is not None and segments else None
    matrix = None
    encoder = None if isinstance(vectors, Mapping) else vectors
    vocabulary_name = "GloVe" if encoder is None else "Sentence-BERT"
    if mode != "manual":
        if encoder is not None:
            matrix = encoder.phrase_matrix([r.phrase for r in rules])
        else:
            dim = len(next(iter(vectors.values()))) if vectors else 300
            matrix = np.stack([phrase_vector(r.phrase, vectors, dim) for r in rules])
        rule_norms = np.linalg.norm(matrix, axis=1)
    result = []
    for index, segment in enumerate(segments):
        entry = {"text": segment, "start_seconds": index * slot if slot else None, "duration_seconds": slot}
        hit = match_manual(segment, manual_rules) if mode != "auto" else None
        if hit is not None:
            rule, pattern = hit
            entry.update(gesture_id=rule.gestures[int(rng.integers(len(rule.gestures)))], similarity=None, map="manual", keyword=rule.keyword, pattern=pattern)
            result.append(entry)
            continue
        reason = "no manual keyword"
        if matrix is not None:
            if encoder is not None:  # computed once per chunk
                query = encoder.phrase_matrix([segment])[0]
            else:
                query = phrase_vector(segment, vectors, matrix.shape[1])
            qn = float(np.linalg.norm(query))
            scores = np.divide(matrix @ query, rule_norms * qn, out=np.full(len(rules), -1.0, np.float32), where=rule_norms * qn > 0)
            best = int(np.argmax(scores))
            if qn > 0 and scores[best] > -1 and (min_similarity is None or scores[best] >= min_similarity):
                entry.update(gesture_id=rules[best].gesture_id, similarity=float(scores[best]), map="auto", rule_phrase=rules[best].phrase)
                result.append(entry)
                continue
            reason = f"no {vocabulary_name} vocabulary overlap" if qn == 0 or scores[best] <= -1 else f"best similarity {scores[best]:.3f} below {min_similarity}"
        if oov == "error":
            raise ValueError(f"{reason} for text chunk: {segment}")
        if oov == "idle":
            entry.update(gesture_id=idle_id, similarity=None, map="idle", reason=reason)
            result.append(entry)
    return result


def load_glove(path: str | Path, required_words: set[str] | None = None) -> dict[str, np.ndarray]:
    vectors = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            word, *values = line.rstrip().split(" ")
            if required_words is None or word in required_words:
                vectors[word] = np.asarray(values, dtype=np.float32)
    dims = {len(v) for v in vectors.values()}
    if len(dims) > 1:
        raise ValueError("inconsistent embedding dimensions")
    return vectors


def write_rules(path: str | Path, rules: Iterable[Rule]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rule in rules:
            fh.write(json.dumps(asdict(rule)) + "\n")


def read_rules(path: str | Path) -> list[Rule]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Rule(**json.loads(line)) for line in fh if line.strip()]
