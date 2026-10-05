"""Local BEAT co-speech demo modes as thin adapters over the vendored paper packages.

Each mode prepares a small local index from an ignored BEAT bank and answers
text queries through the paper package's own algorithms:

* ``automatic``  -> ``automatic_text_to_gesture.core``: padding-aware
  ``GestureBank``, ``mine_clips`` (Algorithm 1, percentile-calibrated
  threshold, seeded random pick among passing gestures) and ``retrieve``
  (Algorithm 2, hybrid manual/auto map, idle on OOV).
* ``wild``       -> ``wild_pose_matching``: Algorithm 3 ``extract_units``,
  ``training.train_gestureclr`` (GestureCLR, paper augmentation, demo step
  budget), ``cluster_latents``, ``build_rules`` and six-gram ``retrieve`` with
  seeded random in-cluster sampling.
* ``multilingual`` -> ``multilingual_gesture``: Algorithm 1
  ``extract_unit_spans``, the package's ``train`` command (GestureCLR with
  per-sample augmentation), ``bisect`` and ``multilingual_retrieve`` with a
  ``Translator`` (dictionary by default; HTTP or local MT when configured).
* ``ridge``      -> ``ridge_gesture``: strong rules aligned to phrase-timed
  spans (``align_phrase``), two-stage ``RidgeModel`` training through the
  package's ``train`` command on association windows only (the bank is held
  out), ``hybrid_retrieve`` and ``GCA``.

Text matching uses Sentence-BERT whenever a local model folder is configured
(``sbert=``, ``BEAT_SBERT_MODEL``, ``SBERT_MODEL`` or ``<repo>/models``);
otherwise a TF-IDF fallback is used and named in every response's
``text_encoder`` field (model names only, never local paths). Prepared motion,
fitted weights and imported annotations belong in ignored output directories.
This small-data demonstration does not reproduce paper benchmarks.

Idle slots (route ``idle_no_match``) follow each paper: Automatic idles only
chunks without vocabulary overlap (Algorithm 2 has no similarity floor); Wild
and Multilingual play the best rule and use ``min_similarity`` only as the
papers' optional low-similarity fallback; untranslated multilingual input idles
with a note instead of raising.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

MODES = {"automatic", "wild", "multilingual", "ridge"}
WORDS = re.compile(r"[\w']+", re.UNICODE)
CODE_VERSION = "2026-10-06.f1"
IDLE_ID = "idle"
MIN_SIMILARITY = 0.2           # RIDGE index default (its rule threshold and fallback are separate)
SBERT_MIN_SIMILARITY = 0.35    # legacy value, kept for callers that pass it explicitly
# Wild/Multilingual pick the best rule for every chunk (paper); the idle threshold is only the papers' optional
# low-similarity fallback. TF-IDF: idle only when no rule shares a content word (cosine 0). Sentence-BERT: idle
# below a low cosine that unrelated or non-English text stays under while ordinary conversational lines pass.
POSE_TFIDF_FLOOR = 1e-6
POSE_SBERT_FLOOR = 0.15  # MiniLM: default application lines score 0.16-0.43 against BEAT rules, gibberish 0.10-0.12
AUTOMATIC_PHRASE_WORDS = 5
AUTOMATIC_PERCENTILE = 80.0
POSE_CHUNK_WORDS = 6
FEATURE_FPS = 15
TRAIN_YAW = 30.0
POSE_DEMO_STEPS = 100          # GestureCLR optimiser steps (batch 64) for the demo preset: about 30-50 s on CPU
MAX_TRAIN_WINDOWS = 384
RIDGE_SPAN_MIN_FRAMES = 20
RIDGE_MIN_VALIDATION_PAIRS = 100
RIDGE_DEMO_STEPS = 60
WILD_CAMERA = {"yaw": 20.0, "pitch": 5.0, "noise": 0.03, "dropout": 0.05, "jitter": 1}
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have he her
him his how i i'm if in into is it it's its just like me my no not of oh ok okay on or our out she so some
than that that's the their them then there these they this those to too uh um up us very was we well were
what when where which who why will with would yeah yes you your""".split())
DEPENDENCIES = {"automatic": ["automatic_text_to_gesture"], "wild": ["wild_pose_matching"],
                "multilingual": ["multilingual_gesture"], "ridge": ["ridge_gesture"]}
FINGER_TERMS = ("thumb", "index", "middle", "ring", "little", "pinky")
UPPER_BODY = ("Hips", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand")
JOINT_ALIASES = {"leftupperarm": "leftarm", "leftlowerarm": "leftforearm",
                 "rightupperarm": "rightarm", "rightlowerarm": "rightforearm"}


# ----------------------------------------------------------------------------
# Bank I/O

_JSON_CACHE: dict = {}


def _read_json(path):
    """Parse a JSON file once per (path, mtime, size); callers must treat the result as read-only."""
    path = Path(path)
    stat = path.stat()
    key = str(path.resolve())
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _JSON_CACHE.get(key)
    if cached is None or cached[0] != stamp:
        raw = path.read_bytes()
        if len(_JSON_CACHE) > 8:
            _JSON_CACHE.clear()
        cached = (stamp, json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
        _JSON_CACHE[key] = cached
    return cached[1], cached[2]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_segments(names, items, need_text=True):
    for item in items:
        p = np.asarray(item["positions"], np.float32)
        if p.ndim != 3 or p.shape[1:] != (len(names), 3) or not len(p) or not np.isfinite(p).all():
            raise ValueError("Every BEAT segment needs finite positions [frames,joints,3]")
        if need_text and not str(item.get("text", "")).strip():
            raise ValueError("Every BEAT segment needs aligned transcript text")


def _check_bank(bank):
    names = bank.get("joint_names")
    clips = bank.get("clips", [])
    if not names or not clips or int(bank.get("fps", 0)) <= 0:
        raise ValueError("BEAT bank requires fps, joint_names, and clips")
    if not bank.get("associations"):
        raise ValueError("BEAT bank requires a disjoint association pool")
    ids = [str(c["id"]) for c in clips]
    if len(set(ids)) != len(ids) or IDLE_ID in ids:
        raise ValueError("BEAT gesture IDs must be unique and must not be 'idle'")
    _check_segments(names, clips + bank["associations"])
    base = bank.get("base_ids") or ids[: int(bank.get("seed_count", 3))]
    if len(base) != 3 or len(set(base)) != 3 or not set(base) <= set(ids):
        raise ValueError("BEAT bank needs three distinct base_ids present in clips")
    return ids, base


def _check_playback(bank):
    """Derived playback banks (units, phrase spans) need unique ids and finite frames, not transcripts."""
    names, clips = bank.get("joint_names"), bank.get("clips", [])
    if not names or not clips or int(bank.get("fps", 0)) <= 0:
        raise ValueError("Playback bank requires fps, joint_names, and clips")
    ids = [str(c["id"]) for c in clips]
    if len(set(ids)) != len(ids) or IDLE_ID in ids:
        raise ValueError("Playback IDs must be unique and must not be 'idle'")
    _check_segments(names, clips, need_text=False)
    return ids


def load_streams(bank, bank_path):
    """Continuous library/train takes saved next to the bank by build_library.save (empty when absent)."""
    meta = bank.get("streams")
    if not meta:
        return []
    path = Path(bank_path).parent / meta["file"]
    if not path.is_file():
        raise ValueError(f"BEAT bank streams file is missing: {path.name}; rebuild the bank")
    data = np.load(path, allow_pickle=False)
    digest = hashlib.sha256()
    out = []
    for take in meta["takes"]:
        positions = np.asarray(data[take["key"]], np.float32)
        digest.update(positions.tobytes())
        out.append(dict(take, positions=positions))
    if meta.get("content_sha256") and digest.hexdigest() != meta["content_sha256"]:
        raise ValueError("BEAT bank streams changed after the bank was written; rebuild the bank")
    return out


def joint_weights(joint_names):
    """Per-joint emphasis: fingers are tested before the broader hand/arm terms."""
    weights = []
    for name in joint_names:
        low = name.casefold()
        if any(term in low for term in FINGER_TERMS):
            weights.append(.7)
        elif any(term in low for term in ("leg", "foot", "toe")):
            weights.append(.2)
        elif any(term in low for term in ("shoulder", "arm", "hand")):
            weights.append(2.)
        else:
            weights.append(.8)
    return np.asarray(weights, np.float32)


def masked_frame_cosine(a, b):
    """Padding-aware mean frame cosine: pad to a common length, average valid frames only."""
    a = np.asarray(a, np.float32).reshape(len(a), -1)
    b = np.asarray(b, np.float32).reshape(len(b), -1)
    length = max(len(a), len(b))
    pa = np.zeros((length, a.shape[1]), np.float32); pa[:len(a)] = a
    pb = np.zeros((length, b.shape[1]), np.float32); pb[:len(b)] = b
    valid = (np.arange(length) < len(a)) & (np.arange(length) < len(b))
    norms = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1)
    valid &= norms > 1e-8
    if not valid.any():
        return 0.0
    return float((np.sum(pa * pb, axis=1)[valid] / norms[valid]).mean())


def _rest_frames(bank, count):
    """A still neutral pose (median first frame of the seed clips) for idle slots."""
    base = set(bank.get("base_ids") or [])
    firsts = [np.asarray(c["positions"][0], np.float32) for c in bank["clips"] if str(c["id"]) in base] or \
             [np.asarray(bank["clips"][0]["positions"][0], np.float32)]
    rest = np.round(np.median(np.stack(firsts), axis=0), 5).tolist()
    return [rest] * int(count)


def _dependency_files(mode):
    files = []
    for package in DEPENDENCIES[mode]:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        for location in (spec.submodule_search_locations or []) if spec else []:
            files.extend(sorted(Path(location).glob("*.py")))
    return files


SBERT_NAME = "all-MiniLM-L6-v2"
SBERT_ENV = ("BEAT_SBERT_MODEL", "SBERT_MODEL")       # the paper-method scripts read SBERT_MODEL
GLOVE_ENV = ("BEAT_GLOVE_PATH", "GLOVE_PATH")
GLOVE_NAMES = ("glove.6B.300d.txt", "glove.840B.300d.txt", "glove.42B.300d.txt")


def _models_dir():
    """``<repo>/models`` for a vendored copy in ``<repo>/scripts`` (the paper-method scripts' default location)."""
    return Path(__file__).resolve().parent.parent / "models"


def sbert_setting(explicit=None):
    """Local Sentence-BERT folder: the argument, ``BEAT_SBERT_MODEL``, ``SBERT_MODEL`` or ``<repo>/models/<name>``.

    None means the labelled TF-IDF fallback. Nothing is downloaded.
    """
    if explicit:
        return str(explicit)
    for name in SBERT_ENV:
        if os.environ.get(name):
            return os.environ[name]
    local = _models_dir() / SBERT_NAME
    return str(local) if local.is_dir() else None


def glove_setting():
    """Local GloVe text file: ``BEAT_GLOVE_PATH``, ``GLOVE_PATH`` or a standard file name in ``<repo>/models``."""
    for name in GLOVE_ENV:
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return value
    return next((str(_models_dir() / n) for n in GLOVE_NAMES if (_models_dir() / n).is_file()), None)


def model_name(path):
    """Report a local model by its folder or file name, never by its absolute path."""
    return Path(str(path)).name or str(path)


def cache_key(mode, bank_path, *, epochs=60, seed=7, strong_rules_path=None, sbert=None):
    """Hash of the bank, settings, this adapter and the paper package sources it calls."""
    digest = hashlib.sha256()
    digest.update(CODE_VERSION.encode())
    digest.update(Path(__file__).read_bytes())
    for path in _dependency_files(mode):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    digest.update(_sha256(bank_path).encode())
    rules = _sha256(strong_rules_path) if strong_rules_path else None
    digest.update(json.dumps({"mode": mode, "epochs": int(epochs), "seed": int(seed), "rules": rules,
                              "sbert": sbert_setting(sbert),
                              "glove": glove_setting() if mode == "automatic" else None}, sort_keys=True).encode())
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# Pose features shared by the pose modes

def _canonical(name):
    low = str(name).casefold()
    return JOINT_ALIASES.get(low, low)


def _feature_index(names):
    """Indices of the 11-joint upper-body contract (paper input); all joints when names are unknown."""
    canon = [_canonical(n) for n in names]
    index = [canon.index(j.casefold()) for j in UPPER_BODY if j.casefold() in canon]
    return index if len(index) >= 6 else list(range(len(names)))


def _sub_index(index, names, joint):
    canon = [_canonical(names[i]) for i in index]
    return canon.index(joint.casefold()) if joint.casefold() in canon else None


def _step(fps):
    return max(1, int(round(float(fps) / FEATURE_FPS)))


def _motion15(positions, fps, index, neck):
    """Neck-centred upper-body motion at the paper's 15 fps, [F, J, 3] metres."""
    p = np.asarray(positions, np.float32)[::_step(fps)][:, index]
    return p - p[:, neck:neck + 1]


def _rotate(p, yaw=0.0, pitch=0.0):
    y, x = math.radians(yaw), math.radians(pitch)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    return (np.asarray(p) @ (rx @ ry).T).astype(np.float32)


def _project(p, yaw=0.0, pitch=0.0):
    """Orthographic camera on +Z looking at the subject, orbited by yaw/pitch: keep X and Y."""
    return _rotate(p, yaw, pitch)[..., :2] if (yaw or pitch) else np.asarray(p, np.float32)[..., :2]


def _corrupt(view, rng, noise=.03, dropout=.05, jitter=1):
    """OpenPose-like corruption of [F,J,2]: temporal jitter, noise relative to body extent, zeroed joints."""
    view = np.asarray(view, np.float32)
    frames = len(view)
    if jitter:
        view = view[(np.arange(frames) + rng.integers(-jitter, jitter + 1, frames)).clip(0, frames - 1)]
    extent = float(np.sqrt(np.mean(view ** 2))) or 1.0
    view = view + rng.normal(0, noise * extent, view.shape).astype(np.float32)
    view[rng.random(view.shape[:2]) < dropout] = 0
    return view


def _pad(seqs, frames=None):
    """Stack variable-length [F,J,c] sequences, zero-padded at the end, with a real-frame mask."""
    frames = frames or max(len(s) for s in seqs)
    out = np.zeros((len(seqs), frames) + seqs[0].shape[1:], np.float32)
    mask = np.zeros((len(seqs), frames), bool)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s[:frames]
        mask[i, :min(len(s), frames)] = True
    return out, mask


def _timed_words(item, frames=None, key="word"):
    """Ordered, non-overlapping, positive-length word intervals inside the segment."""
    frames = len(item["positions"]) if frames is None else frames
    out, last = [], 0
    for w in item.get("words") or []:
        text = str(w.get("text", w.get("word", ""))).strip()
        start, end = max(int(w["start_frame"]), last), min(int(w["end_frame"]), frames)
        if text and end > start:
            out.append({key: text, "start_frame": start, "end_frame": end})
            last = end
    return out


# ----------------------------------------------------------------------------
# Text encoders: local Sentence-BERT when configured, TF-IDF fallback otherwise

def _content(text):
    return {w for w in WORDS.findall(str(text).casefold()) if w not in STOPWORDS}


class TfidfText:
    """TF-IDF bag of words; a text with no in-vocabulary content word encodes to the zero vector."""
    kind = "tfidf"

    def __init__(self, vocab, idf, note=None):
        self.vocab = list(vocab)
        self.idf = np.asarray(idf, np.float32)
        self.lookup = {w: i for i, w in enumerate(self.vocab)}
        self.note = note

    @classmethod
    def fit(cls, texts, note=None):
        tokens = [[w.casefold() for w in WORDS.findall(str(t))] for t in texts]
        vocab = sorted({w for row in tokens for w in row})
        lookup = {w: i for i, w in enumerate(vocab)}
        df = np.zeros(len(vocab), np.float32)
        for row in tokens:
            for w in set(row):
                df[lookup[w]] += 1
        return cls(vocab, np.log((len(tokens) + 1) / (df + 1)) + 1, note)

    @property
    def label(self):
        base = "tfidf-fallback (Sentence-BERT not configured; set BEAT_SBERT_MODEL to a local model folder)"
        return base if not self.note else f"tfidf-fallback ({self.note})"

    def encode(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        matrix = np.zeros((len(texts), max(1, len(self.vocab))), np.float32)
        for i, text in enumerate(texts):
            if not _content(text) & self.lookup.keys():
                continue
            for word in WORDS.findall(str(text).casefold()):
                if word in self.lookup:
                    matrix[i, self.lookup[word]] += 1
        if self.vocab:
            matrix[:, :len(self.vocab)] *= self.idf
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)

    def coverage(self, text):
        content = _content(text)
        known = content & self.lookup.keys()
        return len(known) / max(1, len(content)), known

    vocabulary_coverage = coverage

    def to_dict(self):
        return {"kind": "tfidf", "vocab": self.vocab, "idf": [round(float(v), 6) for v in self.idf],
                "note": self.note}


_SBERT_MODELS: dict = {}
# Lazily loaded models and vector caches are shared by the demo server's request threads.
_MODEL_LOCK = threading.RLock()
_ENCODE_LOCK = threading.Lock()


def _load_sentence_transformer(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cpu")


class SbertText:
    kind = "sbert"

    def __init__(self, path, name=None):
        self.path = str(path)
        self.name = name or model_name(path)

    @property
    def label(self):
        return f"sentence-bert ({self.name})"

    def model(self):
        # Double-checked under a module lock: two first requests on a fresh server must not construct the
        # model concurrently (torch then fails with "Cannot copy out of meta tensor"), and only a fully
        # loaded model is ever cached, so a failed load is retried rather than poisoning the process.
        model = _SBERT_MODELS.get(self.path)
        if model is None:
            with _MODEL_LOCK:
                model = _SBERT_MODELS.get(self.path)
                if model is None:
                    if not Path(self.path).is_dir():
                        raise FileNotFoundError(f"Sentence-BERT folder not found: {model_name(self.path)}")
                    model = _load_sentence_transformer(self.path)
                    _SBERT_MODELS[self.path] = model
        return model

    def encode(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        model = self.model()
        with _ENCODE_LOCK:  # one forward pass at a time: the shared model is not documented as thread-safe
            vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, np.float32).reshape(len(texts), -1)

    def coverage(self, text):
        content = _content(text)
        return (1.0 if content else 0.0), content

    def known(self, word):
        """A word is in vocabulary when it (or a part around an apostrophe) is a whole word-piece of the model."""
        vocab = getattr(self.model().tokenizer, "vocab", None) or {}
        parts = [p for p in str(word).casefold().split("'") if len(p) > 1 or p.isdigit()]
        return any(p in vocab and p not in STOPWORDS for p in parts)

    def vocabulary_coverage(self, text):
        """Share of content words in the model's word-piece vocabulary (true OOV text scores 0)."""
        content = _content(text)
        known = {w for w in content if self.known(w)}
        return len(known) / max(1, len(content)), known

    def to_dict(self):
        # The index names the model only; the local folder is resolved again at query time (no absolute paths).
        return {"kind": "sbert", "model": self.name}


def fit_text_encoder(texts, sbert=None):
    """Sentence-BERT from a local folder when configured and loadable, else TF-IDF fitted on ``texts``."""
    path = sbert_setting(sbert)
    if path:
        encoder = SbertText(path)
        try:
            encoder.encode(["hello"])
            return encoder
        except Exception as exc:  # optional dependency or missing folder: label the fallback honestly
            return TfidfText.fit(texts, note=f"Sentence-BERT {model_name(path)} unavailable: {type(exc).__name__}")
    return TfidfText.fit(texts)


def text_encoder(spec):
    if spec["kind"] == "sbert":
        path = spec.get("path") or sbert_setting()
        if not path:
            raise ValueError(f"Sentence-BERT model {spec.get('model', SBERT_NAME)} used at prepare time is not "
                             "configured now; set BEAT_SBERT_MODEL (or SBERT_MODEL) or rerun prepare_beat_demo.py")
        return SbertText(path, spec.get("model"))
    return TfidfText(spec["vocab"], spec["idf"], spec.get("note"))


def encoder_label(info):
    """Human-readable text-matching route of a prepared index (for the library endpoint)."""
    spec = info.get("text_encoder") or {}
    if spec.get("kind") == "sbert":
        return f"sentence-bert ({spec.get('model') or model_name(spec.get('path', SBERT_NAME))})"
    if spec.get("kind") == "tfidf":
        return text_encoder(spec).label
    return _automatic_vector_label()


# ----------------------------------------------------------------------------
# Automatic Text-to-Gesture (automatic_text_to_gesture.core)

AUTOMATIC_JOINTS = ("Neck", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                    "RightShoulder", "RightArm", "RightForeArm", "RightHand")


def _automatic_inputs(bank):
    """Frontal 2D arm/hand poses, standardised per coordinate, for core's frame cosine.

    The gesture bank is every bank clip. Raw neck-relative poses share a large
    static component (every bank gesture scores ~0.96 against every window), and
    after removing only the dataset mean pose the coordinates with the largest
    spread (raised hands) still dominate the cosine, so Algorithm 1 piles its
    rules onto one or two gestures. Removing the dataset mean pose and dividing
    by the per-coordinate standard deviation keeps the paper's criterion (mean
    frame cosine above a threshold, random pick among passing gestures) and
    spreads the rules over the bank.
    """
    from automatic_text_to_gesture.core import Clip
    names = bank["joint_names"]
    canon = [_canonical(n) for n in names]
    index = [canon.index(j.casefold()) for j in AUTOMATIC_JOINTS if j.casefold() in canon]
    if len(index) < 5:
        index = _feature_index(names)
    neck = _sub_index(index, names, "Neck") or 0
    raw = lambda positions: np.asarray(positions, np.float32)[:, index, :2]  # frontal camera: X, Y
    rel = lambda positions: raw(positions) - raw(positions)[:, neck:neck + 1]
    stacked = np.concatenate([rel(c["positions"]) for c in list(bank["clips"]) + list(bank["associations"])])
    mean, std = stacked.mean(0), stacked.std(0)
    std[neck] = 1.0  # the neck is the origin (all zeros); keep it zero after scaling
    std = np.maximum(std, 1e-3)
    view = lambda positions: (rel(positions) - mean) / std
    gestures = {str(c["id"]): view(c["positions"]) for c in bank["clips"]}
    clips = []
    for item in bank["associations"]:
        words = _timed_words(item)
        if words:
            clips.append(Clip(str(item["id"]), view(item["positions"]), words))
    return gestures, clips, neck


def _automatic_mine(bank, base=None, *, seed, threshold=None, percentile=None):
    """Algorithm 1 over every association window through core.mine_clips, against every bank gesture."""
    from automatic_text_to_gesture.core import mine_clips
    gestures, clips, neck = _automatic_inputs(bank)
    if not clips:
        raise ValueError("Automatic mining needs association windows with aligned words")
    rules, report = mine_clips(clips, gestures, threshold=threshold, seed=int(seed),
                               phrase_words=AUTOMATIC_PHRASE_WORDS, neck_joint=neck,
                               threshold_percentile=percentile)
    by_id = {str(a["id"]): a for a in bank["associations"]}
    rows = [{"text": r.phrase, "gesture_id": r.gesture_id, "score": round(r.score, 5), "route": "mined_pose_rule",
             "source": {"association_id": r.source, "start_frame": r.start_frame, "end_frame": r.end_frame,
                        "association_source": by_id[r.source].get("source", {})}} for r in rules]
    return rows, report


def _automatic_metrics(rules, report):
    usage = Counter(r["gesture_id"] for r in rules)
    at = report.get("at_threshold", {})
    out = {"threshold": round(float(report["threshold"]), 5), "mining_windows": report.get("windows", 0),
           "mined_rules": len(rules), "pair_pass_rate": round(float(at.get("window_pass_rate", 0.0)), 4),
           "mean_passing_gestures": round(float(at.get("mean_passing_gestures", 0.0)), 4),
           "rule_usage": dict(sorted(usage.items())),
           "max_clip_share": round(max(usage.values()) / max(1, len(rules)), 4) if usage else 0.0}
    if report.get("warning"):
        out["calibration_warning"] = report["warning"]
    return out


def _seed_phrases(clip, size=AUTOMATIC_PHRASE_WORDS):
    tokens = [w["word"] for w in _timed_words(clip)] or WORDS.findall(clip["text"])
    return [" ".join(tokens[i:i + size]) for i in range(0, len(tokens), size)]


def _prepare_automatic(bank, base, seed, **_):
    # Calibrate with core.threshold_from_percentile, then mine at that value rounded down to the 0.01 step
    # the UI slider shows (so the default slider position reproduces the prepared rules).
    _, calibration = _automatic_mine(bank, base, seed=seed, percentile=AUTOMATIC_PERCENTILE)
    threshold = math.floor(float(calibration["threshold"]) * 100 - 1e-6) / 100
    rules, report = _automatic_mine(bank, base, seed=seed, threshold=threshold)
    report["threshold_percentile"] = AUTOMATIC_PERCENTILE
    report["percentile_value"] = round(float(calibration["threshold"]), 5)
    seeds = [{"text": phrase, "gesture_id": str(c["id"]), "score": 1.0, "route": "seed_rule",
              "source": c.get("source", {})} for c in bank["clips"] if str(c["id"]) in base for phrase in _seed_phrases(c)]
    return {"rules": seeds + rules, "default_threshold": threshold,
            "calibration": {k: v for k, v in report.items() if k not in {"clips", "grid"}},
            "threshold_rule": f"p{AUTOMATIC_PERCENTILE:g} of window-by-gesture frame cosines rounded down to 0.01 "
                              "(core.threshold_from_percentile; neck-normalised frontal arm/hand XY, "
                              "standardised per coordinate with the dataset mean and spread)",
            "algorithm": "Automatic Text-to-Gesture (automatic_text_to_gesture.core): padding-aware GestureBank over "
                         "every bank clip, mine_clips at stride = gesture length with seeded random pick among "
                         "passing gestures, hybrid manual/auto map retrieval over 5-word chunks (Algorithm 2: best "
                         "rule whenever a chunk shares vocabulary, idle otherwise)",
            "metrics": _automatic_metrics(rules, report), "training_pairs": report.get("windows", 0),
            "playback_ids": [str(c["id"]) for c in bank["clips"]], "text_encoder": {"kind": "bag_of_words"}}


_GLOVE: dict = {}
_WORD_VECTORS: dict = {}
BOW_LABEL = "bag-of-words fallback (summed one-hot IDF vectors; set BEAT_GLOVE_PATH for GloVe)"


def _automatic_vector_label():
    glove, sbert = glove_setting(), sbert_setting()
    if glove:
        return f"glove ({model_name(glove)})"
    if sbert:
        return (f"sentence-bert word vectors ({model_name(sbert)}; each word encoded alone and summed as in "
                "Algorithm 2; set BEAT_GLOVE_PATH for the paper's GloVe)")
    return BOW_LABEL


def _automatic_vectors(rules, text):
    """Word vectors for Algorithm 2's summed-vector phrase match.

    GloVe when a local file is configured (the paper's vectors); otherwise, with a
    local Sentence-BERT, each word is encoded on its own and used as its word
    vector; otherwise one-hot IDF vectors over the rule content words.
    """
    from automatic_text_to_gesture.core import TOKEN, load_glove
    required = {w for r in rules for w in TOKEN.findall(r.phrase.lower())} | set(TOKEN.findall(text.lower()))
    glove = glove_setting()
    if glove:
        with _MODEL_LOCK:
            loaded, absent = _GLOVE.setdefault(glove, ({}, set()))
            missing = required - loaded.keys() - absent
            if missing:
                found = load_glove(glove, missing)
                loaded.update(found); absent.update(missing - found.keys())
            return {w: loaded[w] for w in required if w in loaded}, _automatic_vector_label()
    sbert = sbert_setting()
    if sbert:
        try:
            encoder = SbertText(sbert)
            with _MODEL_LOCK:
                cache = _WORD_VECTORS.setdefault(sbert, {})
                # Only content words in the model's word-piece vocabulary get a vector, so true OOV text
                # (gibberish, untranslated non-Latin script) still idles as in Algorithm 2.
                missing = sorted(w for w in required - cache.keys() if w not in STOPWORDS and encoder.known(w))
                if missing:
                    cache.update(zip(missing, encoder.encode(missing)))
                return {w: cache[w] for w in required if w in cache}, _automatic_vector_label()
        except Exception:  # optional dependency or unreadable folder: fall back to the labelled bag of words
            pass
    encoder = TfidfText.fit([r.phrase for r in rules])
    content = [i for i, w in enumerate(encoder.vocab) if w not in STOPWORDS]
    vectors = {}
    for k, i in enumerate(content):
        v = np.zeros(len(content), np.float32); v[k] = encoder.idf[i]
        vectors[encoder.vocab[i]] = v
    return vectors, BOW_LABEL


def _query_automatic(text, params, info, bank, floor, seed):
    from automatic_text_to_gesture.core import ManualRule, Rule, TOKEN, retrieve
    threshold = _float_param(params, "threshold", info.get("default_threshold", 0.92), -1.0, 1.0)
    base = info["base_ids"]
    rules = info["rules"]
    seeds = [r for r in rules if r["route"] == "seed_rule"]
    if abs(threshold - float(info.get("default_threshold", threshold))) < 5e-4 and seed == info["seed"]:
        mined, metrics = [r for r in rules if r["route"] == "mined_pose_rule"], dict(info.get("metrics", {}))
    else:
        mined, report = _automatic_mine(bank, seed=seed, threshold=threshold)
        metrics = _automatic_metrics(mined, report)
    patterns = {}
    for r in seeds:
        key = " ".join(TOKEN.findall(r["text"].lower()))
        if key:
            patterns.setdefault(key, []).append(r["gesture_id"])
    manual = [ManualRule(k, (k,), tuple(dict.fromkeys(v)), 0) for k, v in patterns.items()]
    objects = [Rule(r["text"], r["gesture_id"], float(r["score"]), int(r["source"].get("start_frame", 0)),
                    int(r["source"].get("end_frame", 0)), str(r["source"].get("association_id", "clip")))
               for r in mined]
    mode = str(params.get("map") or "hybrid").lower()
    if mode not in {"hybrid", "auto", "manual"}:
        raise ValueError("map must be hybrid, auto or manual")
    if not objects:
        mode = "manual"
    if mode != "auto" and not manual:
        mode = "auto"
    vectors, label = _automatic_vectors(objects or [Rule(r["text"], r["gesture_id"], 1.0, 0, 0) for r in seeds], text)
    entries = retrieve(text, objects, vectors, manual_first=False, mode=mode, manual=manual,
                       chunk_words=AUTOMATIC_PHRASE_WORDS, seed=seed, oov="idle", idle_id=IDLE_ID,
                       min_similarity=floor)
    by_phrase = {r["text"]: r for r in mined}
    slots = []
    for e in entries:
        if e["map"] == "idle":
            slots.append((e["text"], None, {"kind": "idle", "reason": e.get("reason"), "floor": floor}, {}, 0.0))
        elif e["map"] == "manual":
            slots.append((e["text"], e["gesture_id"], {"kind": "manual_map", "keyword": e.get("keyword"),
                                                        "pattern": e.get("pattern")}, {"route": "seed_rule"}, 1.0))
        else:
            rule = by_phrase.get(e.get("rule_phrase"), {})
            slots.append((e["text"], e["gesture_id"], rule.get("source", {}),
                          {"route": "mined_pose_rule", "similarity": round(float(e["similarity"]), 5),
                           "pose_score": rule.get("score"), "rule_phrase": e.get("rule_phrase")},
                          float(e["similarity"])))
    extra = {**metrics, "map": mode, "threshold": threshold}
    return slots, extra, label, {"map": mode, "chunks": entries}


# ----------------------------------------------------------------------------
# Units and training data for the pose modes (wild, multilingual)

def _unit_sources(bank, streams):
    """Continuous library takes when the bank stores them, else each bank window."""
    library = [s for s in streams if s.get("role") == "library"]
    if library:
        return "library_streams", [{"id": s["take"], "take": s["take"], "speaker": s.get("speaker"),
                                    "start_frame": int(s.get("start_frame", 0)), "positions": s["positions"],
                                    "words": s.get("words", []), "meta": s} for s in library]
    return "bank_windows", [{"id": str(c["id"]), "take": c.get("source", {}).get("take", str(c["id"])),
                             "speaker": c.get("source", {}).get("speaker"),
                             "start_frame": int(c.get("source", {}).get("start_frame", 0)),
                             "positions": np.asarray(c["positions"], np.float32), "words": c.get("words") or [],
                             "text": c.get("text", ""), "meta": c.get("source", {})} for c in bank["clips"]]


def _make_unit(src, start, end, algorithm):
    positions = np.asarray(src["positions"], np.float32)[start:end]
    words = []
    for w in src.get("words") or []:
        a, b = int(w["start_frame"]), int(w["end_frame"])
        if start <= (a + b) / 2 < end:
            words.append({"text": str(w.get("text", w.get("word", ""))), "start_frame": max(0, a - start),
                          "end_frame": min(end - start, b - start)})
    take_start = src["start_frame"] + start
    identifier = f"{src['take']}:{take_start}-{take_start + len(positions)}"
    meta = src["meta"]
    source = {"dataset": "BEAT", "speaker": src.get("speaker"), "take": src["take"], "window_id": identifier,
              "start_frame": take_start, "end_frame": take_start + len(positions), "parent": src["id"],
              "route": meta.get("route"), "unit_algorithm": algorithm,
              "motion_url": meta.get("motion_url"), "alignment_url": meta.get("alignment_url")}
    text = " ".join(w["text"] for w in words).strip()
    return {"id": identifier, "text": text, "positions": np.round(positions, 4).tolist(), "words": words,
            "source": source, "frames": len(positions)}


def _training_windows(bank, streams, index, neck, fps, seed, kind):
    """3D training motion [F,J,3] at 15 fps: dense 2-3 s windows over train takes, else train associations."""
    train = [s for s in streams if s.get("role") == "train"]
    seqs = []
    if train:
        for i, stream in enumerate(train):
            m = _motion15(stream["positions"], fps, index, neck)
            if kind == "wild":
                from wild_pose_matching.units import dense_windows
                spans = dense_windows(len(m), FEATURE_FPS, 2.0, 3.0, stride=5, seed=seed + i)
            else:  # paper: fixed 45-frame (3 s) sequences
                spans = [(s, s + 45) for s in range(0, len(m) - 45 + 1, 9)]
            seqs.extend(m[a:b] for a, b in spans)
        origin = ("units.dense_windows (2-3 s, stride 5)" if kind == "wild" else "45-frame windows (stride 9)") + \
            f" over {len(train)} train-role take(s)"
    else:
        pool = [a for a in bank["associations"] if a.get("role") == "train"] or \
            bank["associations"][:len(bank["associations"]) // 2]
        seqs = [_motion15(a["positions"], fps, index, neck) for a in pool]
        origin = f"{len(pool)} train-role association windows"
    if len(seqs) > MAX_TRAIN_WINDOWS:
        seqs = [seqs[int(k)] for k in np.linspace(0, len(seqs) - 1, MAX_TRAIN_WINDOWS).round()]
    return seqs, origin


def _mining_pool(bank):
    pool = bank["associations"]
    mine = [a for a in pool if a.get("role") == "wild"]
    return mine or pool[len(pool) // 2:]


def _wild_views(mine, fps, index, neck, seed):
    """Held-out wild 2D views: projected at the wild camera and corrupted (OpenPose-like)."""
    cam = WILD_CAMERA
    views3 = [_motion15(a["positions"], fps, index, neck) for a in mine]
    views2 = [_corrupt(_project(m, cam["yaw"], cam["pitch"]), np.random.default_rng(seed + 1000 + i),
                       cam["noise"], cam["dropout"], cam["jitter"]) for i, m in enumerate(views3)]
    return views3, views2


def _cluster_count(units):
    return max(2, min(len(units) // 2 or 1, int(round(len(units) / 6))))


def _playback_bank(bank, clips, base=None, origin_sha=None, extra=None):
    ids = [c["id"] for c in clips]
    return {"schema": "paperreach.beat-playback-bank.v1", "fps": bank["fps"], "joint_names": bank["joint_names"],
            "axisSigns": bank.get("axisSigns", [1, 1, 1]), "clips": clips,
            "base_ids": base or (ids[:3] if len(ids) >= 3 else ids), "association_count": len(bank["associations"]),
            "origin_bank_sha256": origin_sha, "provenance": bank.get("provenance"), **(extra or {})}


def _pose_rule_metrics(rules, units, clusters):
    usage = Counter(r["gesture_id"] for r in rules)
    return {"learned_rule_usage": dict(sorted(usage.items())),
            "distinct_matched_units": len(usage), "unit_count": len(units),
            "max_unit_share": round(max(usage.values()) / max(1, len(rules)), 4) if usage else 0.0,
            "rule_routes": {"learned_pose_rule": len(rules)},
            "rule_match": "nearest unit by cosine of per-modality mean-centred GestureCLR latents",
            "cluster_sizes": {k: len(v) for k, v in clusters.items()}}


def _centred(latents):
    """Mean-centre unit-norm latents of one modality and renormalise.

    The demo-budget GestureCLR latents are anisotropic (unit latents share a
    mean pairwise cosine near 0.8), so a plain nearest-unit match sends most
    wild sequences to one hub unit. Removing each modality's mean keeps the
    papers' nearest-unit criterion (cosine argmax) and spreads the rules.
    """
    z = np.asarray(latents, np.float32)
    z = z - z.mean(0, keepdims=True)
    return z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-8)


def _threads():
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# ----------------------------------------------------------------------------
# Wild Pose Matching (wild_pose_matching: units, training, pipeline)

def _prepare_wild(bank, bank_path, output, epochs, seed, sbert, origin_sha, **_):
    from wild_pose_matching.pipeline import build_rules, cluster_latents
    from wild_pose_matching.training import config_from, encode, save_checkpoint, top1, train_gestureclr
    from wild_pose_matching.units import Normalization, extract_units, suggest_variance_threshold, window_variances
    _threads()
    names, fps = bank["joint_names"], bank["fps"]
    index = _feature_index(names)
    neck = _sub_index(index, names, "Neck") or 0
    left, right = _sub_index(index, names, "LeftArm"), _sub_index(index, names, "RightArm")
    norm = Normalization(neck, (left, right) if left is not None and right is not None else None)
    streams = load_streams(bank, bank_path)
    route, sources = _unit_sources(bank, streams)
    motions = [_motion15(s["positions"], fps, index, neck) for s in sources]
    lo, hi, step = 2 * FEATURE_FPS, 3 * FEATURE_FPS, _step(fps)
    if route == "library_streams":
        long_enough = [m for m in motions if len(m) >= (lo + hi) // 2]
        threshold = suggest_variance_threshold(window_variances(long_enough, 3, (lo + hi) // 2, norm), "percentile", 25.0)
        rule = "percentile 25 of window variances (units.suggest_variance_threshold)"
    else:
        threshold, rule = 0.0, "single-take bank: variance threshold 0, one closure-minimal unit per bank window"
    units, whole = [], 0
    for src, m in zip(sources, motions):
        spans = extract_units(m, FEATURE_FPS, 2.0, 3.0, threshold, None, 3, norm) if len(m) >= lo else []
        if not spans and route == "bank_windows":
            spans, whole = [(0, len(m))], whole + 1
        units.extend(_make_unit(src, a * step, min(b * step, len(src["positions"])), "wild_pose_matching.units.extract_units")
                     for a, b in spans)
    if len(units) < 3:
        raise ValueError("Algorithm 3 produced fewer than three units; add takes or lower the motion-energy floor")
    # GestureCLR pairs: 3D motion and its 2D projection under a random camera yaw (+-30 deg) per pair.
    seqs, origin = _training_windows(bank, streams, index, neck, fps, seed, "wild")
    rng = np.random.default_rng(seed)
    x3, mask = _pad(seqs)
    x2 = np.stack([_project(x, rng.uniform(-TRAIN_YAW, TRAIN_YAW)) for x in x3])
    x2n = norm.apply(x2.reshape(len(x2), x2.shape[1], -1), 2, mask)
    x3n = norm.apply(x3.reshape(len(x3), x3.shape[1], -1), 3, mask)
    # Demo budget: the requested epochs, capped at POSE_DEMO_STEPS optimiser steps (train_gestureclr's split).
    n_val = max(2, int(round(len(x3) * .1))) if len(x3) >= 6 else 0
    per_epoch = max(1, (len(x3) - n_val) // min(64, max(2, len(x3) - n_val)))
    cfg = config_from("demo", epochs=max(1, int(epochs)), seed=int(seed),
                      max_steps=min(POSE_DEMO_STEPS, max(1, int(epochs)) * per_epoch))
    started = time.perf_counter()
    model, history = train_gestureclr(x2n, x3n, mask, cfg, log=None)
    seconds = time.perf_counter() - started
    save_checkpoint(output / "gestureclr.pt", model, x2n.shape[-1], x3n.shape[-1], cfg, norm, history)
    # Rules: held-out wild 2D views -> nearest 3D unit in the latent space (build_rules).
    unit_motion = [_motion15(u["positions"], fps, index, neck) for u in units]
    u3, umask = _pad(unit_motion)
    uz = encode(model.motion3d, norm.apply(u3.reshape(len(u3), u3.shape[1], -1), 3, umask), umask)
    mine = _mining_pool(bank)
    w3, w2 = _wild_views(mine, fps, index, neck, seed)
    w3p, wmask = _pad(w3)
    w2p, _ = _pad(w2)
    wz = encode(model.pose2d, norm.apply(w2p.reshape(len(w2p), w2p.shape[1], -1), 2, wmask), wmask)
    mz = encode(model.motion3d, norm.apply(w3p.reshape(len(w3p), w3p.shape[1], -1), 3, wmask), wmask)
    uc, wc = _centred(uz), _centred(wz)
    labels, _ = cluster_latents(uc, _cluster_count(units), seed=int(seed))
    texts = [str(a["text"]) for a in mine]
    encoder = fit_text_encoder(texts, sbert)
    rows = build_rules(encoder.encode(texts), texts, wc, uc, [u["id"] for u in units], labels)
    for row, item in zip(rows, mine):
        row.update(route="learned_pose_rule", score=round(row["pose_match"], 5),
                   source={"association_id": item["id"], **item.get("source", {})})
    clusters = {str(i): [u["id"] for u, label in zip(units, labels) if label == i] for i in sorted(set(labels.tolist()))}
    metrics = {"train_pairs": history["train_pairs"], "val_pairs": history["val_pairs"],
               "training_windows": origin, "training_steps": history["epochs"][-1]["step"] if history["epochs"] else 0,
               "training_seconds": round(seconds, 2), "best_epoch": history.get("best_epoch"),
               "train_top1_eval": _round(history.get("train_top1_eval")), "val_top1_best": _round(history.get("val_top1_best")),
               "wild_windows": len(mine), "wild_camera": WILD_CAMERA,
               "heldout_cross_view_top1": round(top1(wz, mz), 4), "heldout_chance": round(1 / len(mine), 4),
               **_pose_rule_metrics(rows, units, clusters)}
    playback = _playback_bank(bank, units, origin_sha=origin_sha, extra={"unit_route": route})
    return {"rules": rows, "clusters": clusters, "training_loss": _round(history.get("best_monitor")),
            "metrics": metrics, "training_pairs": history["train_pairs"], "text_encoder": encoder.to_dict(),
            "units": {"extractor": "wild_pose_matching.units.extract_units (Algorithm 3)", "source": route,
                      "variance_threshold": round(float(threshold), 6), "variance_rule": rule,
                      "count": len(units), "whole_window_fallbacks": whole, "fps": FEATURE_FPS,
                      "normalization": norm.to_dict(), "lengths_frames": Counter(u["frames"] for u in units).most_common(5)},
            "train_config": {"preset": "demo", "epochs": cfg.epochs, "max_steps": cfg.max_steps,
                             "batch_size": cfg.batch_size, "lr": cfg.lr, "noise_variances": list(cfg.noise_variances),
                             "shift_prob": cfg.shift_prob},
            "algorithm": "Wild Pose Matching (wild_pose_matching): Algorithm 3 units, GestureCLR trained with paper "
                         "augmentation, Bisecting K-Means clusters, build_rules from held-out projected wild poses, "
                         "six-gram retrieval with seeded random in-cluster sampling",
            "playback": playback}


def _round(value, digits=4):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), digits)


def _idle_reason(similarity, floor):
    if similarity <= 1e-9:
        return "no rule shares vocabulary with this chunk"
    return f"best rule similarity {similarity:.3f} below the low-similarity fallback {floor:g}"


def _query_wild(text, params, info, floor, seed):
    from wild_pose_matching.pipeline import retrieve
    encoder = text_encoder(info["text_encoder"])
    clusters = {int(k): v for k, v in info["clusters"].items()}
    by_text = {}
    for r in info["rules"]:
        by_text.setdefault(r["text"], r)
    rows = retrieve(text, info["rules"], encoder.encode, clusters, seed, chunk_words=POSE_CHUNK_WORDS,
                    min_similarity=floor, idle_id=IDLE_ID)
    slots = []
    for row in rows:
        detail = {"similarity": round(row["similarity"], 5), "cluster_id": row["cluster_id"]}
        if row["gesture_id"] != IDLE_ID and not encoder.vocabulary_coverage(row["text"])[1]:
            row = dict(row, gesture_id=IDLE_ID, map="idle", reason="no in-vocabulary content word")
        if row["gesture_id"] == IDLE_ID:
            reason = row.get("reason") or _idle_reason(row["similarity"], floor)
            slots.append((row["text"], None, {"kind": "idle", "reason": reason, "floor": floor}, detail, 0.0))
            continue
        rule = by_text.get(row.get("rule_text"), {})
        slots.append((row["text"], row["gesture_id"],
                      {"matched_gesture_id": rule.get("gesture_id"), "cluster_id": row["cluster_id"],
                       "rule_text": row.get("rule_text"), "association_source": rule.get("source", {}),
                       "sampling": "numpy default_rng(seed) choice within the matched cluster"},
                      {**detail, "route": "learned_pose_rule", "pose_score": rule.get("score")}, row["similarity"]))
    return slots, {"seed": seed}, encoder.label, {"chunks": rows}


# ----------------------------------------------------------------------------
# Multilingual Gesture (multilingual_gesture: pipeline, cli train, translate)

def _quiet(call, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        call(*args)
    return out.getvalue()


def _prepare_multilingual(bank, bank_path, output, epochs, seed, sbert, origin_sha, **_):
    from multilingual_gesture import cli
    from multilingual_gesture.pipeline import bisect, extract_unit_spans, normalize_batch, pad_units
    _threads()
    names, fps = bank["joint_names"], bank["fps"]
    index = _feature_index(names)
    neck = _sub_index(index, names, "Neck") or 0
    streams = load_streams(bank, bank_path)
    route, sources = _unit_sources(bank, streams)
    step = _step(fps)
    units, reports = [], []
    for src in sources:
        m = _motion15(src["positions"], fps, index, neck)
        variance = "auto" if route == "library_streams" else 0.0
        spans, report = extract_unit_spans(m.reshape(len(m), -1), FEATURE_FPS, 2.0, 3.0, variance)
        if not spans and route == "bank_windows":
            spans = [(0, len(m))]; report["whole_window_fallback"] = True
        reports.append({"source": src["id"], **{k: v for k, v in report.items() if k != "scale"}})
        units.extend(_make_unit(src, a * step, min(b * step, len(src["positions"])),
                                "multilingual_gesture.pipeline.extract_unit_spans") for a, b in spans)
    if len(units) < 3:
        raise ValueError("Algorithm 1 produced fewer than three units; add takes or lower the motion-energy floor")
    seqs, origin = _training_windows(bank, streams, index, neck, fps, seed, "multilingual")
    rng = np.random.default_rng(seed)
    x3, mask = _pad(seqs)
    lengths = mask.sum(1)
    x2 = np.stack([_project(x, rng.uniform(-TRAIN_YAW, TRAIN_YAW)) for x in x3])
    pairs = output / "train-pairs.npz"
    np.savez(pairs, pose2d=x2.reshape(len(x2), x2.shape[1], -1), motion3d=x3.reshape(len(x3), x3.shape[1], -1),
             lengths=lengths.astype(np.int64), ids=np.asarray([f"train_{i:04d}" for i in range(len(x3))]))
    n_train = len(x3) - (int(round(len(x3) * .1)) if int(round(len(x3) * .1)) >= 2 else 0)
    per_epoch = max(1, math.ceil(n_train / 64))
    run_epochs = max(1, min(int(epochs), math.ceil(POSE_DEMO_STEPS / per_epoch)))
    checkpoint, history_path = output / "gestureclr.pt", output / "train-history.json"
    started = time.perf_counter()
    _quiet(cli.main, ["train", "--pairs", str(pairs), "--output", str(checkpoint), "--preset", "demo",
                      "--epochs", str(run_epochs), "--seed", str(int(seed)), "--log-every", "0",
                      "--history", str(history_path)])
    seconds = time.perf_counter() - started
    pairs.unlink(missing_ok=True)
    model, ck = cli.load_gestureclr(checkpoint)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    unit_motion = [_motion15(u["positions"], fps, index, neck).reshape(-1, len(index) * 3) for u in units]
    u3, ulen = pad_units(unit_motion, max(len(u) for u in unit_motion))
    mine = _mining_pool(bank)
    w3, w2 = _wild_views(mine, fps, index, neck, seed)
    w3p, wmask = _pad(w3)
    w2p, _ = _pad(w2)
    wlen = wmask.sum(1)
    w2f, w3f = w2p.reshape(len(w2p), w2p.shape[1], -1), w3p.reshape(len(w3p), w3p.shape[1], -1)
    if ck.get("normalize", False):
        u3, w2f, w3f = normalize_batch(u3, ulen), normalize_batch(w2f, wlen), normalize_batch(w3f, wlen)
    uz = cli.encode_batches(model.motion3d, u3, ulen)
    wz = cli.encode_batches(model.pose2d, w2f, wlen)
    mz = cli.encode_batches(model.motion3d, w3f, wlen)
    uc, wc = _centred(uz), _centred(wz)
    labels, _ = bisect(uc, _cluster_count(units), int(seed))
    texts = [str(a["text"]) for a in mine]
    encoder = fit_text_encoder(texts, sbert)
    embeddings = encoder.encode(texts)
    sims = wc @ uc.T
    nearest = sims.argmax(1)
    rows = [{"english_text": t, "text": t, "text_embedding": e.tolist(), "cluster_id": int(labels[j]),
             "source_gesture_id": units[j]["id"], "gesture_id": units[j]["id"],
             "pose_similarity": round(float(sims[i, j]), 5), "score": round(float(sims[i, j]), 5),
             "wild_speaker": str(item.get("source", {}).get("speaker", "")), "route": "learned_pose_rule",
             "source": {"association_id": item["id"], **item.get("source", {})}}
            for i, (t, e, j, item) in enumerate(zip(texts, embeddings, nearest, mine))]
    clusters = {str(i): [u["id"] for u, label in zip(units, labels) if label == i] for i in sorted(set(labels.tolist()))}
    best = min(history["history"], key=lambda r: r.get("val_loss", r["loss"]))
    heldout = float(((wz @ mz.T).argmax(1) == np.arange(len(mine))).mean())
    metrics = {"train_pairs": history["config"]["train_pairs"], "val_pairs": history["config"]["val_pairs"],
               "training_windows": origin, "training_epochs": run_epochs, "training_seconds": round(seconds, 2),
               "best_epoch": best["epoch"], "val_top1_best": _round(best.get("val_top1")),
               "augment": history["config"]["augment"], "wild_windows": len(mine), "wild_camera": WILD_CAMERA,
               "heldout_cross_view_top1": round(heldout, 4), "heldout_chance": round(1 / len(mine), 4),
               **_pose_rule_metrics(rows, units, clusters)}
    playback = _playback_bank(bank, units, origin_sha=origin_sha, extra={"unit_route": route})
    return {"rules": rows, "clusters": clusters, "training_loss": _round(best.get("val_loss", best["loss"])),
            "metrics": metrics, "training_pairs": metrics["train_pairs"], "text_encoder": encoder.to_dict(),
            "units": {"extractor": "multilingual_gesture.pipeline.extract_unit_spans (Algorithm 1)", "source": route,
                      "variance_threshold": "auto (elbow)" if route == "library_streams" else 0.0,
                      "count": len(units), "fps": FEATURE_FPS, "takes": reports, "blend_frames": 5,
                      "lengths_frames": Counter(u["frames"] for u in units).most_common(5)},
            "train_config": {"command": "multilingual_gesture.cli train --preset demo", "epochs": run_epochs,
                             "batch_size": 64, "augment": history["config"]["augment"]},
            "algorithm": "Multilingual Gesture (multilingual_gesture): Algorithm 1 units, GestureCLR trained by the "
                         "package with per-sample augmentation, Bisecting K-Means, translate-to-English Translator, "
                         ">30-word sentence split, six-word retrieval with idle threshold and blend frames",
            "playback": playback}


class _Translator:
    """Explicit dictionary first (examples/beat-translations.json plus request overrides), then configured MT."""

    def __init__(self, table, fallback=None, label="dictionary"):
        from multilingual_gesture.translate import DictTranslator
        self.table = {str(k).strip(): v for k, v in (table or {}).items()}
        self.dictionary = DictTranslator(self.table)
        self.fallback, self.label, self.used = fallback, label, set()
        self.missing = {}  # source chunk -> why no English text was available

    def translate(self, text, source_language, target_language="en"):
        key = str(text).strip()
        try:
            out = self.dictionary.translate(key, source_language, target_language)
            self.used.add("dictionary" if key in self.table else "identity")
            return out
        except ValueError as exc:
            reason = str(exc)
        if self.fallback is not None:
            try:
                out = self.fallback.translate(text, source_language, target_language)
                self.used.add(self.label)
                return out
            except Exception as exc:  # unreachable or failing MT service: idle with a note, never an error
                reason = f"{self.label} failed ({type(exc).__name__}: {exc})"
        # No translation: the chunk retrieves nothing and plays an explicit idle slot with this note.
        detail = f" ({reason})" if self.fallback is not None else ""
        self.missing[key] = (f"no {source_language}->{target_language} translation for this text{detail}; add it to "
                             "examples/beat-translations.json, send english_text, or set BEAT_TRANSLATOR=http|local")
        self.used.add("untranslated (idle)")
        return ""


def _translator(table):
    """BEAT_TRANSLATOR=http|local enables the package's HTTP or local MT client behind the dictionary."""
    from multilingual_gesture.translate import make_translator
    kind = (os.environ.get("BEAT_TRANSLATOR") or "dict").lower()
    if kind == "dict":
        return _Translator(table)
    fallback = make_translator(kind, url=os.environ.get("BEAT_TRANSLATOR_URL"),
                               model=os.environ.get("BEAT_TRANSLATOR_MODEL"),
                               api=os.environ.get("BEAT_TRANSLATOR_API", "openai"),
                               api_key_env=os.environ.get("BEAT_TRANSLATOR_API_KEY_ENV", "OPENAI_API_KEY"),
                               model_path=os.environ.get("BEAT_MT_MODEL_PATH"))
    return _Translator(table, fallback, f"{kind} MT")


def _query_multilingual(text, params, info, floor, seed):
    from multilingual_gesture.pipeline import multilingual_retrieve, split_long_text
    encoder = text_encoder(info["text_encoder"])
    clusters = {int(k): v for k, v in info["clusters"].items()}
    language = str(params.get("source_language") or "en")
    table = params.get("translation_map") if isinstance(params.get("translation_map"), dict) else {}
    translator = _translator(table)
    source = text
    if not language.lower().startswith("en") and text.strip() in translator.table and len(split_long_text(text)) > 1:
        # An explicit whole-line translation of long input: split and retrieve on the English side.
        text, language = str(translator.table[text.strip()]), "en"
        translator.used.add("dictionary (whole line)")
    memo = {}

    def encode(texts):
        texts = list(texts)
        todo = [t for t in dict.fromkeys(texts) if t not in memo]
        if todo:
            memo.update(zip(todo, np.asarray(encoder.encode(todo), np.float32)))
        return np.stack([memo[t] for t in texts])

    out = multilingual_retrieve(text, language, translator, info["rules"], encode, clusters, seed,
                                min_similarity=floor, idle_id=IDLE_ID, max_words=30)
    by_text = {}
    for r in info["rules"]:
        by_text.setdefault(r["text"], r)
    matrix = np.asarray([r["text_embedding"] for r in info["rules"]], np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)

    def _best_rule_text(_, g):
        # The package returns only the cluster; report which rule it matched (same argmax as retrieve()).
        q = memo.get(g["english_text"])
        return None if q is None else info["rules"][int((matrix @ q).argmax())]["text"]
    slots = []
    for chunk in out["chunks"]:
        note = translator.missing.get(str(chunk["source_text"]).strip())
        if note is not None and not chunk["gestures"]:
            # Untranslated source text: an explicit idle slot that carries the note (no error, no guess).
            slots.append((chunk["source_text"], None, {"kind": "idle", "reason": note},
                          {"similarity": 0.0, "chunk_index": chunk["index"], "untranslated": True}, 0.0))
            continue
        for g in chunk["gestures"]:
            detail = {"similarity": round(g["similarity"], 5), "cluster_id": g["cluster_id"],
                      "chunk_index": g["chunk_index"], "blend_frames": g["blend_frames"]}
            oov = not g["idle"] and not encoder.vocabulary_coverage(g["english_text"])[1]
            if g["idle"] or oov:
                reason = "no in-vocabulary content word" if oov else _idle_reason(g["similarity"], floor)
                slots.append((g["english_text"], None, {"kind": "idle", "reason": reason, "floor": floor},
                              dict(detail, cluster_id=None), 0.0))
            else:
                rule = by_text.get(_best_rule_text(info, g), {})
                slots.append((g["english_text"], g["gesture_id"],
                              {"cluster_id": g["cluster_id"], "sampling": "numpy default_rng(seed + chunk) choice "
                               "within the matched cluster", "matched_gesture_id": rule.get("gesture_id"),
                               "rule_text": rule.get("text")},
                              {**detail, "route": "learned_pose_rule"}, g["similarity"]))
    trace = {"retrieval_text": out["english_text"], "tts_text": out["tts_text"],
             "chunks": [{k: c[k] for k in ("index", "source_text", "english_text", "tts_text")} for c in out["chunks"]],
             "translation": ", ".join(sorted(translator.used)) or None, "source_text": source}
    if translator.missing:
        trace["translation_note"] = next(iter(translator.missing.values()))
    return slots, {"seed": seed, "sentence_chunks": len(out["chunks"])}, encoder.label, trace


# ----------------------------------------------------------------------------
# RIDGE (ridge_gesture: pipeline, annotate, model, cli train)

def _strong_rules(path, clip_ids):
    if not path:
        return []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = raw.get("rules", raw) if isinstance(raw, dict) else raw
    result = []
    for rule in rows:
        if rule.get("gesture_id") not in clip_ids or not str(rule.get("phrase", "")).strip():
            raise ValueError("Strong rules need a valid phrase and bank gesture_id")
        provenance = rule.get("provenance", {})
        if provenance.get("kind") not in {"llm_json", "manual_annotation"}:
            raise ValueError("Strong rules require explicit llm_json or manual_annotation provenance")
        result.append({"phrase": rule["phrase"], "gesture_id": rule["gesture_id"], "provenance": provenance})
    return result


def _span_clip(clip, start, end, phrase, route):
    positions = np.asarray(clip["positions"], np.float32)
    if end - start < RIDGE_SPAN_MIN_FRAMES:
        grow = RIDGE_SPAN_MIN_FRAMES - (end - start)
        start = max(0, start - grow // 2)
        end = min(len(positions), start + RIDGE_SPAN_MIN_FRAMES)
        start = max(0, end - RIDGE_SPAN_MIN_FRAMES)
    origin = clip.get("source", {})
    offset = int(origin.get("start_frame", 0))
    identifier = f"{clip['id']}:{start}-{end}"
    words = [{"text": w["word"], "start_frame": max(0, w["start_frame"] - start), "end_frame": w["end_frame"] - start}
             for w in _timed_words(clip) if start <= (w["start_frame"] + w["end_frame"]) / 2 < end]
    source = dict(origin, start_frame=offset + start, end_frame=offset + end, parent=str(clip["id"]),
                  window_id=f"{origin.get('take', clip['id'])}:{offset + start}-{offset + end}",
                  alignment="ridge_gesture.pipeline.align_phrase", rule_route=route)
    return {"id": identifier, "text": phrase, "positions": np.round(positions[start:end], 4).tolist(),
            "words": words, "source": source}


def _ridge_rule_spans(bank, strong):
    """Bind each rule phrase to its timed span inside the bank clip (align_phrase)."""
    from ridge_gesture.annotate import annotate_record
    from ridge_gesture.pipeline import align_phrase
    by_id = {str(c["id"]): c for c in bank["clips"]}
    rows, spans = [], {}
    if strong:
        candidates = [(r["phrase"], by_id[r["gesture_id"]], "strong_rule", r["provenance"]) for r in strong]
    else:  # no cached LLM/manual annotation matches this bank: the package's labelled heuristic
        candidates = []
        for clip in bank["clips"]:
            record = {"record_id": str(clip["id"]), "text": clip["text"], "words": _timed_words(clip)}
            row = annotate_record(record, mode="heuristic", limit=1)
            for phrase in row["phrases"]:
                candidates.append((phrase["phrase"], clip, "heuristic_rule",
                                   {"kind": "heuristic_annotation", "annotator": "ridge_gesture.annotate heuristic",
                                    "note": row["provenance"].get("note")}))
    for phrase, clip, route, provenance in candidates:
        words = _timed_words(clip)
        try:
            start, end = align_phrase(phrase, words)
        except ValueError:
            start, end = 0, len(clip["positions"])
        span = _span_clip(clip, start, end, phrase, route)
        spans.setdefault(span["id"], span)
        rows.append({"phrase": phrase, "gesture_id": span["id"], "clip_id": str(clip["id"]), "route": route,
                     "provenance": provenance, "start_frame": span["source"]["start_frame"],
                     "end_frame": span["source"]["end_frame"]})
    return rows, list(spans.values())


def _ridge_motion(items, fps, index, neck, scale=None):
    seqs = [_motion15(c["positions"], fps, index, neck) for c in items]
    x, mask = _pad(seqs)
    if scale is None:
        real = x[mask]
        scale = float(np.median(np.sqrt((real ** 2).sum(-1).mean(-1)))) or 1.0
    return (x / scale).reshape(len(x), x.shape[1], -1), mask.sum(1).astype(np.int64), scale


def _prepare_ridge(bank, output, epochs, seed, sbert, origin_sha, strong_rules_path=None, ids=None, **_):
    import torch
    from ridge_gesture import cli
    from ridge_gesture.model import encode_text, load_checkpoint
    from ridge_gesture.pipeline import GCA
    _threads()
    names, fps = bank["joint_names"], bank["fps"]
    index = _feature_index(names)
    neck = _sub_index(index, names, "Neck") or 0
    strong = _strong_rules(strong_rules_path, ids)
    rules, spans = _ridge_rule_spans(bank, strong)
    train = bank["associations"]
    if len(train) < 2:
        raise ValueError("RIDGE fallback needs at least two paired association clips")
    texts = [str(a["text"]) for a in train]
    fallback_encoder = fit_text_encoder(texts, sbert)
    rule_encoder = fit_text_encoder([r["phrase"] for r in rules] + texts, sbert)
    for r in rules:
        r["embedding"] = rule_encoder.encode([r["phrase"]])[0].tolist()
    motion, lengths, scale = _ridge_motion(train, fps, index, neck)
    pairs = output / "train-pairs.npz"
    np.savez(pairs, motion=motion, text_embeddings=fallback_encoder.encode(texts), lengths=lengths,
             ids=np.asarray([str(a["id"]) for a in train]), texts=np.asarray(texts),
             speakers=np.asarray([str(a.get("source", {}).get("speaker", "")) for a in train]))
    stage1, checkpoint = output / "ridge-pretrain.pt", output / "ridge-model.pt"
    # Demo budget: about RIDGE_DEMO_STEPS optimiser steps per stage (pretrain uses one full batch per epoch).
    run_epochs = max(1, min(int(epochs), RIDGE_DEMO_STEPS // max(1, math.ceil(len(train) / 64))))
    started = time.perf_counter()
    # Early stopping monitors validation loss only when the split is large enough to mean something.
    val_fraction = "0.1" if len(train) >= RIDGE_MIN_VALIDATION_PAIRS else "0"
    common = ["--pairs", str(pairs), "--seed", str(int(seed)), "--log-every", "0", "--epochs", str(run_epochs),
              "--val-fraction", val_fraction]
    _quiet(cli.main, ["train", "--output", str(stage1), "--preset", "pretrain",
                      "--history", str(output / "pretrain-history.json"), *common])
    _quiet(cli.main, ["train", "--output", str(checkpoint), "--preset", "finetune", "--init", str(stage1),
                      "--history", str(output / "finetune-history.json"), *common])
    seconds = time.perf_counter() - started
    pairs.unlink(missing_ok=True); stage1.unlink(missing_ok=True)
    model, ck = load_checkpoint(checkpoint)
    clips = bank["clips"]
    bank_motion, bank_lengths, _ = _ridge_motion(clips, fps, index, neck, scale)
    with torch.no_grad():
        zm = model.encode_motion(torch.from_numpy(bank_motion), torch.from_numpy(bank_lengths)).numpy()
        zt = encode_text(model, torch.from_numpy(fallback_encoder.encode([c["text"] for c in clips]))).numpy()
        train_text = encode_text(model, torch.from_numpy(fallback_encoder.encode(texts))).numpy()
    train_motion = ck["motion_latents"].cpu().numpy()
    retrieved = (zt @ zm.T).argmax(1)
    top1 = float((retrieved == np.arange(len(clips))).mean())
    gca = GCA(max(2, min(10, len(train) // 8)), 3, int(seed)).fit(train_text, train_motion)
    history = json.loads((output / "finetune-history.json").read_text(encoding="utf-8"))["history"]
    best = min(history, key=lambda r: r.get("val_loss", r["loss"]))
    metrics = {"train_pairs": int(ck["train_pairs"]), "val_pairs": int(ck["val_pairs"]), "heldout_pairs": len(clips),
               "heldout_top1": round(top1, 4), "heldout_chance": round(1 / len(clips), 4),
               "heldout_gca_retrieved": round(gca.score(zt, zm[retrieved]), 4),
               "heldout_gca_ground_truth": round(gca.score(zt, zm), 4),
               "heldout": "bank clip transcripts -> bank clip motion (never trained on)",
               "training": "ridge_gesture.cli train: pretrain preset then finetune preset (--init)",
               "training_epochs": run_epochs, "best_epoch": int(ck["best_epoch"]), "training_seconds": round(seconds, 2),
               "rule_count": len(rules), "rule_routes": dict(Counter(r["route"] for r in rules)),
               "phrase_spans": len(spans)}
    playback = _playback_bank(bank, [dict(c, id=str(c["id"])) for c in clips] + spans,
                              base=list(bank.get("base_ids") or [str(c["id"]) for c in clips[:3]]),
                              origin_sha=origin_sha)
    return {"strong_rules": rules, "fallback_ids": [str(c["id"]) for c in clips],
            "fallback_latents": np.round(zm, 6).tolist(), "motion_scale": scale,
            "fallback_encoder": fallback_encoder.to_dict(), "text_encoder": rule_encoder.to_dict(),
            "training_loss": _round(best.get("val_loss", best["loss"])), "metrics": metrics,
            "training_pairs": int(ck["train_pairs"]), "tau": float(ck.get("tau", .07)),
            "algorithm": "RIDGE (ridge_gesture): strong rules on phrase-timed spans (align_phrase) with "
                         "hybrid_retrieve over every 3-10-word span, RidgeModel fallback trained in two stages on "
                         "association windows with the bank held out",
            "playback": playback}


_RIDGE_CACHE: dict = {}


def _ridge_model(artifact):
    import torch
    from ridge_gesture.model import load_checkpoint
    path = Path(artifact) / "ridge-model.pt"
    stamp = (str(path), path.stat().st_mtime_ns)
    model = _RIDGE_CACHE.get(stamp)
    if model is None:
        with _MODEL_LOCK:  # same double-checked pattern as the Sentence-BERT loader
            model = _RIDGE_CACHE.get(stamp)
            if model is None:
                model = load_checkpoint(path)[0]
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                _RIDGE_CACHE.clear()
                _RIDGE_CACHE[stamp] = model
    return model


def _query_ridge(text, params, info, artifact, floor):
    import torch
    from ridge_gesture.model import encode_text
    from ridge_gesture.pipeline import hybrid_retrieve, split_even
    threshold = _float_param(params, "strong_rule_threshold", .65)
    model = _ridge_model(artifact)
    rule_encoder = text_encoder(info["text_encoder"])
    fallback_encoder = text_encoder(info["fallback_encoder"])
    latents = np.asarray(info["fallback_latents"], np.float32)
    ids = info["fallback_ids"]
    memo = {}

    def encode_fallback(chunk):
        if chunk not in memo:
            with torch.no_grad():
                z = encode_text(model, torch.from_numpy(fallback_encoder.encode([chunk]))).numpy()[0]
            memo[chunk] = z / max(np.linalg.norm(z), 1e-8)
        return memo[chunk]

    rules = info["strong_rules"]
    if rules:
        rows = hybrid_retrieve(text, rules, rule_encoder.encode, threshold, latents, ids, encode_fallback)
    else:
        words = WORDS.findall(text)
        rows = []
        for a, b in split_even(0, len(words), 6):
            chunk = " ".join(words[a:b]); score = latents @ encode_fallback(chunk); j = int(score.argmax())
            rows.append({"text": chunk, "gesture_id": ids[j], "source": "fallback", "similarity": float(score[j])})
    by_span = {r["gesture_id"]: r for r in rules}
    tau = float(info.get("tau", .07))
    slots = []
    for row in rows:
        if row["source"] == "rule":
            rule = by_span.get(row["gesture_id"], {})
            slots.append((row["text"], row["gesture_id"], rule.get("provenance", {}),
                          {"route": rule.get("route", "strong_rule"), "similarity": round(row["similarity"], 5),
                           "rule_phrase": row.get("rule_phrase"), "clip_id": rule.get("clip_id")},
                          row["similarity"]))
            continue
        coverage, known = fallback_encoder.coverage(row["text"])
        detail = {"similarity": round(row["similarity"], 5), "vocabulary_coverage": round(coverage, 3)}
        if not known or coverage <= 0:
            slots.append((row["text"], None, {"kind": "idle", "reason": "no in-vocabulary content word"}, detail, 0.0))
            continue
        scores = latents @ encode_fallback(row["text"])
        weights = np.exp((scores - scores.max()) / tau)
        share = float(weights[int(scores.argmax())] / weights.sum())
        # Confidence: softmax share of the best held-out clip scaled by content-word coverage.
        slots.append((row["text"], row["gesture_id"], {"kind": "local_training"},
                      {**detail, "route": "trained_text_motion_fallback"}, share * coverage))
    return slots, {"strong_rule_threshold": threshold}, rule_encoder.label, {"spans": rows}


# ----------------------------------------------------------------------------
# Prepare / query entry points

def _default_floor(mode, encoder):
    """Default idle threshold per mode (a request's ``min_similarity`` overrides it)."""
    if mode == "automatic":
        return None
    if mode in {"wild", "multilingual"}:
        return POSE_SBERT_FLOOR if encoder.get("kind") == "sbert" else POSE_TFIDF_FLOOR
    return SBERT_MIN_SIMILARITY if encoder.get("kind") == "sbert" else MIN_SIMILARITY


def prepare(bank_path, output_dir, mode, *, epochs=60, seed=7, strong_rules_path=None, sbert=None, **_):
    """Prepare a local, mode-specific retrieval index from an ignored BEAT bank."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    source = Path(bank_path)
    sbert = sbert_setting(sbert)
    key = cache_key(mode, source, epochs=epochs, seed=seed, strong_rules_path=strong_rules_path, sbert=sbert)
    bank, origin_sha = _read_json(source)
    ids, base = _check_bank(bank)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    options = dict(bank_path=source, output=output, epochs=epochs, seed=int(seed), sbert=sbert,
                   origin_sha=origin_sha, strong_rules_path=strong_rules_path, ids=ids)
    builders = {"automatic": lambda: _prepare_automatic(bank, base, int(seed)),
                "wild": lambda: _prepare_wild(bank, **options),
                "multilingual": lambda: _prepare_multilingual(bank, **options),
                "ridge": lambda: _prepare_ridge(bank, **options)}
    result = builders[mode]()
    playback = result.pop("playback", None)
    playback_path = source
    if playback is not None:
        playback_path = output / f"{mode}-bank.json"
        playback_path.write_text(json.dumps(playback, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        playback_ids = _check_playback(playback)
    else:
        playback_ids = result.pop("playback_ids")
    try:
        portable = os.path.relpath(playback_path.resolve(), output.resolve())
    except ValueError:
        portable = str(playback_path.resolve())
    encoder = result.get("text_encoder", {})
    info = {"mode": mode, "bank_path": portable, "bank_sha256": _sha256(playback_path),
            "origin_bank_sha256": origin_sha, "cache_key": key, "code_version": CODE_VERSION,
            "base_ids": (playback or bank).get("base_ids", base), "playback_ids": playback_ids, "fps": bank["fps"],
            "joint_order": bank["joint_names"], "axisSigns": bank.get("axisSigns", [-1, 1, 1]),
            "data_label": "local BEAT-derived public-data demo; fitted locally",
            "training_pairs": len(bank["associations"]), "seed_pairs": 3, "bank_count": len(playback_ids),
            "seed": int(seed), "epochs": int(epochs),
            "min_similarity": _default_floor(mode, encoder),
            "bank_provenance": bank.get("provenance"), **result}
    info["prepare_seconds"] = round(time.perf_counter() - started, 2)
    index = output / "index.json"
    index.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    info["suggested_queries"] = _suggest(mode, info, output, bank)
    index.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    rules = info.get("rules", info.get("strong_rules", []))
    return {"artifact_dir": str(output), "mode": mode, "rule_count": len(rules),
            "metrics": {"training_pairs": info["training_pairs"], "training_loss": info.get("training_loss"),
                        "prepare_seconds": info["prepare_seconds"], **info.get("metrics", {})}}


def _suggest(mode, info, artifact, bank):
    """Queries that exercise the fitted method, verified by running them through ``query``."""
    def slots(text):
        try:
            return query(text, {}, mode, artifact)["slots"]
        except ValueError:
            return []

    candidates = []
    if mode == "automatic":
        base = set(info["base_ids"])
        candidates = [(_seed_phrases(c)[0], {"seed_rule"}, None) for c in bank["clips"] if str(c["id"]) in base]
        seen = set()
        for r in info["rules"]:
            if r["route"] == "mined_pose_rule" and r["gesture_id"] not in seen and len(WORDS.findall(r["text"])) >= 3:
                seen.add(r["gesture_id"]); candidates.append((r["text"], {"mined_pose_rule"}, None))
    elif mode in {"wild", "multilingual"}:
        # One candidate per cluster, best pose match first; a suggestion must retrieve through its own rule.
        seen = set()
        for r in sorted(info["rules"], key=lambda r: -float(r.get("score", 0))):
            words = WORDS.findall(r["text"])[:POSE_CHUNK_WORDS]
            if len(words) >= 3 and r["cluster_id"] not in seen:
                seen.add(r["cluster_id"]); candidates.append((" ".join(words), {"learned_pose_rule"}, r))
    else:
        candidates = [(r["phrase"], {"strong_rule", "heuristic_rule"}, None) for r in info["strong_rules"][:2]]
        pool = bank["associations"]
        for a in pool[1::max(1, len(pool) // 6)]:
            candidates.append((" ".join(WORDS.findall(a["text"])[:10]), {"trained_text_motion_fallback"}, None))
    examples, limit = [], (5 if mode in {"automatic", "wild", "multilingual"} else 3)
    for text, accepted, rule in candidates:
        if not text or text in examples:
            continue
        found = slots(text)
        routes = {s["route"] for s in found}
        if not found or not routes & accepted or "idle_no_match" in routes:
            continue
        if rule is not None and any(s.get("rule_source", {}).get("rule_text") not in (None, rule["text"])
                                    for s in found):
            continue  # the chunk matched another rule: the suggestion would not show its own association
        examples.append(text)
        if len(examples) >= limit:
            break
    combined = ". ".join(examples[:3])
    if len(examples) >= 3 and all(s["route"] != "idle_no_match" for s in slots(combined) or [{"route": "idle_no_match"}]):
        examples.append(combined)
    return examples


def _float_param(params, name, default, low=0.0, high=1.0):
    value = params.get(name, default)
    if isinstance(value, list):
        value = value[0] if value else default
    if value is None or (isinstance(value, str) and not value.strip()):
        value = default
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def query(text, params, mode, artifact_dir):
    """Return playable source frames and a transparent retrieval trace."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if not str(text).strip():
        raise ValueError("Query text is empty")
    params = params or {}
    artifact = Path(artifact_dir)
    info, _ = _read_json(artifact / "index.json")
    if info["mode"] != mode:
        raise ValueError(f"Prepared artifact is {info['mode']}, not {mode}")
    source = Path(info["bank_path"])
    if not source.is_absolute():
        source = artifact / source
    if not source.is_file():
        raise ValueError("Prepared BEAT bank is missing or has changed; run prepare again")
    bank, digest = _read_json(source)
    if digest != info["bank_sha256"]:
        raise ValueError("Prepared BEAT bank is missing or has changed; run prepare again")
    by_id = {str(c["id"]): c for c in bank["clips"]}
    default_floor = info.get("min_similarity", MIN_SIMILARITY)
    requested = params.get("min_similarity")
    if isinstance(requested, list):
        requested = requested[0] if requested else None
    if default_floor is None and (requested is None or (isinstance(requested, str) and not requested.strip())):
        floor = None  # Automatic (Algorithm 2): no similarity floor; only chunks without shared vocabulary idle
    else:
        floor = _float_param(params, "min_similarity", default_floor)
    seed = int(_float_param(params, "seed", info["seed"], 0, 2 ** 31))
    trace_extra = {}
    if mode == "automatic":
        slots, extra, label, trace_extra = _query_automatic(text, params, info, bank, floor, seed)
    elif mode == "wild":
        slots, extra, label, trace_extra = _query_wild(text, params, info, floor, seed)
    elif mode == "multilingual":
        slots, extra, label, trace_extra = _query_multilingual(text, params, info, floor, seed)
    else:
        slots, extra, label, trace_extra = _query_ridge(text, params, info, artifact, floor)
    out = []
    fps = bank["fps"]
    playback = set(info["playback_ids"])
    for chunk, gid, provenance, detail, score in slots:
        detail = dict(detail)
        route = detail.pop("route", "idle_no_match" if gid is None else "rule")
        blend = int(detail.pop("blend_frames", info.get("units", {}).get("blend_frames", 0)) or 0)
        if gid is None:
            count = max(15, int(round(fps * .35 * max(1, len(WORDS.findall(chunk))))))
            out.append({"gesture_id": IDLE_ID, "text": chunk, "frames": _rest_frames(bank, count),
                        "route": "idle_no_match", "confidence": 0.0, "source": {"kind": "idle"},
                        "rule_source": provenance, "blend_frames": 0, **detail})
            continue
        if gid not in by_id or gid not in playback:
            raise ValueError("Retrieval index references a gesture outside its playback bank")
        clip = by_id[gid]
        out.append({"gesture_id": gid, "text": chunk, "frames": clip["positions"], "route": route,
                    "confidence": round(float(score), 5), "source": clip.get("source", {}), "rule_source": provenance,
                    "blend_frames": blend, **detail})
    routes = [x["route"] for x in out]
    stored = info.get("metrics", {})
    retrieval_text = trace_extra.pop("retrieval_text", text)
    translation = trace_extra.pop("translation", None)
    rules = info.get("rules", info.get("strong_rules", []))
    return {"fps": fps, "joint_order": bank["joint_names"],
            "axisSigns": bank.get("axisSigns", [-1, 1, 1]), "slots": out,
            "no_match": bool(out) and all(r == "idle_no_match" for r in routes),
            "text_encoder": label,
            "trace": {"input": text, "retrieval_text": retrieval_text, "routes": routes,
                      "translation": translation, "unit_refinement": info.get("units"),
                      "text_encoder": label, **trace_extra},
            "algorithm": info["algorithm"], "data_label": info["data_label"],
            "metrics": {"rule_count": len(rules), "bank_count": info["bank_count"], "seed_pairs": info["seed_pairs"],
                        "extended_rules": max(0, len(info.get("rules", [])) - info["seed_pairs"]),
                        "refined_units": info.get("units", {}).get("count", 0),
                        "training_pairs": info["training_pairs"], "training_loss": info.get("training_loss"),
                        "route_counts": dict(Counter(routes)), "min_similarity": floor, "text_encoder": label,
                        **{k: v for k, v in stored.items() if k in {"heldout_top1", "heldout_chance",
                                                                     "heldout_cross_view_top1", "heldout_gca_retrieved"}},
                        **extra}}
