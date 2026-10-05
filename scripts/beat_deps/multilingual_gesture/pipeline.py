"""Paper algorithms: Algorithm 1 unit extraction, GestureCLR augmentation,
Bisecting K-Means clustering and six-word multilingual retrieval."""
from __future__ import annotations

import math
import re

import numpy as np

NOISE_VARIANCES = (0.001, 0.01, 0.1)
CONDITIONS = ("clean", "noise", "shift_mean", "shift_zero")
ALL_CONDITIONS = CONDITIONS + ("noise_shift",)
WORD = re.compile(r"[\w']+")
SENTENCE = re.compile(r"[^.!?。！？]+[.!?。！？]*")


# ---------------------------------------------------------------------------
# Scale normalisation


def motion_scale(x, length=None) -> float:
    """Body scale of a root/neck-centred sequence: median per-frame RMS coordinate.

    Dividing by this makes thresholds and noise levels independent of whether
    poses are in metres, centimetres or pixels.
    """
    x = np.asarray(x, np.float64)
    if length is not None:
        x = x[: int(length)]
    x = x.reshape(len(x), -1)
    if not len(x):
        return 1.0
    scale = float(np.median(np.sqrt((x ** 2).mean(axis=1))))
    return scale if scale > 1e-8 else 1.0


def normalize_sequence(x, length=None) -> np.ndarray:
    """Return ``x`` divided by its own body scale (computed on valid frames)."""
    x = np.asarray(x, np.float32)
    return (x / motion_scale(x, length)).astype(np.float32)


def normalize_batch(x, lengths=None) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lengths = full_lengths(x) if lengths is None else np.asarray(lengths)
    return np.stack([normalize_sequence(seq, n) for seq, n in zip(x, lengths)]) if len(x) else x


def full_lengths(x) -> np.ndarray:
    return np.full(len(x), np.asarray(x).shape[1], dtype=np.int64)


# ---------------------------------------------------------------------------
# Algorithm 1: gesture-unit extraction


def candidate_stats(motion, lo: int, hi: int, scale: float = 1.0):
    """Closure and variance for every clip ``motion[s:s+L]`` with ``lo<=L<=hi``.

    Returns arrays ``closure[S, L']`` and ``variance[S, L']`` (``L'=hi-lo+1``),
    with ``inf``/``-inf`` where the clip runs past the sequence. Closure is
    the RMS coordinate distance between first and last pose; variance is the
    mean per-coordinate variance over the clip. Both are divided by ``scale``
    (closure) and ``scale**2`` (variance).
    """
    x = np.asarray(motion, np.float64).reshape(len(motion), -1)
    n, d = x.shape
    lengths = np.arange(lo, hi + 1)
    closure = np.full((n, len(lengths)), np.inf)
    variance = np.full((n, len(lengths)), -np.inf)
    c1 = np.vstack([np.zeros((1, d)), np.cumsum(x, axis=0)])
    c2 = np.vstack([np.zeros((1, d)), np.cumsum(x * x, axis=0)])
    for j, length in enumerate(lengths):
        count = n - length + 1
        if count <= 0:
            continue
        s = np.arange(count)
        e = s + length
        mean = (c1[e] - c1[s]) / length
        var = (c2[e] - c2[s]) / length - mean ** 2
        variance[:count, j] = np.clip(var, 0, None).mean(axis=1) / scale ** 2
        diff = x[s] - x[e - 1]
        closure[:count, j] = np.sqrt((diff ** 2).mean(axis=1)) / scale
    return closure, variance, lengths


def elbow_threshold(values) -> float:
    """Automatic stand-in for the paper's expert elbow calibration.

    Candidate variances are sorted in ascending order and taken on a log scale.
    Static clips form a steep initial rise; the elbow is the point furthest
    above the chord, where that rise flattens into ordinary gesturing. On a
    linear scale the knee of a heavy-tailed variance curve would keep only the
    top ~10% of clips, whereas the paper's yield (2,035 units from about two
    hours) implies that most non-static clips are kept.
    """
    v = np.sort(np.asarray(values, np.float64)[np.isfinite(values)])
    if len(v) < 3:
        return float(v[0]) if len(v) else 0.0
    if len(v) > 20000:
        v = v[np.linspace(0, len(v) - 1, 20000).astype(int)]
    lv = np.log10(np.clip(v, 1e-12, None))
    if lv[-1] - lv[0] <= 1e-9:
        return float(v[0])
    x = np.linspace(0.0, 1.0, len(lv))
    y = (lv - lv[0]) / (lv[-1] - lv[0])
    return float(v[int(np.argmax(y - x))])


def extract_unit_spans(motion, fps=15, min_seconds=2.0, max_seconds=3.0,
                       variance_threshold="auto", closure_threshold=float("inf"),
                       normalize=True):
    """Algorithm 1 over one continuous take.

    Every (start, end) with a 2–3 s length is a candidate. Candidates are taken
    in order of minimal start/end pose distance (longer first on ties). A
    candidate that still lies entirely in unused motion is removed from the
    sequence; it becomes a unit when its variance passes the threshold and is
    discarded otherwise, as in the paper's loop. The loop ends when no unused
    stretch of at least ``min_seconds`` remains.

    ``variance_threshold`` is a number or ``"auto"`` (elbow of the sorted
    candidate variances). With ``normalize=True`` closure and variance are
    measured in body-scale units (see :func:`motion_scale`).

    Returns ``(spans, info)`` where ``spans`` is a list of ``(start, end)``.
    """
    x = np.asarray(motion, np.float32)
    x = x.reshape(len(x), -1)
    lo, hi = int(round(min_seconds * fps)), int(round(max_seconds * fps))
    if lo < 2 or hi < lo:
        raise ValueError("unit length bounds must satisfy 2 <= min <= max frames")
    scale = motion_scale(x) if normalize else 1.0
    info = {"scale": scale, "normalized": bool(normalize), "min_frames": lo, "max_frames": hi}
    if len(x) < lo:
        info.update(variance_threshold=None, candidates=0)
        return [], info
    closure, variance, lengths = candidate_stats(x, lo, hi, scale)
    valid = np.isfinite(closure)
    if isinstance(variance_threshold, str):
        if variance_threshold != "auto":
            raise ValueError("variance threshold must be a number or 'auto'")
        threshold = elbow_threshold(variance[valid])
        info["variance_mode"] = "auto-elbow"
    else:
        threshold = float(variance_threshold)
        info["variance_mode"] = "fixed"
    info["variance_threshold"] = threshold
    starts, cols = np.nonzero(valid & (closure <= closure_threshold))
    info["candidates"] = int(len(starts))
    order = np.lexsort((-lengths[cols], closure[starts, cols]))
    used = np.zeros(len(x) + 1, bool)
    spans, discarded = [], 0
    for k in order:
        s = int(starts[k]); e = s + int(lengths[cols[k]])
        if used[s:e].any():
            continue
        used[s:e] = True
        if variance[s, cols[k]] >= threshold:
            spans.append((s, e))
        else:
            discarded += 1
    spans.sort()
    info["units"] = len(spans)
    info["discarded_low_variance"] = discarded
    return spans, info


def extract_units(motion, fps=15, min_seconds=2.0, max_seconds=3.0,
                  variance_threshold="auto", closure_threshold=float("inf"), normalize=True):
    """Return the clips selected by :func:`extract_unit_spans` (in time order)."""
    spans, _ = extract_unit_spans(motion, fps, min_seconds, max_seconds,
                                  variance_threshold, closure_threshold, normalize)
    x = np.asarray(motion, np.float32)
    return [x[s:e] for s, e in spans]


def pad_units(clips, length, mode="edge"):
    """Stack variable-length clips into ``[N,length,D]`` plus their true lengths."""
    if any(len(c) > length for c in clips):
        raise ValueError("a clip is longer than the padded length")
    padded = np.stack([np.pad(c, ((0, length - len(c)), (0, 0)), mode=mode) for c in clips])
    return padded.astype(np.float32), np.asarray([len(c) for c in clips], np.int64)


# ---------------------------------------------------------------------------
# GestureCLR augmentation (paper Section III-C)


def add_noise(x, rng, variance):
    """Gaussian noise with the given *variance* (std = sqrt(variance))."""
    return (x + rng.normal(0.0, math.sqrt(variance), x.shape)).astype(np.float32)


def temporal_shift(x, rng, length=None, fill="mean", segment=30, buffer=45, max_offset=15):
    """Paper temporal shift: a random contiguous ``segment``-frame piece of the
    valid sequence is inserted at a random offset in ``[1, max_offset]`` of a
    ``buffer``-frame sequence filled with the mean pose or with zero poses."""
    x = np.asarray(x, np.float32)
    n = len(x) if length is None else int(length)
    seg = min(segment, n)
    src = int(rng.integers(0, n - seg + 1))
    offset = int(rng.integers(1, min(max_offset, buffer - seg) + 1)) if buffer > seg else 0
    if fill == "mean":
        out = np.broadcast_to(x[:n].mean(0), (buffer, x.shape[-1])).copy()
    elif fill == "zero":
        out = np.zeros((buffer, x.shape[-1]), np.float32)
    else:
        raise ValueError("fill must be 'mean' or 'zero'")
    out[offset:offset + seg] = x[src:src + seg]
    return out


def augment_2d(x, rng, length=None, conditions=CONDITIONS, variances=NOISE_VARIANCES):
    """Draw one augmentation condition per sample (not all combined).

    ``clean`` returns the input, ``noise`` adds Gaussian noise with a variance
    from ``variances``, ``shift_mean``/``shift_zero`` apply the paper's 30-in-45
    shift with mean or zero fill, and ``noise_shift`` combines both. Returns
    ``(sequence, valid_length, condition)``; the sequence has the input's
    number of frames (shifted outputs fill the 45-frame buffer, padded or cut
    to that size).
    """
    x = np.asarray(x, np.float32)
    n = len(x) if length is None else int(length)
    condition = str(rng.choice(list(conditions)))
    if condition == "clean":
        return x.copy(), n, condition
    if condition == "noise":
        out = x.copy(); out[:n] = add_noise(x[:n], rng, float(rng.choice(variances)))
        return out, n, condition
    fill = "zero" if condition == "shift_zero" else "mean"
    if len(x) >= 45:
        shifted = temporal_shift(x, rng, n, fill=fill)
    else:
        # Shorter windows keep the paper's 30:45 proportion inside their own length.
        segment = max(1, int(round(n * 30 / 45)))
        shifted = temporal_shift(x, rng, n, fill=fill, segment=segment, buffer=len(x),
                                 max_offset=max(len(x) - segment, 1))
    if condition == "noise_shift":
        shifted = add_noise(shifted, rng, float(rng.choice(variances)))
    frames = len(x)
    if len(shifted) >= frames:
        return shifted[:frames], frames, condition
    out = np.concatenate([shifted, np.repeat(shifted[-1:], frames - len(shifted), 0)])
    return out, len(shifted), condition


# ---------------------------------------------------------------------------
# Clustering and retrieval


def bisect(latents, k, seed=0):
    from sklearn.cluster import BisectingKMeans
    labels = BisectingKMeans(n_clusters=min(k, len(latents)), random_state=seed).fit_predict(latents)
    n = labels.max() + 1
    centers = np.stack([latents[labels == i].mean(0) for i in range(n)])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
    return labels, centers


def sixgrams(text):
    w = WORD.findall(text)
    return [" ".join(w[i:i + 6]) for i in range(0, len(w), 6)]


def sixgram_spans(text, size=6):
    """Six-word chunks with word indices and character offsets in ``text``."""
    matches = list(WORD.finditer(text))
    out = []
    for i in range(0, len(matches), size):
        group = matches[i:i + size]
        out.append({"text": " ".join(m.group(0) for m in group), "word_start": i,
                    "word_end": i + len(group), "char_start": group[0].start(),
                    "char_end": group[-1].end()})
    return out


def split_long_text(text, max_words=30):
    """Paper timing rule: input over ``max_words`` words is split into
    sentence-level chunks. A single sentence that is still too long is cut
    into ``max_words``-word pieces."""
    if len(WORD.findall(text)) <= max_words:
        return [text.strip()] if text.strip() else []
    chunks = []
    for sentence in (m.group(0).strip() for m in SENTENCE.finditer(text)):
        if not sentence:
            continue
        words = sentence.split()
        if len(WORD.findall(sentence)) <= max_words:
            chunks.append(sentence)
            continue
        piece = []
        for token in words:
            piece.append(token)
            if len(WORD.findall(" ".join(piece))) >= max_words:
                chunks.append(" ".join(piece)); piece = []
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def require_english(text, source_language, translations):
    """Dictionary translation with the original strict behaviour."""
    if source_language.lower().startswith("en"):
        return text
    if text not in translations:
        raise ValueError("non-English input requires an explicit English translation in --translations")
    return translations[text]


def retrieve(english_text, rules, encode, clusters, seed=0, min_similarity=None, idle_id="idle",
             blend_frames=5):
    """Six-word Sentence-BERT lookup with random choice inside the matched cluster.

    With ``min_similarity`` set, a chunk whose best rule is below the floor plays
    ``idle_id`` instead (paper: "a minimum threshold can be applied ... and an
    idle gesture can be played").
    """
    if not rules:
        raise ValueError("rule map is empty")
    rng = np.random.default_rng(seed)
    bank = np.asarray([r["text_embedding"] for r in rules], np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True).clip(1e-8)
    out = []
    for span in sixgram_spans(english_text):
        q = np.asarray(encode([span["text"]])[0], np.float32)
        q /= max(np.linalg.norm(q), 1e-8)
        sims = bank @ q
        i = int(sims.argmax())
        cid = int(rules[i]["cluster_id"])
        idle = min_similarity is not None and float(sims[i]) < float(min_similarity)
        gid = idle_id if idle else str(rng.choice(clusters[cid]))
        out.append({"english_text": span["text"], "gesture_id": gid, "cluster_id": None if idle else cid,
                    "similarity": float(sims[i]), "idle": bool(idle), "blend_frames": int(blend_frames),
                    "word_start": span["word_start"], "word_end": span["word_end"],
                    "char_start": span["char_start"], "char_end": span["char_end"]})
    return out


def map_word_spans(gestures, english_words, tts_words):
    """Proportionally map each gesture's English word span onto the words of
    the spoken (TTS) text, so translated speech can drive gesture timing."""
    total_en, total_tts = max(english_words, 1), max(tts_words, 0)
    for g in gestures:
        start = int(round(g["word_start"] * total_tts / total_en))
        end = int(round(g["word_end"] * total_tts / total_en))
        g["tts_word_start"], g["tts_word_end"] = start, max(end, min(start + 1, total_tts))
    return gestures


def schedule(gestures, audio_seconds=None, word_times=None, start_key="tts_word_start", end_key="tts_word_end"):
    """Paper timing: coarse pacing (audio duration / gesture count), refined by
    word-level TTS timestamps ``[{"start": s, "end": e}, ...]`` when supplied."""
    if audio_seconds is not None and gestures:
        step = float(audio_seconds) / len(gestures)
        for i, g in enumerate(gestures):
            g["start_seconds"], g["end_seconds"] = i * step, (i + 1) * step
            g["duration_seconds"] = step
    if word_times:
        for g in gestures:
            a = g.get(start_key, g.get("word_start")); b = g.get(end_key, g.get("word_end"))
            if a is None or b is None or a >= len(word_times):
                continue
            b = min(max(b, a + 1), len(word_times))
            g["start_seconds"] = float(word_times[a]["start"])
            g["end_seconds"] = float(word_times[b - 1]["end"])
            g["duration_seconds"] = max(g["end_seconds"] - g["start_seconds"], 1e-3)
    return gestures


def _is_english(language):
    return (language or "en").lower().startswith("en")


def multilingual_retrieve(text, source_language, translator, rules, encode, clusters, seed=0,
                          min_similarity=None, idle_id="idle", max_words=30, tts_language=None):
    """Full runtime path: split >30-word input into sentence chunks, translate
    each chunk to English, retrieve per six-word chunk and return speech text.

    Each chunk carries ``tts_text``: the source chunk by default, or its
    translation into ``tts_language`` when that differs from the source.
    """
    source_language = source_language or "en"
    tts_language = tts_language or source_language
    pairs = []
    for chunk in split_long_text(text, max_words):
        english = chunk if _is_english(source_language) else translator.translate(chunk, source_language, "en")
        if tts_language.lower() == source_language.lower():
            spoken = chunk
        elif _is_english(tts_language):
            spoken = english
        else:
            spoken = translator.translate(english, "en", tts_language)
        pairs.append((chunk, english, spoken))
    if not pairs:
        raise ValueError("enter query text")
    chunks, gestures_all, offset = [], [], 0
    for index, (chunk, english, spoken) in enumerate(pairs):
        gestures = retrieve(english, rules, encode, clusters, seed + index, min_similarity, idle_id)
        map_word_spans(gestures, len(WORD.findall(english)), len(spoken.split()))
        for g in gestures:
            # TTS word indices refer to the joined ``tts_text`` so one word-timestamp
            # list from the speech engine can drive :func:`schedule`.
            g["chunk_index"] = index
            g["tts_word_start"] += offset
            g["tts_word_end"] += offset
        offset += len(spoken.split())
        chunks.append({"index": index, "source_text": chunk, "english_text": english,
                       "tts_text": spoken, "gestures": gestures})
        gestures_all.extend(gestures)
    return {"source_text": text, "source_language": source_language,
            "english_text": " ".join(c["english_text"] for c in chunks),
            "tts_text": " ".join(c["tts_text"] for c in chunks), "tts_language": tts_language,
            "chunks": chunks, "gestures": gestures_all}
