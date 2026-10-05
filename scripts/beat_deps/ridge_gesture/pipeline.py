"""RIDGE core: phrase alignment, hybrid rule/fallback retrieval and GCA."""
from __future__ import annotations

import math
import re

import numpy as np

WORD = re.compile(r"[\w']+")
STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "is", "are", "was", "were",
        "be", "it", "that", "this"}


def heuristic_phrases(text, min_words=3, max_words=10, limit=5):
    """Transparent content-word heuristic (not the paper's LLM extraction)."""
    words = WORD.findall(text); candidates = []
    for n in range(min_words, min(max_words, len(words)) + 1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]; score = sum(w.casefold() not in STOP for w in span) / n + .02 * n
            candidates.append((score, i, " ".join(span)))
    chosen = []; occupied = set()
    for _, i, phrase in sorted(candidates, reverse=True):
        ids = set(range(i, i + len(phrase.split())))
        if not ids & occupied:
            chosen.append(phrase); occupied |= ids
        if len(chosen) == limit:
            break
    return chosen


def _token_stream(words):
    """Casefolded tokens of a timed word list and the word index of each token."""
    tokens, owner = [], []
    for i, item in enumerate(words):
        for tok in WORD.findall(str(item["word"]).casefold()):
            tokens.append(tok); owner.append(i)
    return tokens, owner


def phrase_occurrences(phrase, words):
    """Every contiguous match of ``phrase`` in ``words`` as ``(first_word, last_word_exclusive)``."""
    target = [x.casefold() for x in WORD.findall(phrase)]
    tokens, owner = _token_stream(words)
    if not target:
        return []
    out = []
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i + len(target)] == target:
            span = (owner[i], owner[i + len(target) - 1] + 1)
            if not out or out[-1] != span:
                out.append(span)
    return out


def align_phrase(phrase, words, occurrence=0):
    """Frame span ``(start, end)`` of the ``occurrence``-th contiguous match."""
    found = phrase_occurrences(phrase, words)
    if len(found) <= occurrence:
        raise ValueError(f"phrase is not a contiguous transcript span: {phrase}")
    a, b = found[occurrence]
    return int(words[a]["start_frame"]), int(words[b - 1]["end_frame"])


def validate_phrases(phrases, words, min_words=3, max_words=10):
    """Check proposed phrases against the timed transcript.

    A phrase must have ``min_words``..``max_words`` words and occur as a
    contiguous span. When the same phrase is proposed several times, the k-th
    proposal is bound to the k-th occurrence in the transcript. Returns
    ``(accepted, rejected)``; accepted rows hold word and frame spans.
    """
    accepted, rejected, seen = [], [], {}
    for raw in phrases:
        phrase = " ".join(WORD.findall(str(raw)))
        n = len(phrase.split())
        if not min_words <= n <= max_words:
            rejected.append({"phrase": str(raw), "reason": f"has {n} words; expected {min_words}-{max_words}"})
            continue
        key = phrase.casefold()
        found = phrase_occurrences(phrase, words)
        k = seen.get(key, 0)
        if not found:
            rejected.append({"phrase": str(raw), "reason": "not a contiguous span of the transcript"})
            continue
        if k >= len(found):
            rejected.append({"phrase": str(raw), "reason": "proposed more often than it occurs"})
            continue
        seen[key] = k + 1
        a, b = found[k]
        accepted.append({"phrase": phrase, "occurrence": k, "occurrences": len(found),
                         "word_start": a, "word_end": b,
                         "start_frame": int(words[a]["start_frame"]), "end_frame": int(words[b - 1]["end_frame"])})
    return accepted, rejected


def neck_normalize(x, neck=1, coords=3):
    x = np.asarray(x, np.float32); p = x.reshape(*x.shape[:-1], -1, coords)
    return (p - p[..., neck:neck + 1, :]).reshape(x.shape)


def overlapping_segments(words, start, min_words=3, max_words=10):
    """Candidate spans sharing a start position, longest first."""
    limit = min(max_words, len(words) - start)
    if limit < min_words:
        return []
    return [" ".join(words[start:start + n]) for n in range(limit, min_words - 1, -1)]


def fallback_segments(text, size=6):
    w = WORD.findall(text); return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def split_even(start, end, size=6):
    """Cut ``[start, end)`` into the fewest near-equal chunks of at most ``size`` words."""
    n = end - start
    if n <= 0:
        return []
    count = math.ceil(n / size)
    bounds = [start + round(i * n / count) for i in range(count + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def hybrid_retrieve(text, rules, embed, rule_threshold, fallback_text_latents, fallback_ids, encode_fallback,
                    min_words=3, max_words=10, fallback_words=6):
    """RIDGE inference: rule spans first, learned fallback for the rest.

    Every 3–10-word span at every start position is scored against the rule
    base. Spans above ``rule_threshold`` are accepted greedily by score (the
    best-scoring span wins, not the longest, so leading or trailing filler is
    not absorbed) without overlapping. The words between accepted rule spans are
    re-segmented into chunks of at most ``fallback_words`` and sent to the
    contrastive fallback, so a fallback chunk always stops at the next rule.
    """
    if not rules:
        raise ValueError("rule map is empty")
    if len(fallback_text_latents) == 0 or len(fallback_ids) != len(fallback_text_latents):
        raise ValueError("fallback index is empty or inconsistent")
    bank = np.asarray([r["embedding"] for r in rules], np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True).clip(1e-8)
    words = WORD.findall(text)
    spans = [(s, s + n) for s in range(len(words)) for n in range(min_words, max_words + 1) if s + n <= len(words)]
    accepted = []
    if spans:
        queries = np.asarray(embed([" ".join(words[a:b]) for a, b in spans]), np.float32)
        if queries.ndim != 2 or queries.shape[1] != bank.shape[1]:
            raise ValueError("query and rule embedding dimensions differ")
        queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(1e-8)
        scores = queries @ bank.T
        best_rule, best = scores.argmax(1), scores.max(1)
        used = np.zeros(len(words), bool)
        # Highest score first; on equal scores prefer the longer span, then the earlier one.
        for k in sorted(range(len(spans)), key=lambda k: (-best[k], -(spans[k][1] - spans[k][0]), spans[k][0])):
            if best[k] < rule_threshold:
                break
            a, b = spans[k]
            if used[a:b].any():
                continue
            used[a:b] = True
            r = int(best_rule[k])
            accepted.append({"text": " ".join(words[a:b]), "gesture_id": rules[r]["gesture_id"], "source": "rule",
                             "similarity": float(best[k]), "rule_phrase": rules[r].get("phrase"),
                             "word_start": a, "word_end": b})
    accepted.sort(key=lambda x: x["word_start"])
    out, cursor = [], 0
    for item in accepted + [None]:
        stop = item["word_start"] if item else len(words)
        for a, b in split_even(cursor, stop, fallback_words):
            chunk = " ".join(words[a:b]); z = encode_fallback(chunk)
            score = fallback_text_latents @ z; j = int(score.argmax())
            out.append({"text": chunk, "gesture_id": fallback_ids[j], "source": "fallback",
                        "similarity": float(score[j]), "word_start": a, "word_end": b})
        if item:
            out.append(item); cursor = item["word_end"]
    return out


def _unit_rows(x):
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


class GCA:
    """Gesture Cluster Affinity. Text and gesture embeddings are L2-normalised
    before Bisecting K-Means, so clustering (Euclidean on the unit sphere)
    agrees with the cosine similarity used for scoring, as in the paper."""

    def __init__(self, text_clusters=100, gesture_clusters=20, seed=0):
        self.k = text_clusters; self.gk = gesture_clusters; self.seed = seed

    def fit(self, text, motion):
        from sklearn.cluster import BisectingKMeans
        text, motion = _unit_rows(text), _unit_rows(motion)
        self.text_model = BisectingKMeans(n_clusters=min(self.k, len(text)), random_state=self.seed).fit(text)
        labels = self.text_model.labels_; self.gesture_models = {}
        for k in np.unique(labels):
            group = motion[labels == k]; n = min(self.gk, len(group))
            self.gesture_models[int(k)] = BisectingKMeans(n_clusters=n, random_state=self.seed).fit(group)
        return self

    def score(self, text, motion):
        labels = self.text_model.predict(_unit_rows(text)); values = []
        for k, g in zip(labels, _unit_rows(motion)):
            centers = _unit_rows(self.gesture_models[int(k)].cluster_centers_)
            values.append(float(np.max(centers @ g)))
        return float(np.mean(values))
