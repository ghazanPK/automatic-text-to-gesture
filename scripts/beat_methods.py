"""Local BEAT co-speech retrieval adapters shared by independent paper demos.

Prepared motion, fitted weights, and imported annotations belong in ignored output
directories. This small-data demonstration does not reproduce paper benchmarks.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

MODES = {"automatic", "wild", "multilingual", "ridge"}
WORDS = re.compile(r"[\w']+", re.UNICODE)


def _package(name):
    """Import the canonical package from this standalone repository."""
    try:
        return __import__(name, fromlist=["pipeline"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Install this paper's package before preparing {name}") from exc


def _check_bank(bank):
    names = bank.get("joint_names")
    clips = bank.get("clips", [])
    if not names or not clips or int(bank.get("fps", 0)) <= 0:
        raise ValueError("BEAT bank requires fps, joint_names, and clips")
    if not bank.get("associations"):
        raise ValueError("BEAT bank requires a disjoint association pool")
    ids = [str(c["id"]) for c in clips]
    if len(set(ids)) != len(ids):
        raise ValueError("BEAT gesture IDs must be unique")
    for item in clips + bank["associations"]:
        p = np.asarray(item["positions"], np.float32)
        if p.ndim != 3 or p.shape[1:] != (len(names), 3) or not len(p) or not np.isfinite(p).all():
            raise ValueError("Every BEAT segment needs finite positions [frames,joints,3]")
        if not str(item.get("text", "")).strip():
            raise ValueError("Every BEAT segment needs aligned transcript text")
    base = bank.get("base_ids") or ids[: int(bank.get("seed_count", 3))]
    if len(base) != 3 or len(set(base)) != 3 or not set(base) <= set(ids):
        raise ValueError("BEAT bank needs three distinct base_ids present in clips")
    return ids, base


def _pose(item, frames=16, joint_names=None):
    p = np.asarray(item["positions"], np.float32)
    neck = next((i for i, name in enumerate(joint_names or []) if name.casefold() == "neck"), 0)
    p = p - p[:, neck:neck + 1]
    p = p[np.linspace(0, len(p) - 1, frames).round().astype(int)]
    scale = max(float(np.sqrt(np.mean(p * p))), 1e-4)
    p = p / scale
    if joint_names:
        weights = []
        for name in joint_names:
            low = name.casefold()
            if any(term in low for term in ("leg", "foot", "toe")):
                weights.append(.2)
            elif any(term in low for term in ("shoulder", "arm", "hand")):
                weights.append(2.)
            elif any(term in low for term in ("thumb", "index", "middle", "ring", "little")):
                weights.append(.7)
            else:
                weights.append(.8)
        p *= np.asarray(weights, np.float32)[None, :, None]
    return p


def _project(item, frames=16, joint_names=None):
    return _pose(item, frames, joint_names)[:, :, [0, 1]]


def _text_features(texts, vocab=None):
    tokens = [[w.casefold() for w in WORDS.findall(t)] for t in texts]
    if vocab is None:
        vocab = sorted({w for row in tokens for w in row})
    lookup = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab), np.float32)
    for row in tokens:
        for w in set(row):
            if w in lookup:
                df[lookup[w]] += 1
    idf = np.log((len(tokens) + 1) / (df + 1)) + 1
    matrix = np.zeros((len(tokens), len(vocab)), np.float32)
    for i, row in enumerate(tokens):
        for w in row:
            if w in lookup:
                matrix[i, lookup[w]] += 1
    matrix *= idf
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
    return matrix, vocab, idf


def _encode_text(texts, vocab, idf):
    lookup = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((len(texts), len(vocab)), np.float32)
    for i, text in enumerate(texts):
        for word in WORDS.findall(text.casefold()):
            if word in lookup:
                matrix[i, lookup[word]] += 1
    matrix *= np.asarray(idf, np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)


def _cosine(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def _weak_associations(bank, ids):
    core = _package("automatic_text_to_gesture.core")
    clips = {str(c["id"]): c for c in bank["clips"] if str(c["id"]) in ids}
    # Use the published implementation's mean frame-wise cosine primitive.
    out = []
    neck = next((i for i, name in enumerate(bank["joint_names"]) if name.casefold() == "neck"), 0)
    for item in bank["associations"]:
        query = core.normalize_pose(_project(item, joint_names=bank["joint_names"]), neck_joint=neck)
        scores = {gid: core.average_frame_cosine(query, core.normalize_pose(_project(c, joint_names=bank["joint_names"]), neck_joint=neck)) for gid, c in clips.items()}
        gid = max(scores, key=scores.get)
        out.append({"text": item["text"], "gesture_id": gid, "score": scores[gid],
                    "route": "weak_pose_rule", "source": item.get("source", {})})
    return out


def _refine_multilingual_units(bank):
    """Choose a short closed motion unit with the paper package's extractor."""
    from multilingual_gesture.pipeline import extract_units

    refined = {**bank, "clips": []}
    changes = []
    for clip in bank["clips"]:
        positions = np.asarray(clip["positions"], np.float32)
        units = extract_units(positions, fps=bank["fps"], min_seconds=2.,
                              max_seconds=2.5, variance_threshold=0.,
                              closure_threshold=float("inf"))
        unit = units[0] if units else positions
        count = len(unit)
        copy = {**clip, "positions": unit.tolist()}
        words = clip.get("words")
        if words:
            selected = []
            for word in words:
                start = int(word["start_frame"])
                end = int(word["end_frame"])
                if end > 0 and start < count:
                    selected.append({**word, "start_frame": max(0, start),
                                     "end_frame": min(count, end)})
            if selected:
                copy["words"] = selected
                copy["text"] = " ".join(str(w.get("text", w.get("word", ""))) for w in selected).strip()
        origin = clip.get("source", {})
        source = dict(origin)
        if "start_frame" in source:
            source["end_frame"] = int(source["start_frame"]) + count
        copy["source"] = source
        refined["clips"].append(copy)
        changes.append({"gesture_id": str(clip["id"]), "original_frames": len(positions),
                        "unit_frames": count, "trimmed": count < len(positions),
                        "text_trimmed": bool(words) and copy["text"] != clip["text"],
                        "word_timing_available": bool(words)})
    return refined, changes


def _fit_pose(bank, output, ids, epochs, seed, multilingual=False):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from wild_pose_matching.model import ntxent
    from wild_pose_matching.pipeline import build_rules, cluster_latents
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    clips = [c for c in bank["clips"] if str(c["id"]) in ids]
    pool = bank["associations"]
    if len(pool) < 4:
        raise ValueError("Pose encoder needs at least four association clips for disjoint fit and mining pools")
    train = pool[:len(pool) // 2]
    mine = pool[len(pool) // 2:]
    class Encoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, 32)
            layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, activation="gelu", dropout=0)
            self.net = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.out = nn.Linear(32, 16)
            pos = torch.arange(16).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, 32, 2).float() * (-math.log(10000.) / 32))
            pe = torch.zeros(16, 32)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)
        def forward(self, x):
            return F.normalize(self.out(self.net(self.proj(x) + self.pe[:x.shape[1]]).mean(1)), dim=-1)
    pose2 = Encoder(len(bank["joint_names"]) * 2)
    motion3 = Encoder(len(bank["joint_names"]) * 3)
    optimizer = torch.optim.AdamW(list(pose2.parameters()) + list(motion3.parameters()), lr=.003)
    names = bank["joint_names"]
    a = torch.tensor(np.stack([_project(c, joint_names=names).reshape(16, -1) for c in train]))
    b = torch.tensor(np.stack([_pose(c, joint_names=names).reshape(16, -1) for c in train]))
    # The two views are actual paired projections and 3D motion, with mild
    # augmentation; there are no clip identity embeddings.
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, int(epochs))):
        pose2.train(); motion3.train(); optimizer.zero_grad()
        view = a
        if multilingual and epoch % 4 == 0:
            from multilingual_gesture.pipeline import augment_2d
            augmented = torch.tensor(np.stack([augment_2d(row, rng) for row in a.numpy()]))
            view = .75 * a + .25 * augmented
        za = pose2(view + torch.randn_like(a) * .005)
        zb = motion3(b + torch.randn_like(b) * .005)
        loss = ntxent(za, zb, temperature=.12)
        loss.backward(); optimizer.step()
    pose2.eval(); motion3.eval()
    with torch.no_grad():
        mine_pose = torch.tensor(np.stack([_project(c, joint_names=names).reshape(16, -1) for c in mine]))
        wild = pose2(mine_pose).numpy()
        units = motion3(torch.tensor(np.stack([_pose(c, joint_names=names).reshape(16, -1) for c in clips]))).numpy()
    labels, _ = cluster_latents(units, min(4, len(clips)), seed=seed)
    clusters = {str(i): [str(c["id"]) for c, label in zip(clips, labels) if label == i]
                for i in sorted(set(labels.tolist()))}
    text_features, _, _ = _text_features([item["text"] for item in mine])
    mined = build_rules(text_features, [item["text"] for item in mine], wild, units,
                        [str(c["id"]) for c in clips], labels)
    matches = [{"text": c["text"], "gesture_id": str(c["id"]),
                "cluster_id": int(label), "score": 1.0, "route": "aligned_bank_pair",
                "source": c.get("source", {})} for c, label in zip(clips, labels)]
    matches += [{"text": row["text"], "gesture_id": row["gesture_id"],
                "cluster_id": row["cluster_id"], "score": row["pose_match"], "route": "learned_pose_rule",
                "source": item.get("source", {})} for row, item in zip(mined, mine)]
    torch.save({"pose2": pose2.state_dict(), "motion3": motion3.state_dict(),
                "epochs": max(1, int(epochs)), "train_pairs": len(train)}, output / "pose-model.pt")
    return matches, float(loss.detach()), clusters, len(train)


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
        result.append({"phrase": rule["phrase"], "gesture_id": rule["gesture_id"],
                       "provenance": provenance})
    return result


def _fit_ridge(bank, output, epochs, seed):
    import torch
    from ridge_gesture.model import TextMotionModel, contrastive
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    train = bank["associations"] + bank["clips"]
    if len(train) < 2:
        raise ValueError("RIDGE fallback needs at least two paired association clips")
    text, vocab, idf = _text_features([x["text"] for x in train])
    if not vocab:
        raise ValueError("RIDGE fallback needs nonempty text vocabulary")
    motion = np.stack([_pose(x, joint_names=bank["joint_names"]).reshape(16, -1) for x in train])
    model = TextMotionModel(len(vocab), motion.shape[-1], latent=16, width=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    t, m = torch.tensor(text), torch.tensor(motion)
    for _ in range(max(1, int(epochs))):
        model.train(); optimizer.zero_grad()
        zt, zm = model(t, m)
        loss = contrastive(zt, zm, temperature=.12)
        loss.backward(); optimizer.step()
    model.eval()
    torch.save({"model": model.state_dict(), "vocab": vocab, "idf": idf.tolist(),
                "train_pairs": len(train), "epochs": max(1, int(epochs))}, output / "ridge-model.pt")
    return float(loss.detach())


def prepare(bank_path, output_dir, mode, *, epochs=60, seed=7, strong_rules_path=None, **_):
    """Prepare a local, mode-specific retrieval index from an ignored BEAT bank."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    source = Path(bank_path)
    bank = json.loads(source.read_text(encoding="utf-8"))
    ids, base = _check_bank(bank)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    origin_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    unit_changes = []
    if mode == "multilingual":
        bank, unit_changes = _refine_multilingual_units(bank)
        _check_bank(bank)
        source = output / "multilingual-bank.json"
        source.write_text(json.dumps(bank, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    selected = base if mode == "automatic" else ids
    try:
        portable_source = os.path.relpath(source.resolve(), output.resolve())
    except ValueError:
        portable_source = str(source.resolve())
    info = {"mode": mode, "bank_path": portable_source,
            "bank_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "origin_bank_sha256": origin_sha256,
            "base_ids": base, "playback_ids": selected, "fps": bank["fps"],
            "joint_order": bank["joint_names"], "axisSigns": bank.get("axisSigns", [-1, 1, 1]),
            "data_label": "local BEAT-derived public-data demo; fitted locally",
            "training_pairs": len(bank["associations"]), "seed_pairs": 3,
            "bank_count": len(selected), "seed": seed}
    if mode == "automatic":
        info["rules"] = [
            {"text": c["text"], "gesture_id": str(c["id"]), "score": 1.0,
             "route": "seed_rule", "source": c.get("source", {})}
            for c in bank["clips"] if str(c["id"]) in base
        ] + _weak_associations(bank, base)
        info["algorithm"] = "Automatic Text-to-Gesture weak pose associations over fixed three-gesture bank"
    elif mode in {"wild", "multilingual"}:
        info["rules"], info["training_loss"], info["clusters"], info["training_pairs"] = _fit_pose(
            bank, output, ids, epochs, seed, multilingual=mode == "multilingual")
        info["algorithm"] = "Wild model-based paired 2D/3D pose matching" + (
            " with Multilingual Gesture closure-selected units, projection augmentation and English translation"
            if mode == "multilingual" else "")
        if mode == "multilingual":
            info["unit_refinement"] = {"extractor": "multilingual_gesture.pipeline.extract_units",
                                       "min_seconds": 2., "max_seconds": 2.5,
                                       "closure_threshold": "unbounded; minimum closure candidate",
                                       "variance_threshold": 0., "clips": unit_changes,
                                       "trimmed_count": sum(c["trimmed"] for c in unit_changes),
                                       "blend_frames": 5}
    else:
        info["strong_rules"] = _strong_rules(strong_rules_path, ids)
        info["training_loss"] = _fit_ridge(bank, output, epochs, seed)
        info["training_pairs"] = len(bank["associations"]) + len(bank["clips"])
        info["algorithm"] = "RIDGE explicit strong rules plus fitted text-motion fallback"
    (output / "index.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"artifact_dir": str(output), "mode": mode, "rule_count": len(info.get("rules", info.get("strong_rules", []))),
            "metrics": {"training_pairs": info["training_pairs"], "training_loss": info.get("training_loss")}}


def _chunks(text, size=10):
    output = []
    for clause in re.split(r"[.!?;|]+", text):
        words = WORDS.findall(clause)
        output.extend(" ".join(words[i:i + size]) for i in range(0, len(words), size))
    return output


def _rule_route(text, rules):
    phrases = [r.get("text", r.get("phrase")) for r in rules]
    matrix, vocab, idf = _text_features(phrases)
    output = []
    for chunk in _chunks(text):
        exact = [i for i, p in enumerate(phrases) if p.casefold() == chunk.casefold()]
        scores = matrix @ _encode_text([chunk], vocab, idf)[0]
        j = exact[0] if exact else int(np.argmax(scores))
        output.append((chunk, rules[j], float(scores[j]), bool(exact)))
    return output


def _ridge_fallback(chunk, bank, artifact_dir):
    import torch
    from ridge_gesture.model import TextMotionModel
    checkpoint = torch.load(Path(artifact_dir) / "ridge-model.pt", map_location="cpu", weights_only=True)
    vocab, idf = checkpoint["vocab"], checkpoint["idf"]
    model = TextMotionModel(len(vocab), len(bank["joint_names"]) * 3, latent=16, width=32)
    model.load_state_dict(checkpoint["model"]); model.eval()
    clips = bank["clips"]
    query = torch.tensor(_encode_text([chunk], vocab, idf))
    motions = torch.tensor(np.stack([_pose(c, joint_names=bank["joint_names"]).reshape(16, -1) for c in clips]))
    with torch.no_grad():
        # Model forward requires equally sized paired batches.
        zt, zm = model(query.repeat(len(clips), 1), motions)
        scores = (zm @ zt[0]).numpy()
    j = int(np.argmax(scores))
    return str(clips[j]["id"]), float(scores[j])


def query(text, params, mode, artifact_dir):
    """Return playable source frames and a transparent retrieval trace."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if not str(text).strip():
        raise ValueError("Query text is empty")
    params = params or {}
    artifact = Path(artifact_dir)
    info = json.loads((artifact / "index.json").read_text(encoding="utf-8"))
    if info["mode"] != mode:
        raise ValueError(f"Prepared artifact is {info['mode']}, not {mode}")
    source = Path(info["bank_path"])
    if not source.is_absolute():
        source = artifact/source
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != info["bank_sha256"]:
        raise ValueError("Prepared BEAT bank is missing or has changed; run prepare again")
    bank = json.loads(source.read_text(encoding="utf-8"))
    ids, _ = _check_bank(bank)
    by_id = {str(c["id"]): c for c in bank["clips"]}
    original = text
    if mode == "multilingual":
        from multilingual_gesture.pipeline import require_english
        language = params.get("source_language", "en")
        text = require_english(text, language, params.get("translation_map", {}))
    if mode == "ridge":
        rules = info["strong_rules"]
        slots = []
        threshold = float(params.get("strong_rule_threshold", .65))
        if not 0 <= threshold <= 1:
            raise ValueError("strong_rule_threshold must be in [0,1]")
        for clause in re.split(r"[.!?;|]+", text):
            words = WORDS.findall(clause)
            start = 0
            while start < len(words):
                best = None
                for length in range(min(10, len(words) - start), 2, -1):
                    span = " ".join(words[start:start + length])
                    matches = _rule_route(span, rules) if rules else []
                    if matches:
                        _, rule, score, exact = matches[0]
                        phrase_length = len(WORDS.findall(rule['phrase']))
                        score *= min(length, phrase_length)/max(length, phrase_length)
                        if exact:
                            # Do not let a cached salient phrase consume adjacent
                            # generic text merely because its tokens are contained.
                            score = 1.0
                        if best is None or score > best[0]:
                            best = (score, length, span, rule)
                if best and best[0] >= threshold:
                    score, length, chunk, rule = best
                    gid, route = rule["gesture_id"], "strong_rule"
                    provenance = rule["provenance"]
                else:
                    length = min(10, len(words) - start)
                    # Preserve a forthcoming salient span rather than swallowing
                    # it into a generic chunk that starts with filler words.
                    for offset in range(1, length):
                        upcoming = [w.casefold() for w in words[start+offset:]]
                        if any(upcoming[:len(WORDS.findall(r["phrase"]))] ==
                               [w.casefold() for w in WORDS.findall(r["phrase"])] for r in rules):
                            length = offset
                            break
                    chunk = " ".join(words[start:start + length])
                    gid, score = _ridge_fallback(chunk, bank, artifact)
                    route, provenance = "trained_text_motion_fallback", {"kind": "local_training"}
                slots.append((chunk, gid, route, score, provenance))
                start += length
    else:
        rules = info["rules"]
        if mode == 'automatic':
            threshold = float(params.get('threshold', 0.92))
            if not 0 <= threshold <= 1:
                raise ValueError('Pose threshold must be between zero and one')
            rules = [r for r in rules if r['route'] == 'seed_rule' or r['score'] >= threshold]
        if not rules:
            raise ValueError("Prepared rule index is empty")
        slots = []
        for i, (chunk, rule, score, _) in enumerate(_rule_route(text, rules)):
            matched = rule["gesture_id"]
            gid = matched
            provenance = rule.get("source", {})
            if mode in {"wild", "multilingual"}:
                cluster = info["clusters"][str(rule["cluster_id"])]
                digest = int(hashlib.sha256(chunk.casefold().encode("utf-8")).hexdigest()[:8], 16)
                gid = cluster[(digest + i + int(params.get("seed", info["seed"]))) % len(cluster)]
                provenance = {"matched_gesture_id": matched, "cluster_id": rule["cluster_id"],
                              "association_source": provenance}
            slots.append((chunk, gid, rule["route"], score, provenance))
    out = []
    for chunk, gid, route, score, provenance in slots:
        if gid not in by_id or gid not in info["playback_ids"]:
            raise ValueError("Retrieval index references a gesture outside its playback bank")
        clip = by_id[gid]
        out.append({"gesture_id": gid, "text": chunk, "frames": clip["positions"],
                    "route": route, "confidence": round(float(score), 5),
                    "source": clip.get("source", {}), "rule_source": provenance,
                    "blend_frames": info.get("unit_refinement", {}).get("blend_frames", 0)})
    return {"fps": bank["fps"], "joint_order": bank["joint_names"],
            "axisSigns": bank.get("axisSigns", [-1, 1, 1]), "slots": out,
            "trace": {"input": original, "retrieval_text": text, "routes": [x["route"] for x in out],
                      "translation": "explicit_map" if original != text else None,
                      "unit_refinement": info.get("unit_refinement")},
            "algorithm": info["algorithm"], "data_label": info["data_label"],
            "metrics": {"rule_count": len(info.get("rules", info.get("strong_rules", []))),
                        "bank_count": info["bank_count"], "seed_pairs": info["seed_pairs"],
                        "extended_rules": max(0, len(info.get("rules", [])) - info["seed_pairs"]),
                        "refined_units": info.get("unit_refinement", {}).get("trimmed_count", 0),
                        "training_pairs": info["training_pairs"], "training_loss": info.get("training_loss")}}
