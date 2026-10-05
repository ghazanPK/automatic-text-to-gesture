"""``multigesture`` command line: extract-units, train, mine, retrieve.

Every array input accepts several files and/or manifests (``.json`` list of
paths or ``{"path": ..., "speaker": ...}`` objects, or ``.txt`` with one path
per line), so multi-take and multi-speaker data can be combined without
hand-merging. Rows carry ``speakers`` when the source provides them, and
``--speaker`` filters to one speaker.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PRESETS = {
    # Paper Section III-C: AdamW, cosine annealing, lr 5e-4, wd 1e-4, 1000 epochs, batch 512.
    "paper": {"epochs": 1000, "batch_size": 512, "lr": 5e-4, "weight_decay": 1e-4},
    # Same optimiser and schedule sized for a few hundred demo pairs; converges on one BEAT take.
    "demo": {"epochs": 300, "batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4},
}


# ---------------------------------------------------------------------------
# Input helpers


def expand_inputs(paths):
    """Expand files and manifests into ``[(path, defaults_dict), ...]``."""
    out = []
    for raw in paths or []:
        p = Path(raw)
        if p.suffix.lower() == ".json" and p.is_file():
            items = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(items, dict):
                items = items.get("files", [])
            for item in items:
                entry = {"path": item} if isinstance(item, str) else dict(item)
                path = Path(entry.pop("path"))
                out.append((path if path.is_absolute() else p.parent / path, entry))
        elif p.suffix.lower() == ".txt" and p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    q = Path(line.strip()); out.append((q if q.is_absolute() else p.parent / q, {}))
        else:
            out.append((p, {}))
    if not out:
        raise ValueError("no input files supplied")
    return out


def _pad_frames(x, frames):
    if x.shape[1] == frames:
        return x
    return np.pad(x, ((0, 0), (0, frames - x.shape[1]), (0, 0)), mode="edge")


def load_rows(paths, keys, speaker=None):
    """Concatenate NPZ files with the listed array ``keys`` (frame axis padded
    to the longest file), plus ``lengths``, ``speakers`` and ``ids`` when present."""
    parts = []
    for path, meta in expand_inputs(paths):
        d = np.load(path, allow_pickle=False)
        missing = [k for k in keys if k not in d.files]
        if missing:
            raise ValueError(f"{path} lacks arrays {missing}")
        n = len(d[keys[0]])
        row = {k: d[k] for k in keys}
        frames_key = next((k for k in keys if d[k].ndim == 3), None)
        row["lengths"] = d["lengths"].astype(np.int64) if "lengths" in d.files else (
            np.full(n, d[frames_key].shape[1], np.int64) if frames_key else np.zeros(n, np.int64))
        if "speakers" in d.files:
            row["speakers"] = np.asarray([str(s) for s in d["speakers"]])
        else:
            row["speakers"] = np.asarray([str(meta.get("speaker", ""))] * n)
        if "ids" in d.files:
            row["ids"] = np.asarray([str(s) for s in d["ids"]])
        parts.append(row)
    frames = max((r[k].shape[1] for r in parts for k in keys if r[k].ndim == 3), default=0)
    merged = {}
    for k in parts[0]:
        if not all(k in r for r in parts):
            continue
        arrays = [_pad_frames(r[k], frames) if r[k].ndim == 3 else r[k] for r in parts]
        merged[k] = np.concatenate(arrays)
    if speaker is not None:
        keep = merged["speakers"] == str(speaker)
        if not keep.any():
            raise ValueError(f"no rows for speaker {speaker!r}")
        merged = {k: v[keep] for k, v in merged.items()}
    if "ids" in merged and len(set(merged["ids"].tolist())) != len(merged["ids"]):
        raise ValueError("unit ids collide across inputs; give each take a unique name")
    return merged


def encode_batches(encoder, x, lengths, batch=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(encoder(torch.from_numpy(np.ascontiguousarray(x[i:i + batch], np.float32)),
                               torch.from_numpy(np.asarray(lengths[i:i + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros((0, 10), np.float32)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands


def extract(a):
    from .pipeline import extract_unit_spans, pad_units
    clips, ids, takes, speakers, starts, ends, report = [], [], [], [], [], [], []
    variance = a.variance if a.variance == "auto" else float(a.variance)
    for path, meta in expand_inputs(a.motion):
        if path.suffix == ".npz":
            d = np.load(path); x = d[a.key if a.key in d.files else d.files[0]]
        else:
            x = np.load(path)
        x = np.asarray(x, np.float32).reshape(len(x), -1)
        take = str(meta.get("take", path.stem)); speaker = str(meta.get("speaker", a.speaker or ""))
        spans, info = extract_unit_spans(x, a.fps, a.min_seconds, a.max_seconds, variance, a.closure,
                                         normalize=not a.no_normalize)
        report.append({"take": take, "speaker": speaker, "frames": len(x), **info})
        for s, e in spans:
            clips.append(x[s:e]); ids.append(f"{take}:{s}-{e}"); takes.append(take)
            speakers.append(speaker); starts.append(s); ends.append(e)
    if not clips:
        raise ValueError("no units passed the variance and closure thresholds; try --variance auto")
    if len(set(ids)) != len(ids):
        raise ValueError("unit ids collide; give each motion file a unique name or manifest 'take'")
    padded, lengths = pad_units(clips, int(round(a.max_seconds * a.fps)))
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, motion3d=padded, lengths=lengths, ids=np.asarray(ids), takes=np.asarray(takes),
             speakers=np.asarray(speakers), starts=np.asarray(starts), ends=np.asarray(ends))
    print(json.dumps({"units": len(clips), "output": str(output), "takes": report}))


def _augment_batch(x2, lengths, ids, rng, conditions):
    from .pipeline import augment_2d
    seqs, lens = [], []
    for i in ids:
        seq, n, _ = augment_2d(x2[i], rng, lengths[i], conditions)
        seqs.append(seq); lens.append(n)
    return np.stack(seqs).astype(np.float32), np.asarray(lens)


def train(a):
    import torch
    from .model import GestureCLR, ntxent
    from .pipeline import ALL_CONDITIONS, normalize_batch
    preset = dict(PRESETS[a.preset])
    for key in preset:
        if getattr(a, key) is not None:
            preset[key] = getattr(a, key)
    epochs, batch, lr, wd = preset["epochs"], preset["batch_size"], preset["lr"], preset["weight_decay"]
    d = load_rows(a.pairs, ("pose2d", "motion3d"), a.speaker)
    x2, x3, lengths = d["pose2d"].astype("float32"), d["motion3d"].astype("float32"), d["lengths"]
    if len(x2) != len(x3) or len(x2) < 2:
        raise ValueError("training requires at least two aligned 2D/3D pairs")
    if epochs < 1 or batch < 2:
        raise ValueError("epochs must be positive and batch size must be at least two")
    conditions = tuple(c.strip() for c in a.augment.split(",") if c.strip())
    if not conditions or any(c not in ALL_CONDITIONS for c in conditions):
        raise ValueError(f"--augment must list conditions from {ALL_CONDITIONS}")
    if not a.no_normalize:
        x2, x3 = normalize_batch(x2, lengths), normalize_batch(x3, lengths)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(x2))
    n_val = int(round(len(x2) * a.val_fraction))
    n_val = n_val if n_val >= 2 and len(x2) - n_val >= 2 else 0
    val_ids, train_ids = order[:n_val], order[n_val:]
    val_rng = np.random.default_rng(a.seed + 1)
    val2 = _augment_batch(x2, lengths, val_ids, val_rng, conditions) if n_val else None
    model = GestureCLR(x2.shape[-1], x3.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    best, history = math.inf, []
    meta = {"d2": x2.shape[-1], "d3": x3.shape[-1], "normalize": not a.no_normalize, "preset": a.preset,
            "epochs": epochs, "batch_size": batch, "lr": lr, "weight_decay": wd, "temperature": a.temperature,
            "augment": list(conditions), "train_pairs": int(len(train_ids)), "val_pairs": int(n_val),
            "speaker": a.speaker}
    for epoch in range(epochs):
        model.train(); perm = rng.permutation(train_ids); total = 0.; seen = 0
        for st in range(0, len(perm), batch):
            ids = perm[st:st + batch]
            if len(ids) < 2:
                continue
            aug, aug_len = _augment_batch(x2, lengths, ids, rng, conditions)
            z2, z3 = model(torch.from_numpy(aug), torch.from_numpy(x3[ids]),
                           torch.from_numpy(aug_len), torch.from_numpy(lengths[ids]))
            loss = ntxent(z2, z3, a.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(ids); seen += len(ids)
        row = {"epoch": epoch + 1, "loss": total / max(seen, 1), "lr": opt.param_groups[0]["lr"]}
        sched.step()
        if n_val:
            model.eval(); vloss = 0.; vtop = 0
            with torch.no_grad():
                for st in range(0, n_val, batch):
                    sl = slice(st, st + batch); ids = val_ids[sl]
                    if len(ids) < 2:
                        continue
                    z2, z3 = model(torch.from_numpy(val2[0][sl]), torch.from_numpy(x3[ids]),
                                   torch.from_numpy(val2[1][sl]), torch.from_numpy(lengths[ids]))
                    vloss += float(ntxent(z2, z3, a.temperature)) * len(ids)
                    vtop += int(((z2 @ z3.T).argmax(1) == torch.arange(len(ids))).sum())
            row.update(val_loss=vloss / n_val, val_top1=vtop / n_val)
        score = row.get("val_loss", row["loss"])
        if score < best:
            best = score
            torch.save({"state": model.state_dict(), **meta, "best_epoch": epoch + 1, "metrics": row}, output)
            row["saved"] = True
        history.append(row)
        if a.log_every and ((epoch + 1) % a.log_every == 0 or epoch + 1 == epochs or epoch == 0):
            print(json.dumps(row), flush=True)
    if a.history:
        write_json(a.history, {"config": meta, "history": history})


def load_gestureclr(path):
    import torch
    from .model import GestureCLR
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = GestureCLR(ck["d2"], ck["d3"]); model.load_state_dict(ck["state"]); model.eval()
    return model, ck


def mine(a):
    from sentence_transformers import SentenceTransformer
    from .pipeline import bisect, normalize_batch
    w = load_rows(a.wild, ("pose2d", "texts"), a.wild_speaker)
    u = load_rows(a.units, ("motion3d", "ids"), a.unit_speaker)
    model, ck = load_gestureclr(a.checkpoint)
    w2, u3 = w["pose2d"].astype("float32"), u["motion3d"].astype("float32")
    if ck.get("normalize", False):
        w2, u3 = normalize_batch(w2, w["lengths"]), normalize_batch(u3, u["lengths"])
    wz = encode_batches(model.pose2d, w2, w["lengths"]); uz = encode_batches(model.motion3d, u3, u["lengths"])
    labels, centers = bisect(uz, a.clusters)
    ids = np.asarray([str(x) for x in u["ids"]]); texts = [str(x) for x in w["texts"]]
    te = SentenceTransformer(a.sbert).encode(texts, normalize_embeddings=True)
    sims = wz @ uz.T; nearest = sims.argmax(1)
    rules = [{"english_text": text, "text_embedding": emb.tolist(), "cluster_id": int(labels[j]),
              "source_gesture_id": ids[j], "pose_similarity": float(sims[i, j]),
              "wild_speaker": str(w["speakers"][i])}
             for i, (text, emb, j) in enumerate(zip(texts, te, nearest))]
    if a.min_pose_similarity is not None:
        rules = [r for r in rules if r["pose_similarity"] >= a.min_pose_similarity]
    if not rules:
        raise ValueError("no wild pose matched a unit above --min-pose-similarity")
    prefix = Path(a.output_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".rules.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rules), encoding="utf-8")
    np.savez(str(prefix) + ".clusters.npz", ids=ids, labels=labels, centroids=centers,
             lengths=u["lengths"], speakers=u["speakers"])
    print(json.dumps({"rules": len(rules), "units": len(ids), "clusters": int(labels.max() + 1),
                      "sbert": a.sbert}))


def load_rules(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_clusters(path):
    d = np.load(path)
    return {int(k): [str(x) for x in d["ids"][d["labels"] == k]] for k in np.unique(d["labels"])}


def get(a):
    from sentence_transformers import SentenceTransformer
    from .pipeline import multilingual_retrieve, schedule
    from .translate import translator_from_args
    rules = load_rules(a.rules); clusters = load_clusters(a.clusters)
    translator = translator_from_args(a)
    model = SentenceTransformer(a.sbert)
    out = multilingual_retrieve(a.text, a.source_language, translator, rules,
                                lambda x: model.encode(x, normalize_embeddings=True), clusters, a.seed,
                                a.min_similarity, a.idle_id, a.max_words, a.tts_language)
    word_times = json.loads(Path(a.word_timestamps).read_text(encoding="utf-8")) if a.word_timestamps else None
    if isinstance(word_times, dict):
        word_times = word_times.get("words", [])
    schedule(out["gestures"], a.audio_seconds, word_times)
    out.update(translator=a.translator, min_similarity=a.min_similarity, idle_id=a.idle_id)
    write_json(a.output, out)


def main(argv=None):
    from .translate import add_translator_arguments
    p = argparse.ArgumentParser(prog="multigesture"); s = p.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("extract-units", help="Algorithm 1 gesture-unit extraction over continuous motion")
    e.add_argument("--motion", nargs="+", required=True, help="[T,D] .npy/.npz files or manifests")
    e.add_argument("--output", required=True)
    e.add_argument("--variance", default="auto", help="'auto' (elbow of sorted clip variances) or a number")
    e.add_argument("--closure", type=float, default=float("inf"),
                   help="optional cap on start/end pose distance (body-scale units unless --no-normalize)")
    e.add_argument("--fps", type=int, default=15)
    e.add_argument("--min-seconds", type=float, default=2.0); e.add_argument("--max-seconds", type=float, default=3.0)
    e.add_argument("--no-normalize", action="store_true", help="use raw coordinate units for thresholds")
    e.add_argument("--speaker", help="speaker label for inputs without a manifest speaker")
    e.add_argument("--key", default="motion", help="array name inside .npz motion files")
    t = s.add_parser("train", help="train GestureCLR on paired 2D/3D windows")
    t.add_argument("--pairs", nargs="+", required=True); t.add_argument("--output", required=True)
    t.add_argument("--preset", choices=sorted(PRESETS), default="demo")
    t.add_argument("--epochs", type=int); t.add_argument("--batch-size", type=int)
    t.add_argument("--lr", type=float); t.add_argument("--weight-decay", type=float)
    t.add_argument("--temperature", type=float, default=.07)
    t.add_argument("--val-fraction", type=float, default=.1)
    t.add_argument("--augment", default="clean,noise,shift_mean,shift_zero",
                   help="comma list from clean,noise,shift_mean,shift_zero,noise_shift; one is drawn per sample")
    t.add_argument("--no-normalize", action="store_true"); t.add_argument("--speaker")
    t.add_argument("--seed", type=int, default=0); t.add_argument("--history")
    t.add_argument("--log-every", type=int, default=10)
    m = s.add_parser("mine", help="match wild 2D poses to units, cluster units, write the English rule map")
    m.add_argument("--wild", nargs="+", required=True); m.add_argument("--units", nargs="+", required=True)
    m.add_argument("--checkpoint", required=True); m.add_argument("--output-prefix", required=True)
    m.add_argument("--clusters", type=int, default=100); m.add_argument("--sbert", default="all-MiniLM-L6-v2")
    m.add_argument("--wild-speaker"); m.add_argument("--unit-speaker")
    m.add_argument("--min-pose-similarity", type=float)
    r = s.add_parser("retrieve", help="translate, chunk and retrieve gestures for text")
    r.add_argument("--rules", required=True); r.add_argument("--clusters", required=True)
    r.add_argument("--text", required=True); r.add_argument("--source-language", default="en")
    add_translator_arguments(r)
    r.add_argument("--tts-language", help="language to speak (default: the source language)")
    r.add_argument("--min-similarity", type=float, help="play --idle-id below this rule similarity")
    r.add_argument("--idle-id", default="idle")
    r.add_argument("--max-words", type=int, default=30, help="longer input is split into sentence chunks")
    r.add_argument("--audio-seconds", type=float, help="TTS duration for coarse pacing")
    r.add_argument("--word-timestamps", help="JSON list of {word,start,end} from the TTS engine")
    r.add_argument("--output", required=True); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--sbert", default="all-MiniLM-L6-v2")
    a = p.parse_args(argv)
    {"extract-units": extract, "train": train, "mine": mine, "retrieve": get}[a.cmd](a)


if __name__ == "__main__":
    main()
