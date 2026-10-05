"""``ridge-gesture`` command line.

annotate -> build-rules -> train (pretrain, then fine-tune) -> retrieve, plus
eval-gca. Inputs accept several files or manifests (``.json`` list / ``.txt``
list of paths); rows carry a ``speaker`` field, ``--speaker`` filters to one
speaker and ``--per-speaker`` writes one output per speaker.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_SBERT = "all-MiniLM-L6-v2"
PRESETS = {
    # Paper Section 4.3: pretraining batch 1000, variable LR, early stopping on validation loss.
    "pretrain": {"batch_size": 1000, "lr": 1e-3, "epochs": 300, "patience": 20, "schedule": "plateau"},
    # Paper Section 4.3: fine-tuning batch 64 with the same early stopping.
    "finetune": {"batch_size": 64, "lr": 1e-4, "epochs": 300, "patience": 20, "schedule": "plateau"},
}


def expand(paths):
    out = []
    for raw in paths or []:
        p = Path(raw)
        if p.suffix.lower() == ".json" and p.is_file() and isinstance(json.loads(p.read_text(encoding="utf-8")), list):
            for item in json.loads(p.read_text(encoding="utf-8")):
                q = Path(item if isinstance(item, str) else item["path"])
                out.append(q if q.is_absolute() else p.parent / q)
        elif p.suffix.lower() == ".txt" and p.is_file():
            out += [Path(x.strip()) if Path(x.strip()).is_absolute() else p.parent / x.strip()
                    for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
        else:
            out.append(p)
    return out


def lines(paths):
    rows = []
    for path in expand(paths if isinstance(paths, (list, tuple)) else [paths]):
        rows += [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def dump(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def unique_records(records):
    ids = [r["record_id"] for r in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate record_id values {dup[:5]}; give each take a unique id")
    return {r["record_id"]: r for r in records}


def load_sbert(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


# ---------------------------------------------------------------------------
# annotate / build-rules


def annotate(a):
    from .annotate import annotate_record, record_from_textgrid
    records = lines(a.transcripts) if a.transcripts else []
    for path in expand(a.textgrid):
        records.append(record_from_textgrid(path, a.fps, speaker=a.speaker))
    if not records:
        raise ValueError("supply --transcripts JSONL and/or --textgrid files")
    unique_records(records)
    if a.records_output:
        dump(a.records_output, records)
    if a.speaker_filter:
        records = [r for r in records if r.get("speaker") == a.speaker_filter]
    reviewed = json.loads(Path(a.external_annotations).read_text(encoding="utf-8")) if a.external_annotations else None
    llm = bool(a.llm_endpoint or a.llm_command)
    if llm and a.llm_endpoint and not a.model:
        raise ValueError("--llm-endpoint needs --model")
    rows = []
    for r in records:
        if reviewed is not None and r["record_id"] in reviewed:
            mode, items = "reviewed", reviewed[r["record_id"]]
        else:
            mode, items = ("llm" if llm else "heuristic"), None
        row = annotate_record(r, mode, items, a.llm_endpoint, a.model, a.llm_command, a.api_key_env,
                              a.min_words, a.max_words, a.limit)
        rows.append(row)
        print(json.dumps({"record_id": r["record_id"], "annotator": mode, "phrases": len(row["phrases"]),
                          "rejected": len(row["provenance"]["rejected"])}), file=sys.stderr)
    dump(a.output, rows)


def _rule_rows(records, annotations, speaker=None):
    rows = []
    for item in annotations:
        record = records.get(item["record_id"])
        if record is None:
            raise ValueError(f"annotation refers to unknown record {item['record_id']!r}")
        spk = item.get("speaker", record.get("speaker", ""))
        if speaker is not None and spk != speaker:
            continue
        seen = {}
        for phrase in item["phrases"]:
            if isinstance(phrase, dict):
                text, k = phrase["phrase"], int(phrase.get("occurrence", 0))
            else:  # legacy plain strings: repeated strings bind to successive occurrences
                from .pipeline import WORD
                text = phrase; key = " ".join(WORD.findall(text)).casefold(); k = seen.get(key, 0); seen[key] = k + 1
            from .pipeline import align_phrase
            start, end = align_phrase(text, record["words"], k)
            rows.append({"phrase": text, "gesture_id": f'{item["record_id"]}:{start}-{end}',
                         "record_id": item["record_id"], "speaker": spk, "start_frame": start, "end_frame": end,
                         "occurrence": k, "annotator": item.get("annotator", "unknown")})
    return rows


def build_rules(a):
    records = unique_records(lines(a.records)); annotations = lines(a.annotations)
    sbert = a.sbert or DEFAULT_SBERT
    model = load_sbert(sbert)
    if a.per_speaker:
        speakers = sorted({r.get("speaker", "") for r in records.values()})
        out_dir = Path(a.output); out_dir.mkdir(parents=True, exist_ok=True)
        targets = [(s, out_dir / f"{s or 'unknown'}.rules.jsonl") for s in speakers]
    else:
        targets = [(a.speaker, Path(a.output))]
    for speaker, path in targets:
        rows = _rule_rows(records, annotations, speaker)
        if not rows:
            print(json.dumps({"speaker": speaker, "rules": 0, "skipped": True})); continue
        emb = model.encode([r["phrase"] for r in rows], normalize_embeddings=True)
        for r, e in zip(rows, emb):
            r["embedding"] = e.tolist(); r["sbert"] = sbert
        dump(path, rows)
        print(json.dumps({"speaker": speaker, "rules": len(rows), "output": str(path)}))


# ---------------------------------------------------------------------------
# train


def load_pairs(paths, speaker=None):
    parts = [np.load(p, allow_pickle=False) for p in expand(paths)]
    if not parts:
        raise ValueError("no --pairs files")
    frames = max(d["motion"].shape[1] for d in parts)
    data = {"text_embeddings": [], "motion": [], "ids": [], "speakers": [], "lengths": [], "texts": []}
    sberts = set()
    for d in parts:
        n = len(d["motion"])
        motion = d["motion"]
        if motion.shape[1] < frames:
            motion = np.pad(motion, ((0, 0), (0, frames - motion.shape[1]), (0, 0)), mode="edge")
        data["motion"].append(motion); data["text_embeddings"].append(d["text_embeddings"])
        data["ids"].append(np.asarray([str(x) for x in d["ids"]]))
        data["speakers"].append(np.asarray([str(x) for x in d["speakers"]]) if "speakers" in d.files else np.asarray([""] * n))
        data["lengths"].append(d["lengths"].astype(np.int64) if "lengths" in d.files else np.full(n, d["motion"].shape[1]))
        data["texts"].append(np.asarray([str(x) for x in d["texts"]]) if "texts" in d.files else np.asarray([""] * n))
        if "sbert" in d.files:
            sberts.add(str(d["sbert"]))
    data = {k: np.concatenate(v) for k, v in data.items()}
    if len(sberts) > 1:
        raise ValueError(f"pairs mix Sentence-BERT models {sorted(sberts)}")
    if len(set(data["ids"].tolist())) != len(data["ids"]):
        raise ValueError("pair ids collide across --pairs files")
    if speaker is not None:
        keep = data["speakers"] == speaker
        if not keep.any():
            raise ValueError(f"no pairs for speaker {speaker!r}")
        data = {k: v[keep] for k, v in data.items()}
    return data, (sberts.pop() if sberts else None)


def train_one(a, data, sbert_id, output):
    import torch
    from .model import RidgeModel, contrastive, load_gesture_init
    cfg = dict(PRESETS[a.preset])
    for key in ("batch_size", "lr", "epochs", "patience", "schedule"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    te, mo, ids, lengths = data["text_embeddings"].astype("float32"), data["motion"].astype("float32"), data["ids"], data["lengths"]
    if len(te) != len(mo) or len(ids) != len(te) or len(te) < 2:
        raise ValueError("training requires at least two aligned text/motion pairs and IDs")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 2:
        raise ValueError("epochs must be positive and batch size must be at least two")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    st = None
    if a.finetune_text:
        if not (data["texts"] != "").all():
            raise ValueError("--finetune-text needs a 'texts' array in every pairs file")
        st = load_sbert(a.sbert or sbert_id or DEFAULT_SBERT); st.train()
    if st is not None:
        dim = getattr(st, "get_embedding_dimension", None) or st.get_sentence_embedding_dimension
        text_dim = int(dim())
    else:
        text_dim = te.shape[-1]
    model = RidgeModel(text_dim, mo.shape[-1], a.latent)
    init_info = {}
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        if ck.get("architecture") != RidgeModel.ARCHITECTURE or ck["text_dim"] != text_dim or ck["motion_dim"] != mo.shape[-1]:
            raise ValueError("--init checkpoint has a different architecture or input size")
        model.load_state_dict(ck["state"]); init_info["init"] = str(a.init)
    if a.gesture_init:
        load_gesture_init(model, a.gesture_init); init_info["gesture_init"] = str(a.gesture_init)
    order = rng.permutation(len(te))
    n_val = int(round(len(te) * a.val_fraction))
    n_val = n_val if n_val >= 2 and len(te) - n_val >= 2 else 0
    val_ids, train_ids = order[:n_val], order[n_val:]
    groups = [{"params": model.parameters(), "lr": cfg["lr"]}]
    if st is not None:
        groups.append({"params": st.parameters(), "lr": a.text_lr})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    if cfg["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    elif cfg["schedule"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=max(cfg["patience"] // 4, 1))
    else:
        sched = None

    def text_batch(ix):
        if st is None:
            return torch.from_numpy(te[ix])
        features = st.tokenize([str(t) for t in data["texts"][ix]])
        return torch.nn.functional.normalize(st(features)["sentence_embedding"], dim=-1)

    def batch_loss(ix):
        zt, zm = model(text_batch(ix), torch.from_numpy(mo[ix]), torch.from_numpy(lengths[ix]))
        return contrastive(zt, zm, a.tau), zt, zm

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    best, stale, history, best_epoch = math.inf, 0, [], 0
    for epoch in range(cfg["epochs"]):
        model.train(); perm = rng.permutation(train_ids); total = seen = 0
        if st is not None:
            st.train()
        for s in range(0, len(perm), cfg["batch_size"]):
            ix = perm[s:s + cfg["batch_size"]]
            if len(ix) < 2:
                continue
            loss, _, _ = batch_loss(ix); opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(ix); seen += len(ix)
        row = {"epoch": epoch + 1, "loss": total / max(seen, 1), "lr": opt.param_groups[0]["lr"]}
        if n_val:
            model.eval(); vloss = 0.; vtop = 0
            if st is not None:
                st.eval()
            with torch.no_grad():
                for s in range(0, n_val, cfg["batch_size"]):
                    ix = val_ids[s:s + cfg["batch_size"]]
                    if len(ix) < 2:
                        continue
                    loss, zt, zm = batch_loss(ix); vloss += float(loss) * len(ix)
                    vtop += int(((zt @ zm.T).argmax(1) == torch.arange(len(ix))).sum())
            row.update(val_loss=vloss / n_val, val_top1=vtop / n_val)
        score = row.get("val_loss", row["loss"])
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(score)
        elif sched is not None:
            sched.step()
        if score < best - a.min_delta:
            best, stale, best_epoch = score, 0, epoch + 1
            torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, str(output) + ".tmp")
            if st is not None:
                st.save(str(output) + ".sbert")
        else:
            stale += 1
        history.append(row)
        if a.log_every and ((epoch + 1) % a.log_every == 0 or epoch == 0):
            print(json.dumps(row), flush=True)
        if cfg["patience"] and stale >= cfg["patience"]:
            print(json.dumps({"early_stop": epoch + 1, "best_epoch": best_epoch}), flush=True)
            break
    tmp = Path(str(output) + ".tmp")
    model.load_state_dict(torch.load(tmp, weights_only=True)); tmp.unlink()
    model.eval()
    if st is not None:
        st = load_sbert(str(output) + ".sbert")
    with torch.no_grad():
        latents = torch.cat([model.encode_motion(torch.from_numpy(mo[s:s + 512]), torch.from_numpy(lengths[s:s + 512]))
                             for s in range(0, len(mo), 512)])
    text_id = sbert_id or a.sbert or DEFAULT_SBERT
    torch.save({"state": model.state_dict(), "architecture": RidgeModel.ARCHITECTURE, "text_dim": text_dim,
                "motion_dim": mo.shape[-1], "latent": a.latent, "ids": ids.tolist(), "speakers": data["speakers"].tolist(),
                "motion_latents": latents, "sbert": text_id,
                "text_encoder": "finetuned" if st is not None else "frozen",
                "text_encoder_path": str(output) + ".sbert" if st is not None else text_id,
                "tau": a.tau, "preset": a.preset, "config": cfg, "best_epoch": best_epoch, "best_score": best,
                "train_pairs": int(len(train_ids)), "val_pairs": int(n_val), **init_info}, output)
    if a.history:
        hist = Path(a.history) if not a.per_speaker else Path(a.history).with_name(output.stem + ".history.json")
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps({"config": cfg, "history": history}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "best_epoch": best_epoch, "best_score": best, "pairs": len(te)}))


def train(a):
    data, sbert_id = load_pairs(a.pairs, a.speaker)
    if a.sbert and sbert_id and a.sbert != sbert_id and not a.finetune_text:
        raise ValueError(f"pairs were embedded with {sbert_id!r}, not {a.sbert!r}")
    if not a.per_speaker:
        return train_one(a, data, sbert_id, a.output)
    out_dir = Path(a.output); out_dir.mkdir(parents=True, exist_ok=True)
    for speaker in sorted(set(data["speakers"].tolist())):
        keep = data["speakers"] == speaker
        if keep.sum() < 2:
            print(json.dumps({"speaker": speaker, "skipped": "fewer than two pairs"})); continue
        train_one(a, {k: v[keep] for k, v in data.items()}, sbert_id, out_dir / f"{speaker or 'unknown'}.pt")


# ---------------------------------------------------------------------------
# retrieve / eval-gca


def fallback_encoder(model, ck, sbert_name=None):
    import torch
    from .model import encode_text
    encoder = load_sbert(ck.get("text_encoder_path") if ck.get("text_encoder") == "finetuned"
                         else (sbert_name or ck.get("sbert", DEFAULT_SBERT)))

    def encode(text):
        with torch.no_grad():
            z = encode_text(model, torch.from_numpy(encoder.encode([text], normalize_embeddings=True).astype("float32"))).numpy()[0]
        return z / max(np.linalg.norm(z), 1e-8)
    return encode


def retrieve(a):
    from .model import load_checkpoint
    from .pipeline import hybrid_retrieve
    model, ck = load_checkpoint(a.checkpoint); rules = lines(a.rules)
    rule_sbert = a.sbert or rules[0].get("sbert") or ck.get("sbert") or DEFAULT_SBERT
    sbert = load_sbert(rule_sbert)
    latent = ck["motion_latents"].cpu().numpy().astype("float32"); ids = [str(x) for x in ck["ids"]]
    enc = fallback_encoder(model, ck, a.sbert)
    result = hybrid_retrieve(a.text, rules, lambda x: sbert.encode(x, normalize_embeddings=True), a.threshold,
                             latent, ids, enc)
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def eval_gca(a):
    from .pipeline import GCA
    ref = np.load(a.reference); cand = np.load(a.candidate)
    metric = GCA(a.text_clusters, a.gesture_clusters).fit(ref["text_embeddings"], ref["motion_embeddings"])
    print(json.dumps({"gca": metric.score(cand["text_embeddings"], cand["motion_embeddings"]),
                      "fit_samples": len(ref["text_embeddings"]), "evaluation_samples": len(cand["text_embeddings"])}))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ridge-gesture"); s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("annotate", help="extract 3-10 word gesture phrases (LLM, reviewed JSON or heuristic)")
    a.add_argument("--transcripts", nargs="+", help="transcript JSONL files {record_id, speaker, text, words}")
    a.add_argument("--textgrid", nargs="+", help="BEAT/Praat TextGrid files (record_id = file stem)")
    a.add_argument("--fps", type=int, default=15); a.add_argument("--speaker", help="speaker for --textgrid records")
    a.add_argument("--speaker-filter", help="annotate only this speaker's records")
    a.add_argument("--records-output", help="write the transcript records (useful with --textgrid)")
    a.add_argument("--output", required=True)
    a.add_argument("--external-annotations", help="reviewed JSON {record_id: [phrases]}")
    a.add_argument("--llm-endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:1234/v1")
    a.add_argument("--model", help="model name for --llm-endpoint")
    a.add_argument("--llm-command", help="local LLM command; prompt on stdin, reply on stdout")
    a.add_argument("--api-key-env", default="OPENAI_API_KEY")
    a.add_argument("--min-words", type=int, default=3); a.add_argument("--max-words", type=int, default=10)
    a.add_argument("--limit", type=int, default=5, help="heuristic phrases per record")
    b = s.add_parser("build-rules"); b.add_argument("--records", nargs="+", required=True)
    b.add_argument("--annotations", nargs="+", required=True); b.add_argument("--output", required=True)
    b.add_argument("--sbert", help=f"Sentence-BERT name or local directory (default {DEFAULT_SBERT})")
    b.add_argument("--speaker"); b.add_argument("--per-speaker", action="store_true", help="--output is a directory")
    t = s.add_parser("train", help="contrastive text-motion training (pretrain or fine-tune)")
    t.add_argument("--pairs", nargs="+", required=True); t.add_argument("--output", required=True)
    t.add_argument("--preset", choices=sorted(PRESETS), default="finetune")
    t.add_argument("--epochs", type=int); t.add_argument("--batch-size", type=int); t.add_argument("--lr", type=float)
    t.add_argument("--patience", type=int, help="early-stopping patience in epochs (0 disables)")
    t.add_argument("--schedule", choices=("plateau", "cosine", "none"))
    t.add_argument("--min-delta", type=float, default=0.0, help="minimum validation-loss improvement that resets patience")
    t.add_argument("--weight-decay", type=float, default=1e-4); t.add_argument("--tau", type=float, default=.07)
    t.add_argument("--latent", type=int, default=10); t.add_argument("--val-fraction", type=float, default=.1)
    t.add_argument("--init", help="RIDGE checkpoint to continue from (stage 2 after pretraining)")
    t.add_argument("--gesture-init", help="GestureCLR checkpoint whose motion3d encoder initialises the motion branch")
    t.add_argument("--finetune-text", action="store_true", help="also fine-tune Sentence-BERT (pairs need 'texts')")
    t.add_argument("--text-lr", type=float, default=2e-5)
    t.add_argument("--sbert", help="Sentence-BERT used for --finetune-text (default: the pairs' model)")
    t.add_argument("--speaker"); t.add_argument("--per-speaker", action="store_true", help="--output is a directory")
    t.add_argument("--seed", type=int, default=0); t.add_argument("--history"); t.add_argument("--log-every", type=int, default=10)
    r = s.add_parser("retrieve"); r.add_argument("--rules", nargs="+", required=True); r.add_argument("--checkpoint", required=True)
    r.add_argument("--text", required=True); r.add_argument("--threshold", type=float, default=.72)
    r.add_argument("--output", required=True)
    r.add_argument("--sbert", help="override the Sentence-BERT recorded in the rules/checkpoint")
    g = s.add_parser("eval-gca"); g.add_argument("--reference", required=True); g.add_argument("--candidate", required=True)
    g.add_argument("--text-clusters", type=int, default=100); g.add_argument("--gesture-clusters", type=int, default=20)
    x = p.parse_args(argv)
    {"annotate": annotate, "build-rules": build_rules, "train": train, "retrieve": retrieve, "eval-gca": eval_gca}[x.cmd](x)


if __name__ == "__main__":
    main()
