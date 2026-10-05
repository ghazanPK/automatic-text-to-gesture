from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .core import (DEFAULTS, TOKEN, Clip, import_manual_map, load_glove, load_manual_map, mine_clips,
                   read_rules, retrieve, write_manual_map, write_rules)


def _config(args) -> dict:
    """Paper defaults, overridden by an optional JSON config, overridden by explicit flags."""
    merged = dict(DEFAULTS)
    if getattr(args, "config", None):
        extra = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = set(extra) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}")
        merged.update(extra)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return merged


def _video_paths(args) -> list[Path]:
    paths = [Path(p) for group in (args.video or []) for p in group]
    if args.manifest:
        manifest = Path(args.manifest)
        text = manifest.read_text(encoding="utf-8")
        if manifest.suffix.lower() == ".json":
            data = json.loads(text)
            items = data.get("clips", data.get("videos")) if isinstance(data, dict) else data
            entries = [item["video"] if isinstance(item, dict) else item for item in items]
        else:
            entries = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        paths += [p if p.is_absolute() else manifest.parent / p for p in map(Path, entries)]
    if not paths:
        raise SystemExit("give at least one --video NPZ or a --manifest")
    return paths


def _load_clips(paths: list[Path]) -> list[Clip]:
    clips, seen = [], {}
    for path in paths:
        data = np.load(path, allow_pickle=False)
        source = str(data["clip_id"]) if "clip_id" in data.files else path.stem
        if source in seen:  # keep rule provenance unique across clips
            seen[source] += 1
            source = f"{source}#{seen[source]}"
        else:
            seen[source] = 0
        clips.append(Clip(source, data["pose"], json.loads(str(data["words_json"]))))
    return clips


def _load_bank(paths: list[str]) -> dict[str, np.ndarray]:
    bank: dict[str, np.ndarray] = {}
    for path in paths:
        data = np.load(path, allow_pickle=False)
        for key in data.files:
            if key in bank:
                raise SystemExit(f"gesture id {key!r} appears in more than one bank file")
            bank[key] = data[key]
    return bank


def _summary(report: dict) -> str:
    lines = [f"windows={report['windows']} gestures={report['gestures']} threshold={report['threshold']:.4f} rules={report.get('rules', '-')}"]
    if "at_threshold" in report:
        at = report["at_threshold"]
        lines.append(f"window pass rate={at['window_pass_rate']:.3f} median bank pass fraction={at['median_bank_pass_fraction']:.3f} "
                     f"mean passing gestures/window={at['mean_passing_gestures']:.2f}")
        lines.append("percentile thresholds: " + ", ".join(f"{k}={v:.4f}" for k, v in report["percentile_thresholds"].items()))
    if "warning" in report:
        lines.append("WARNING: " + report["warning"])
    return "\n".join(lines)


def _add_mining_args(p) -> None:
    p.add_argument("--video", action="append", nargs="+", help="video NPZ(s) with pose[F,J,2] and words_json; repeatable")
    p.add_argument("--manifest", help="text file with one video NPZ per line, or JSON list / {clips:[...]}")
    p.add_argument("--bank", action="append", required=True, help="NPZ: gesture id -> [F,J,2]; repeatable, ids must be unique")
    p.add_argument("--threshold", type=float, help=f"frame-cosine cut-off (default {DEFAULTS['threshold']})")
    p.add_argument("--threshold-percentile", type=float, help="use this percentile of all window-gesture scores as the threshold")
    p.add_argument("--phrase-words", dest="phrase_words", type=int, help=f"maximum phrase length (default {DEFAULTS['phrase_words']})")
    p.add_argument("--neck-joint", dest="neck_joint", type=int, help="normalisation joint index (default 1)")
    p.add_argument("--seed", type=int)
    p.add_argument("--config", help="JSON file overriding the paper defaults")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="attg")
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine", help="mine timed text-to-gesture rules over one or many clips (Algorithm 1)")
    _add_mining_args(mine)
    mine.add_argument("--output", required=True)
    mine.add_argument("--report", help="calibration report JSON (default: <output>.calibration.json)")
    cal = sub.add_parser("calibrate", help="threshold-calibration report without writing rules")
    _add_mining_args(cal)
    cal.add_argument("--output", required=True, help="report JSON")
    get = sub.add_parser("retrieve", help="retrieve a gesture sequence (Algorithm 2)")
    get.add_argument("--rules", help="auto rule map JSONL (needed for --map auto|hybrid)")
    get.add_argument("--glove", help="GloVe text vectors (needed for --map auto|hybrid)")
    get.add_argument("--manual", help="manual map JSON (attg-manual-map/1; see import-manual)")
    get.add_argument("--map", choices=("manual", "auto", "hybrid"), help="default: hybrid with --manual, else auto")
    get.add_argument("--text", required=True)
    get.add_argument("--audio-seconds", type=float)
    get.add_argument("--chunk-words", dest="chunk_words", type=int, help=f"words per slot (default {DEFAULTS['chunk_words']})")
    get.add_argument("--oov", choices=("idle", "skip", "error"), help="chunk without vocabulary or match (default idle)")
    get.add_argument("--idle-id", dest="idle_id", help="gesture id for idle slots (default 'idle')")
    get.add_argument("--min-similarity", type=float, help="optional GloVe similarity floor; below it the chunk goes idle")
    get.add_argument("--seed", type=int)
    get.add_argument("--config", help="JSON file overriding the paper defaults")
    get.add_argument("--output", required=True)
    imp = sub.add_parser("import-manual", help="convert an NVBG-like XML or CSV rule table to the manual map JSON")
    imp.add_argument("--input", required=True)
    imp.add_argument("--output", required=True)
    cfg = sub.add_parser("config", help="print the effective defaults")
    cfg.add_argument("--config")
    args = parser.parse_args(argv)

    if args.command in ("mine", "calibrate"):
        conf = _config(args)
        clips = _load_clips(_video_paths(args))
        threshold = None if args.threshold_percentile is not None else conf["threshold"]
        rules, report = mine_clips(clips, _load_bank(args.bank), threshold, conf["seed"], phrase_words=conf["phrase_words"],
                                   neck_joint=conf["neck_joint"], threshold_percentile=args.threshold_percentile)
        if args.command == "mine":
            write_rules(args.output, rules)
            report_path = Path(args.report or f"{args.output}.calibration.json")
        else:
            report_path = Path(args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(_summary(report), file=sys.stderr)
        print(json.dumps({"clips": len(clips), "rules": len(rules), "threshold": report["threshold"], "report": str(report_path)}))
    elif args.command == "retrieve":
        conf = _config(args)
        mode = args.map or ("hybrid" if args.manual else "auto")
        manual = load_manual_map(args.manual) if args.manual else None
        rules, vectors = [], {}
        if mode != "manual":
            if not args.rules or not args.glove:
                raise SystemExit(f"--map {mode} needs --rules and --glove")
            rules = read_rules(args.rules)
            words = {w for r in rules for w in TOKEN.findall(r.phrase.lower())}
            words.update(TOKEN.findall(args.text.lower()))
            vectors = load_glove(args.glove, words)
        result = retrieve(args.text, rules, vectors, args.audio_seconds, mode=mode, manual=manual, chunk_words=conf["chunk_words"], seed=conf["seed"],
                          oov=conf["oov"], idle_id=conf["idle_id"], min_similarity=args.min_similarity)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "import-manual":
        rules = import_manual_map(args.input)
        write_manual_map(args.output, rules)
        print(json.dumps({"rules": len(rules), "output": args.output}))
    else:
        print(json.dumps(_config(args), indent=2))


if __name__ == "__main__":
    main()
