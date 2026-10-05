"""Prepare the Automatic Text-to-Gesture paper method on public BEAT for the browser demo.

Launcher hook (see scripts/beat_demo/BEAT_INGEST.md): progress goes to stderr
and the last stdout line is ``{"ready": ..., "server_args": [...]}``.

Disjoint BEAT speakers take the paper's two roles:

* ``library`` speakers: 3 s neck-centred windows projected frontally to 2D form
  the gesture bank (the paper's predefined animation library);
* ``video`` speakers: whole takes projected through an explicit camera (yaw 20,
  pitch 5) and corrupted like OpenPose tracks, with their word timings, stand
  in for public video. ``attg mine`` slides the bank over every clip
  (Algorithm 1) and records timed <=5-word phrases; ``attg calibrate`` reports
  the score distribution. The last video take is held out of mining and probes
  whether GloVe retrieval of its phrases picks the gesture its pose matches.

Runtime retrieval sums GloVe vectors (Algorithm 2), so a GloVe file is needed;
``scripts/build_glove_subset.py`` makes a small one from glove.6B.300d.txt.
Results are cached under ``outputs/paper-method/<settings hash>/``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_method_common as pm  # noqa: E402

PACKAGE = "automatic_text_to_gesture"
DEFAULT_SPEAKERS = "1,2,3,4,5,6"
DEFAULT_TAKES = 3
GLOVE_CANDIDATES = ("data/glove/glove.6B.300d.subset.txt", "data/glove/glove.6B.300d.txt")
GLOVE_STEPS = (
    "download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ and extract glove.6B.300d.txt into data/glove/",
    "optional, smaller and faster: python scripts/build_glove_subset.py --glove data/glove/glove.6B.300d.txt "
    "--vocab-from /path/to/processed/beat --output data/glove/glove.6B.300d.subset.txt",
    f"or pass --glove <file> (env {pm.ENV_GLOVE})",
)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pm.add_source_args(p, DEFAULT_SPEAKERS, DEFAULT_TAKES, "library, video")
    p.add_argument("--glove", type=Path, help=f"GloVe text vectors (default {GLOVE_CANDIDATES[0]} or {GLOVE_CANDIDATES[1]})")
    p.add_argument("--preset", choices=("demo", "paper"), default="demo",
                   help="demo: threshold from --threshold-percentile, rounded to 0.01; paper: the paper's 0.92")
    p.add_argument("--threshold", type=float, help="explicit frame-cosine threshold (overrides the preset)")
    p.add_argument("--threshold-percentile", type=float, default=95.0,
                   help="demo preset: percentile of all window-gesture scores used as the threshold (default 95)")
    p.add_argument("--heldout-takes", type=int, default=1, help="video takes kept out of mining as probes (default 1)")
    p.add_argument("--yaw", type=float, default=20.0); p.add_argument("--pitch", type=float, default=5.0)
    p.add_argument("--noise", type=float, default=0.02); p.add_argument("--jitter", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--min-similarity", type=float, help="optional GloVe similarity floor; below it a chunk idles")
    return p.parse_args(argv)


def find_glove(value=None):
    if value:
        return (Path(value).resolve(), None) if Path(value).is_file() else (None, f"GloVe file {value} does not exist")
    if os.environ.get(pm.ENV_GLOVE) and Path(os.environ[pm.ENV_GLOVE]).is_file():
        return Path(os.environ[pm.ENV_GLOVE]).resolve(), None
    for candidate in GLOVE_CANDIDATES:
        if (pm.ROOT / candidate).is_file():
            return (pm.ROOT / candidate).resolve(), None
    return None, "GloVe vectors were not found (the paper's summed-GloVe retrieval needs them)"


def _settings(args, source, kind, descriptors, glove):
    stat = glove.stat()
    return {"repo": PACKAGE, "source": str(source), "kind": kind, "selection": pm.selection(args),
            "takes": [d["take"] for d in descriptors], "role": args.role, "role_unit": args.role_unit,
            "seed": args.seed, "max_frames": args.max_frames, "preset": args.preset, "threshold": args.threshold,
            "percentile": args.threshold_percentile, "heldout": args.heldout_takes,
            "glove": [str(glove), stat.st_size, int(stat.st_mtime)], "camera": [args.yaw, args.pitch],
            "corruption": [args.noise, args.jitter, args.dropout], "min_similarity": args.min_similarity}


def split_video(data):
    """Write one clip NPZ per video take (Algorithm 1's outer loop) with take-local, cleaned word timings."""
    video = np.load(data / "video.npz")
    segments = json.loads(str(video["segments_json"]))
    words = json.loads(str(video["words_json"]))
    clips_dir = data / "clips"; clips_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for segment in segments:
        s, e = segment["start_frame"], segment["end_frame"]
        local = [(w["word"], w["start_frame"] - s, w["end_frame"] - s) for w in words if s <= w["start_frame"] < e]
        local = [{"word": w, "start_frame": a, "end_frame": b} for w, a, b in pm.clean_words(local, e - s)]
        path = clips_dir / f"{segment['take']}.npz"
        np.savez(path, pose=video["pose"][s:e], confidence=video["confidence"][s:e],
                 words_json=np.asarray(json.dumps(local)), clip_id=np.asarray(segment["take"]))
        paths.append(path)
    return paths, segments


def heldout_probe(clips, bank_path, rules, glove, threshold, seed):
    """Unseen video windows: does GloVe retrieval of the window's phrase pick the gesture its pose matches best?"""
    from automatic_text_to_gesture.core import TOKEN, GestureBank, aligned_phrase, load_glove, retrieve, window_scores
    bank = dict(np.load(bank_path))
    gb = GestureBank(bank)
    rows = []
    for clip in clips:
        starts, scores = window_scores(clip["pose"], gb)
        for start, row in zip(starts, scores):
            phrase = aligned_phrase(clip["words"], start, start + gb.stride)
            if phrase:
                rows.append((phrase, gb.ids[int(np.argmax(row))], float(row.max())))
    if not rows or not rules:
        return {}, []
    words = {w for r in rules for w in TOKEN.findall(r.phrase.lower())} | {w for p, _, _ in rows for w in TOKEN.findall(p.lower())}
    vectors = load_glove(glove, words)
    hits = scored = 0
    for phrase, pose_best, _ in rows:
        slot = retrieve(phrase, rules, vectors, chunk_words=5, seed=seed)[0]
        if slot["map"] == "auto":
            scored += 1
            hits += slot["gesture_id"] == pose_best
    gestures = len({r.gesture_id for r in rules})
    metrics = {"heldout_top1": round(hits / scored, 4) if scored else None,
               "heldout_chance": round(1 / gestures, 4), "heldout_windows": scored,
               "heldout": "held-out video take: GloVe-retrieved gesture == best pose-matched bank gesture (never mined)"}
    return metrics, [p for p, _, _ in rows if len(p.split()) >= 4][:2]


def prepare(args):
    source, kind = pm.find_source(args.processed, args.beat_root)
    if source is None:
        return pm.not_ready("no local BEAT source found", pm.SOURCE_STEPS)
    glove, why = find_glove(args.glove)
    if glove is None:
        return pm.not_ready(why, GLOVE_STEPS)
    beat = pm.ingest()
    select = pm.selection(args)
    descriptors = beat.list_takes(source, kind=kind, **select)
    if not descriptors:
        return pm.not_ready(f"no BEAT takes in {source} match the selection", pm.SOURCE_STEPS)
    code = [pm.ROOT / "src" / PACKAGE, Path(__file__), Path(pm.__file__), pm.SCRIPTS / "beat_demo" / "beat_ingest.py"]
    folder = Path(args.output_root) / pm.cache_key(_settings(args, source, kind, descriptors, glove), code)
    if not args.force and pm.cached(folder):
        pm.progress(f"cached result {pm.portable(folder)}")
        return pm.ready(folder, pm.cached(folder), cached_result=True)
    if folder.exists():
        shutil.rmtree(folder)
    data = folder / "data"
    timings, started = {}, time.perf_counter()
    pm.use_repository_package(PACKAGE)
    from automatic_text_to_gesture import cli
    from automatic_text_to_gesture.core import read_rules

    pm.progress(f"exporting {len(descriptors)} takes from {source} ({kind}) with disjoint speaker roles")
    t0 = time.perf_counter()
    summary = beat.export_automatic(source, data, fps=pm.FPS, unit_seconds=pm.UNIT_SECONDS, units="cm", seed=args.seed,
                                    camera={"yaw": args.yaw, "pitch": args.pitch, "focal": 1.0, "distance": 3.0},
                                    corruption={"noise": args.noise, "jitter": args.jitter, "dropout": args.dropout},
                                    kind=kind, role_spec=pm.role_spec(args), role_unit=args.role_unit,
                                    max_frames=args.max_frames, **select)
    order = json.loads((data / "joint_order.json").read_text(encoding="utf-8"))
    assignment, bank_ids = order["roles"], set(order["bank_ids"])
    roles = pm.descriptor_roles(assignment)
    paths, segments = split_video(data)
    held = max(0, min(args.heldout_takes, len(paths) - 1))
    mine_paths, probe_paths = paths[:len(paths) - held], paths[len(paths) - held:]
    library_rows, motion = [], []
    for record in pm.load_takes([d for d in descriptors if roles.get(d["take"]) == "library"], args.max_frames):
        for w in pm.take_windows([record]):
            if w["id"] in bank_ids:
                motion.append(w["positions"].reshape(len(w["positions"]), -1))
                library_rows.append({"id": w["id"], "text": w["text"], "speaker": record["speaker"], "take": record["take"],
                                     "start_frame": w["start"], "end_frame": w["end"], "role": "library",
                                     **pm.source_info(record["source"])})
    np.savez(data / "library3d.npz", ids=np.asarray([r["id"] for r in library_rows]), motion=np.stack(motion))
    (folder / "library.json").write_text(json.dumps(library_rows, ensure_ascii=False), encoding="utf-8")
    timings["export_seconds"] = round(time.perf_counter() - t0, 2)

    video_args = ["--video", *mine_paths, "--bank", data / "bank.npz", "--seed", args.seed]
    calibration = None
    if args.threshold is not None or args.preset == "paper":
        threshold = args.threshold if args.threshold is not None else 0.92
        rule = "explicit --threshold" if args.threshold is not None else "paper threshold 0.92"
    else:
        report_path = folder / f"calibration-p{args.threshold_percentile:g}.json"
        calibration, timings["calibrate_seconds"] = pm.run_cli(cli.main, ["calibrate", *video_args, "--threshold-percentile",
                                                                          args.threshold_percentile, "--output", report_path],
                                                               "threshold calibration")
        threshold = round(float(calibration["threshold"]), 2)
        rule = (f"{args.threshold_percentile:g}th percentile of window-gesture frame cosines over the mined video clips "
                f"(attg calibrate), rounded to 0.01; the paper used 0.92")
    rules_path = folder / "rules.jsonl"
    mined, timings["mine_seconds"] = pm.run_cli(cli.main, ["mine", *video_args, "--threshold", threshold, "--output", rules_path],
                                                "Algorithm 1 rule mining")
    report = json.loads(Path(f"{rules_path}.calibration.json").read_text(encoding="utf-8"))
    rules = read_rules(rules_path)
    t0 = time.perf_counter()
    probes = []
    for path in probe_paths:
        d = np.load(path)
        probes.append({"pose": d["pose"], "words": json.loads(str(d["words_json"]))})
    heldout, probe_phrases = heldout_probe(probes, data / "bank.npz", rules, glove, threshold, args.seed)
    timings["probe_seconds"] = round(time.perf_counter() - t0, 2)

    usage = {}
    for r in rules:
        usage[r.gesture_id] = usage.get(r.gesture_id, 0) + 1
    suggested, seen = [], set()
    for r in sorted(rules, key=lambda r: -r.score):
        if r.gesture_id in seen or len(r.phrase.split()) < 3:
            continue
        suggested.append(r.phrase); seen.add(r.gesture_id)
        if len(suggested) == 4:
            break
    if len(suggested) >= 2:
        suggested.append(" ".join(suggested[:2]))
    timings["total_seconds"] = round(time.perf_counter() - started, 2)
    at = report.get("at_threshold", {})
    metrics = {"bank_gestures": len(bank_ids), "video_clips_mined": len(mine_paths), "mining_windows": report["windows"],
               "threshold": threshold, "rules": len(rules), "distinct_rule_gestures": len(usage),
               "max_clip_share": round(max(usage.values()) / len(rules), 4) if rules else None,
               "window_pass_rate": at.get("window_pass_rate"), "mean_passing_gestures": at.get("mean_passing_gestures"),
               "calibration_warning": report.get("warning"), **heldout}
    manifest = {
        "schema": "paperreach.paper-method.v1", "repo": "automatic-text-to-gesture", "mode": "automatic",
        "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"path": str(source), "kind": kind, "takes": len(descriptors)},
        "roles": {**assignment, "probe_takes": [p.stem for p in probe_paths]}, "preset": args.preset,
        "threshold": threshold, "threshold_rule": rule, "seed": args.seed, "glove": pm.portable(glove),
        "min_similarity": args.min_similarity, "fps": pm.FPS, "units": "cm (neck-centred)",
        "files": {"bank": pm.portable(data / "bank.npz"), "library": pm.portable(data / "library3d.npz"),
                  "library_info": "library.json", "rules": "rules.jsonl",
                  "clips": [pm.portable(p) for p in mine_paths]},
        "export": summary, "calibration": calibration, "mine": mined, "metrics": metrics, "timings": timings,
        "suggested_queries": suggested, "heldout_probes": probe_phrases,
        "summary": {"rules": len(rules), "bank_gestures": len(bank_ids), "threshold": threshold,
                    "heldout_top1": heldout.get("heldout_top1"), "heldout_chance": heldout.get("heldout_chance"),
                    "data": f"BEAT {kind} speakers {', '.join(sorted({d['speaker'] for d in descriptors}, key=lambda s: (len(s), s)))} "
                            f"via beat_ingest export-automatic ({args.preset} preset)",
                    "seconds": timings["total_seconds"]},
    }
    pm.write_manifest(folder, manifest)
    pm.progress(f"done in {timings['total_seconds']} s: {len(rules)} rules over {len(usage)}/{len(bank_ids)} bank gestures "
                f"at threshold {threshold}; held-out top-1 {heldout.get('heldout_top1')} (chance {heldout.get('heldout_chance')})")
    return pm.ready(folder, manifest)


def main(argv=None):
    args = parse(argv)
    try:
        return prepare(args)
    except (Exception, SystemExit) as error:  # the launcher falls back to the default demo
        if isinstance(error, SystemExit) and error.code in (0, None):
            raise
        pm.progress(f"failed: {error.__class__.__name__}: {error}")
        return pm.emit({"ready": False, "reason": f"{error.__class__.__name__}: {error}"}, 1)


if __name__ == "__main__":
    raise SystemExit(main())
