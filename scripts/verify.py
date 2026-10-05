"""Run the complete multi-clip mining, calibration and Manual/Auto/Hybrid retrieval path on procedural data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gesture(rng: np.random.Generator, frames: int, joints: int = 5) -> np.ndarray:
    pose = rng.normal(size=(frames, joints, 2)).astype(np.float32)
    pose[:, 1] = 0  # neck
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # Variable-length bank (6-8 frames); the stride is the longest gesture, shorter ones are centre-padded.
    lengths = {"forward_motion": 8, "together_motion": 6, "negation_sweep": 7, "self_point": 6, "inclusive_open": 8}
    bank = {gid: gesture(rng, n) for gid, n in lengths.items()}
    np.savez(out / "bank.npz", **bank)

    # Two clips; each 8-frame window embeds one gesture at its centre-pad offset, other frames are unrelated motion.
    plan = {"clip_a": [("forward_motion", ["move", "forward"]), ("together_motion", ["stand", "together"])],
            "clip_b": [("negation_sweep", ["never", "again"]), ("self_point", ["i", "myself"])]}
    for name, windows in plan.items():
        pose = gesture(rng, 8 * len(windows))
        words = []
        for k, (gid, ws) in enumerate(windows):
            g = bank[gid]
            before = (8 - len(g)) // 2
            pose[8 * k + before : 8 * k + before + len(g)] = g
            words += [{"word": w, "start_frame": 8 * k + 3 * i, "end_frame": 8 * k + 3 * i + 3} for i, w in enumerate(ws)]
        np.savez(out / f"{name}.npz", pose=pose, words_json=np.asarray(json.dumps(words)), clip_id=np.asarray(name))
    (out / "clips.txt").write_text("clip_b.npz\n", encoding="utf-8")

    glove = out / "tiny_glove.txt"
    axes = {"move": 0, "forward": 0, "ahead": 0, "stand": 1, "together": 1, "everyone": 1, "never": 2, "again": 2, "i": 3, "myself": 3}
    lines = []
    for word, axis in axes.items():
        vector = np.zeros(300, np.float32); vector[axis] = 1
        lines.append(f"{word} {' '.join(map(str, vector))}\n")
    glove.write_text("".join(lines), encoding="utf-8")

    cli = [sys.executable, "-m", "automatic_text_to_gesture.cli"]
    subprocess.run(cli + ["mine", "--video", str(out / "clip_a.npz"), "--manifest", str(out / "clips.txt"), "--bank", str(out / "bank.npz"),
                          "--threshold", "0.92", "--output", str(out / "rules.jsonl")], check=True)
    subprocess.run(cli + ["import-manual", "--input", str(ROOT / "examples" / "manual_map_nvbg.xml"), "--output", str(out / "manual_map.json")], check=True)
    text = "move forward and go ahead again and again it goes zebra quantum xylophone blorp flux"
    sequences = {}
    for mode in ("auto", "manual", "hybrid"):
        target = out / f"sequence-{mode}.json"
        extra = ["--manual", str(out / "manual_map.json")] if mode != "auto" else []
        subprocess.run(cli + ["retrieve", "--map", mode, "--rules", str(out / "rules.jsonl"), "--glove", str(glove), *extra,
                              "--text", text, "--audio-seconds", "3.0", "--output", str(target)], check=True)
        sequences[mode] = json.loads(target.read_text(encoding="utf-8"))

    rules = [json.loads(line) for line in (out / "rules.jsonl").read_text(encoding="utf-8").splitlines() if line]
    report = json.loads((out / "rules.jsonl.calibration.json").read_text(encoding="utf-8"))
    planted = {(r["source"], r["start_frame"]): r["gesture_id"] for r in rules}
    expected = {(name, 8 * k): gid for name, windows in plan.items() for k, (gid, _) in enumerate(windows)}
    if planted != expected or min(r["score"] for r in rules) < 0.99:
        raise RuntimeError(f"padding-aware mining did not recover the planted gestures: {planted}")
    routes = {mode: [slot["map"] for slot in seq] for mode, seq in sequences.items()}
    if routes != {"auto": ["auto", "auto", "idle"], "manual": ["manual", "idle", "idle"], "hybrid": ["manual", "auto", "idle"]}:
        raise RuntimeError(f"unexpected map routes: {routes}")
    print(json.dumps({"clips": 2, "rules": len(rules), "threshold": report["threshold"],
                      "window_pass_rate": report["at_threshold"]["window_pass_rate"], "routes": routes, "output": str(out)}))


if __name__ == "__main__":
    main()
