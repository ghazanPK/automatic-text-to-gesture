"""Run the complete rule-mining and retrieval path on procedural data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frames = np.linspace(0.2, 1.0, 8, dtype=np.float32)
    pose = np.zeros((8, 4, 2), dtype=np.float32)
    pose[:, 2, 0] = frames
    pose[:, 3, 1] = frames[::-1]
    words = [
        {"word": "move", "start_frame": 0, "end_frame": 2},
        {"word": "forward", "start_frame": 2, "end_frame": 4},
        {"word": "stand", "start_frame": 4, "end_frame": 6},
        {"word": "together", "start_frame": 6, "end_frame": 8},
    ]
    bank = {"forward_motion": pose[:4], "together_motion": pose[4:]}
    np.savez(out / "video.npz", pose=pose, words_json=np.asarray(json.dumps(words)))
    np.savez(out / "bank.npz", **bank)

    glove = out / "tiny_glove.txt"
    vectors = {}
    for word, axis in (("move", 0), ("forward", 0), ("stand", 1), ("together", 1)):
        vector = np.zeros(300, np.float32); vector[axis] = 1; vectors[word] = vector
    glove.write_text("".join(f"{word} {' '.join(map(str, vector))}\n" for word, vector in vectors.items()), encoding="utf-8")

    cli = [sys.executable, "-m", "automatic_text_to_gesture.cli"]
    subprocess.run(cli + ["mine", "--video", str(out / "video.npz"), "--bank", str(out / "bank.npz"), "--threshold", "0.90", "--output", str(out / "rules.jsonl")], check=True)
    subprocess.run(cli + ["retrieve", "--rules", str(out / "rules.jsonl"), "--glove", str(glove), "--text", "move forward stand together", "--audio-seconds", "2.0", "--output", str(out / "sequence.json")], check=True)
    rules = [json.loads(line) for line in (out / "rules.jsonl").read_text(encoding="utf-8").splitlines() if line]
    sequence = json.loads((out / "sequence.json").read_text(encoding="utf-8"))
    if not rules or not sequence:
        raise RuntimeError("CLI verification workflow produced empty output")
    print(json.dumps({"rules": len(rules), "gestures": len(sequence), "output": str(out)}))


if __name__ == "__main__":
    main()
