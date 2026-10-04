from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from .core import TOKEN, load_glove, mine_rules, read_rules, retrieve, write_rules


def main() -> None:
    parser = argparse.ArgumentParser(prog="attg")
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine", help="mine timed text-to-gesture rules")
    mine.add_argument("--video", required=True, help="NPZ with pose and words_json")
    mine.add_argument("--bank", required=True, help="NPZ: gesture id -> [F,J,2]")
    mine.add_argument("--output", required=True)
    mine.add_argument("--threshold", type=float, default=.92)
    mine.add_argument("--seed", type=int, default=0)
    get = sub.add_parser("retrieve", help="retrieve a gesture sequence")
    get.add_argument("--rules", required=True)
    get.add_argument("--glove", required=True)
    get.add_argument("--text", required=True)
    get.add_argument("--audio-seconds", type=float)
    get.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "mine":
        video = np.load(args.video, allow_pickle=False)
        words = json.loads(str(video["words_json"]))
        bank_file = np.load(args.bank, allow_pickle=False)
        rules = mine_rules(video["pose"], words, {k: bank_file[k] for k in bank_file.files}, args.threshold, args.seed, Path(args.video).stem)
        write_rules(args.output, rules)
    else:
        rules = read_rules(args.rules)
        words = {w for r in rules for w in TOKEN.findall(r.phrase.lower())}
        words.update(TOKEN.findall(args.text.lower()))
        result = retrieve(args.text, rules, load_glove(args.glove, words), args.audio_seconds)
        output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
