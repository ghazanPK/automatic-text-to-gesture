"""Write a small GloVe file: the most frequent words plus every word spoken in a BEAT collection.

glove.6B.300d.txt (from https://nlp.stanford.edu/projects/glove/) lists words by
corpus frequency, so its first ``--top`` lines cover everyday query text; the
BEAT transcripts add every mined phrase word. The result keeps the original
text format, so ``attg retrieve --glove`` and the prepared demo read it
unchanged::

    python scripts/build_glove_subset.py --glove data/glove/glove.6B.300d.txt \
        --vocab-from /path/to/processed/beat --output data/glove/glove.6B.300d.subset.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"[\w']+")


def beat_vocabulary(root):
    """Lower-case word tokens from a processed (meta.json) or raw (TextGrid) BEAT folder."""
    root = Path(root)
    words = set()
    metas = [root / "meta.json"] if (root / "meta.json").is_file() else sorted(root.glob("*/meta.json"))
    for meta in metas:
        for take in json.loads(meta.read_text(encoding="utf-8")).get("takes", []):
            for w in take.get("words", []):
                words.update(TOKEN.findall(str(w.get("text", w.get("word", ""))).lower()))
    if not metas:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "beat_demo"))
        from beat_ingest import parse_textgrid
        for grid in sorted(set(root.glob("*.TextGrid")) | set(root.glob("*/*.TextGrid"))):
            for word, _, _ in parse_textgrid(grid):
                words.update(TOKEN.findall(word.lower()))
    return words


def build(glove, output, top=20000, vocabulary=()):
    wanted = {w.lower() for w in vocabulary}
    kept, found = 0, set()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(glove).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source):
            word = line.split(" ", 1)[0]
            if index < top or word in wanted:
                target.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                found.add(word)
    missing = sorted(wanted - found)
    return {"output": str(output), "words": kept, "vocabulary": len(wanted), "missing": len(missing),
            "missing_examples": missing[:20]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glove", type=Path, required=True, help="user-downloaded glove.6B.300d.txt (or any GloVe text file)")
    p.add_argument("--output", type=Path, default=Path("data/glove/glove.6B.300d.subset.txt"))
    p.add_argument("--top", type=int, default=20000, help="keep the first N (most frequent) words (default 20000)")
    p.add_argument("--vocab-from", type=Path, action="append", default=[],
                   help="processed or raw BEAT folder whose transcript words are kept; repeatable")
    p.add_argument("--extra-words", type=Path, help="text file with one additional word per line")
    a = p.parse_args(argv)
    vocabulary = set()
    for root in a.vocab_from:
        vocabulary |= beat_vocabulary(root)
    if a.extra_words:
        vocabulary |= {x.strip().lower() for x in a.extra_words.read_text(encoding="utf-8").splitlines() if x.strip()}
    print(json.dumps(build(a.glove, a.output, a.top, vocabulary)))


if __name__ == "__main__":
    main()
