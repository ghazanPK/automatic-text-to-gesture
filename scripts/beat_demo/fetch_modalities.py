"""Fetch explicitly named BEAT audio/face source files into local ignored scratch."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = "https://huggingface.co/datasets/H-Liu1997/BEAT/resolve/main/beat_english_v0.2.1/beat_english_v0.2.1"


def fetch(speaker, sequence, extension, output, max_bytes):
    if not sequence.startswith(f"{speaker}_") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in sequence):
        raise ValueError("Invalid speaker/sequence")
    if extension not in ("wav", "json", "bvh"):
        raise ValueError("Only original motion, audio and face files are allowed")
    url = f"{BASE}/{speaker}/{sequence}.{extension}"
    request = urllib.request.Request(url, headers={"User-Agent": "PaperReach-BEAT-sample/1.0"})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError(f"Remote file exceeds cap: {length} > {max_bytes}")
        with output.open("wb") as stream:
            while chunk := response.read(min(1024 * 1024, max_bytes - size + 1)):
                size += len(chunk)
                if size > max_bytes:
                    output.unlink(missing_ok=True)
                    raise ValueError(f"Remote file exceeded {max_bytes} byte cap")
                stream.write(chunk)
                digest.update(chunk)
    return {"url": url, "path": str(output), "bytes": size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", default="tmp/beat-demo/source")
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--include-bvh", action="store_true")
    args = parser.parse_args()
    records = []
    for extension in (("wav", "json", "bvh") if args.include_bvh else ("wav", "json")):
        output = Path(args.output_dir) / f"{args.sequence}.{extension}"
        if output.exists():
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            records.append({"path": str(output), "bytes": output.stat().st_size, "sha256": digest, "cached": True})
        else:
            records.append(fetch(args.speaker, args.sequence, extension, output, args.max_bytes))
    print(json.dumps(records))


if __name__ == "__main__":
    main()
