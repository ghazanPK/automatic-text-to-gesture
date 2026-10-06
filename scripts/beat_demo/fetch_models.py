"""Download the small default text model on first run: all-MiniLM-L6-v2 (about 92 MB).

Maintained in PaperReach ``tools/beat-demo/`` and vendored into each repository
as ``scripts/beat_demo/fetch_models.py`` by ``tools/integrate-beat-methods.py``;
``scripts/start_demo.py`` calls it before BEAT and paper-method preparation.

* The model goes into the repository's ignored ``models/all-MiniLM-L6-v2``
  folder, the default location every Sentence-BERT lookup in these
  repositories reads after ``BEAT_SBERT_MODEL`` and ``SBERT_MODEL``.
* An existing folder is reused. ``BEAT_SBERT_MODEL`` or ``SBERT_MODEL`` (your
  own model) skips the download.
* ``--offline``, ``PAPERREACH_OFFLINE=1`` or ``HF_HUB_OFFLINE=1`` skips the
  download.
* The files come from the pinned Hugging Face revision below; the weights are
  checked against their published size and SHA-256. ``HF_ENDPOINT`` selects a
  mirror.
* A failure is not fatal: the message says what happens instead (the demos use
  their labelled TF-IDF or bag-of-words fallbacks) and how to retry. The exit
  status is 0 when the model is available or deliberately skipped, 1 otherwise.

Only this small encoder is fetched automatically. Larger optional models
(bert-base, Kokoro, Whisper, LLMs, GloVe) stay opt-in and are documented in
each repository's README.

Standard library only, so it also runs before ``sentence-transformers`` is
installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

NAME = "all-MiniLM-L6-v2"
REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LICENSE = "Apache-2.0"
# (file, size in bytes, sha256 or None) at REVISION; model.safetensors is the 90.9 MB weight file.
FILES = (
    ("modules.json", 349, None),
    ("config_sentence_transformers.json", 116, None),
    ("sentence_bert_config.json", 53, None),
    ("config.json", 612, None),
    ("tokenizer.json", 466247, None),
    ("tokenizer_config.json", 350, None),
    ("special_tokens_map.json", 112, None),
    ("vocab.txt", 231508, None),
    ("1_Pooling/config.json", 190, None),
    ("README.md", 10502, None),
    ("model.safetensors", 90868376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
)
TOTAL_BYTES = sum(size for _, size, _ in FILES)
ENV_MODEL = ("BEAT_SBERT_MODEL", "SBERT_MODEL")
ENV_OFFLINE = ("PAPERREACH_OFFLINE", "HF_HUB_OFFLINE")
PROVENANCE = ".paperreach-download.json"
TRUE = {"1", "true", "yes", "on"}


def repo_root() -> Path:
    """``<repo>`` for the vendored copy in ``<repo>/scripts/beat_demo``."""
    return Path(__file__).resolve().parents[2]


def say(message: str) -> None:
    print(f"[models] {message}", flush=True)


def is_model_folder(folder: Path) -> bool:
    """A usable sentence-transformers folder: module list plus weights (downloaded here or saved by
    ``SentenceTransformer.save``)."""
    folder = Path(folder)
    return (folder / "modules.json").is_file() and any(
        (folder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def offline_reason(offline: bool = False) -> str | None:
    if offline:
        return "--offline"
    for name in ENV_OFFLINE:
        if os.environ.get(name, "").strip().lower() in TRUE:
            return f"{name}={os.environ[name]}"
    return None


def _url(name: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{REPO_ID}/resolve/{REVISION}/{name}"


def _download(name: str, size: int, sha256: str | None, target: Path, done: int, timeout: float) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(_url(name), headers={"User-Agent": "paperreach-fetch-models/1"})
    digest = hashlib.sha256()
    received, last = 0, 0.0
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as output:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            output.write(block)
            digest.update(block)
            received += len(block)
            now = time.monotonic()
            if size > 5_000_000 and now - last > 2:
                last = now
                say(f"  {(done + received) / 1e6:.1f} / {TOTAL_BYTES / 1e6:.1f} MB")
    if received != size:
        raise IOError(f"{name}: received {received} bytes, expected {size}")
    if sha256 and digest.hexdigest() != sha256:
        raise IOError(f"{name}: SHA-256 mismatch")
    return received


def fetch(models_dir: Path, timeout: float = 60.0) -> Path:
    """Download the pinned files into ``models_dir/NAME`` (atomically, through a partial folder)."""
    target = Path(models_dir) / NAME
    partial = Path(models_dir) / f".{NAME}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started, done = time.monotonic(), 0
    try:
        for name, size, sha256 in FILES:
            done += _download(name, size, sha256, partial / name, done, timeout)
        (partial / "2_Normalize").mkdir(exist_ok=True)  # module folder named in modules.json (no files)
        (partial / PROVENANCE).write_text(json.dumps({
            "model": NAME, "source": f"https://huggingface.co/{REPO_ID}", "revision": REVISION, "license": LICENSE,
            "bytes": done, "seconds": round(time.monotonic() - started, 1),
            "downloaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Downloaded by scripts/beat_demo/fetch_models.py; models/ is git-ignored and never committed."},
            indent=2) + "\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    say(f"saved {NAME} ({done / 1e6:.1f} MB in {time.monotonic() - started:.1f} s) to {target.parent.name}/{NAME}")
    return target


def ensure(models_dir: Path | None = None, offline: bool = False, timeout: float = 60.0) -> dict:
    """Make the default Sentence-BERT folder available; never raises. Returns a status dictionary."""
    models_dir = Path(models_dir) if models_dir else repo_root() / "models"
    target = models_dir / NAME
    for name in ENV_MODEL:
        if os.environ.get(name):
            say(f"{name} is set; using that model (no download)")
            return {"status": "override", "variable": name}
    if is_model_folder(target):
        return {"status": "present", "path": str(target)}
    reason = offline_reason(offline)
    if reason:
        say(f"{NAME} is not in {models_dir.name}/ and downloads are off ({reason}); the demos use their labelled "
            f"TF-IDF/bag-of-words text fallbacks. Fetch later with: python scripts/beat_demo/fetch_models.py")
        return {"status": "skipped", "reason": reason}
    say(f"first run: downloading {NAME} (about {TOTAL_BYTES / 1e6:.0f} MB, {LICENSE}) into ignored "
        f"{models_dir.name}/{NAME} from Hugging Face {REPO_ID}")
    try:
        fetch(models_dir, timeout)
        return {"status": "downloaded", "path": str(target)}
    except KeyboardInterrupt:
        say("download interrupted; nothing was kept")
        return {"status": "failed", "reason": "interrupted"}
    except Exception as error:  # network, disk or integrity problems must not stop the demo
        say(f"could not download {NAME}: {type(error).__name__}: {error}")
        say("the demo continues with its labelled TF-IDF/bag-of-words text fallback. Retry with "
            "'python scripts/beat_demo/fetch_models.py', set SBERT_MODEL to a local model folder, "
            "or pass --offline to skip this step")
        return {"status": "failed", "reason": f"{type(error).__name__}: {error}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, help="target folder (default: <repo>/models)")
    parser.add_argument("--offline", action="store_true", help="never download; report what is available")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request network timeout in seconds")
    args = parser.parse_args(argv)
    status = ensure(args.models_dir, args.offline, args.timeout)
    return 0 if status["status"] in {"present", "downloaded", "override", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
