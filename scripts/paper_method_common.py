"""Helpers shared by ``prepare_paper_method.py`` and the prepared demo mode.

This file belongs to the repository (it is not a vendored shared file). It
finds a local BEAT source and a Sentence-BERT directory, runs this
repository's own command line in-process, caches results by their settings,
prints the launcher's final JSON line and serves the prepared browser API.
BEAT loading itself goes through the vendored ``scripts/beat_demo/beat_ingest.py``.
Data, prepared arrays and fitted weights stay in ignored ``outputs/``.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs" / "paper-method"
SBERT_NAME = "all-MiniLM-L6-v2"
DEFAULT_SBERT_DIR = ROOT / "models" / SBERT_NAME
ENV_PROCESSED, ENV_RAW, ENV_SBERT, ENV_GLOVE = "BEAT_PROCESSED_ROOT", "BEAT_RAW_ROOT", "SBERT_MODEL", "GLOVE_PATH"
ENV_SBERT_ORDER = ("BEAT_SBERT_MODEL", ENV_SBERT)  # the lookup order of the shared demo layer
SBERT_STEP = ("python scripts/beat_demo/fetch_models.py (downloads all-MiniLM-L6-v2, about 92 MB, into ignored "
              "models/; python scripts/start_demo.py does this on first run)")
FPS = 15
UNIT_SECONDS = 3.0
API_PATHS = {"/api/beat-library", "/api/beat-query", "/api/query"}


# ----------------------------------------------------------------------------
# Imports and progress

def use_repository_package(package):
    """Import ``package`` from this repository's ``src``, not the copy vendored in ``scripts/beat_deps``."""
    src = (ROOT / "src").resolve()
    if str(src) in sys.path:
        sys.path.remove(str(src))
    sys.path.insert(0, str(src))
    for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
        origin = getattr(sys.modules[name], "__file__", None)
        if not origin or src not in Path(origin).resolve().parents:
            del sys.modules[name]


def ingest():
    """The vendored shared BEAT loader (``scripts/beat_demo/beat_ingest.py``)."""
    folder = str(SCRIPTS / "beat_demo")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import beat_ingest
    return beat_ingest


def progress(message):
    """Human-readable progress goes to stderr; stdout ends with the single JSON result line."""
    print(f"[paper-method] {message}", file=sys.stderr, flush=True)


def run_cli(main, argv, label):
    """Run this repository's CLI ``main(argv)`` in-process; return (last JSON object printed, seconds)."""
    argv = [str(a) for a in argv]
    progress(f"{label}: {' '.join(argv)}")
    buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        main(argv)
    text = buffer.getvalue()
    if text:
        sys.stderr.write(text)
    parsed = None
    for line in reversed([x for x in text.splitlines() if x.strip()]):
        try:
            parsed = json.loads(line)
            break
        except ValueError:
            continue
    return parsed, round(time.perf_counter() - start, 2)


def emit(result, code=0):
    """Print the launcher's final JSON line and return the exit status."""
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return code


def not_ready(reason, steps=()):
    progress(f"not ready: {reason}")
    for step in steps:
        progress(f"  next: {step}")
    return emit({"ready": False, "reason": reason, "next_steps": list(steps)})


# ----------------------------------------------------------------------------
# Sources and models

def add_source_args(parser, speakers, max_takes, roles_help):
    parser.add_argument("--processed", type=Path, help=f"OmniMo processed BEAT root (<speaker>/meta.json); env {ENV_PROCESSED}")
    parser.add_argument("--beat-root", type=Path, help=f"raw BEAT folder (beat_english_v0.2.1/<speaker>/<take>.bvh+.TextGrid); env {ENV_RAW}")
    parser.add_argument("--speakers", default=speakers, help=f"comma list of speaker ids/names, or 'all' (default {speakers})")
    parser.add_argument("--takes", help="comma list of take ids or globs")
    parser.add_argument("--max-takes-per-speaker", type=int, default=max_takes,
                        help=f"takes per speaker; 0 means all (default {max_takes})")
    parser.add_argument("--max-frames", type=int, help="cap frames per take at 15 fps (quick checks)")
    if roles_help:
        parser.add_argument("--role", action="append", help=f"role override role=ids|fraction|nCOUNT|rest ({roles_help})")
        parser.add_argument("--role-unit", choices=["auto", "speaker", "take"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS, help="cache root (default outputs/paper-method)")
    parser.add_argument("--force", action="store_true", help="rebuild even when a cached result matches")


def find_source(processed=None, raw=None):
    """Return (path, kind) for the first available BEAT source, or (None, None)."""
    choices = []
    if raw:
        choices.append((Path(raw), "raw"))
    if processed:
        choices.append((Path(processed), "processed"))
    if not choices:
        if os.environ.get(ENV_PROCESSED):
            choices.append((Path(os.environ[ENV_PROCESSED]), "processed"))
        if os.environ.get(ENV_RAW):
            choices.append((Path(os.environ[ENV_RAW]), "raw"))
        choices += [(ROOT / "data/beat/processed", "processed"), (ROOT / "data/beat/beat_english_v0.2.1", "raw"),
                    (ROOT / "data/beat/raw", "raw")]
    for path, kind in choices:
        if path.is_dir():
            try:
                ingest().detect_kind(path)
            except FileNotFoundError:
                if processed or raw:
                    raise
                continue
            return path.resolve(), kind
        if processed or raw:
            raise FileNotFoundError(f"{path} is not a folder")
    return None, None


SOURCE_STEPS = (
    f"processed OmniMo collection: python scripts/prepare_paper_method.py --processed /path/to/processed/beat "
    f"(or set {ENV_PROCESSED}, or place it at data/beat/processed)",
    "raw public BEAT: download beat_english_v0.2.1/<speaker>/<take>.bvh and .TextGrid from the Hugging Face "
    "dataset H-Liu1997/BEAT, then pass --beat-root data/beat/beat_english_v0.2.1",
)


def selection(args):
    speakers = None if str(args.speakers).strip().lower() in ("", "all") else args.speakers
    takes = None if args.max_takes_per_speaker in (None, 0) else int(args.max_takes_per_speaker)
    return {"speakers": speakers, "takes": args.takes, "max_takes_per_speaker": takes}


def role_spec(args):
    return ingest().parse_role_spec(args.role) if getattr(args, "role", None) else None


def find_sbert(value=None):
    """Return (model path or name, None) or (None, instructions).

    Lookup order: ``value``, ``BEAT_SBERT_MODEL``, ``SBERT_MODEL``, ``models/all-MiniLM-L6-v2``. Nothing is
    downloaded here; ``scripts/start_demo.py`` fetches the default model on first run.
    """
    if value:
        path = Path(value)
        return (str(path.resolve()) if path.exists() else str(value)), None
    for name in ENV_SBERT_ORDER:
        if os.environ.get(name):
            return os.environ[name], None
    if DEFAULT_SBERT_DIR.is_dir():
        return str(DEFAULT_SBERT_DIR.resolve()), None
    return None, (f"Sentence-BERT was not found at models/{SBERT_NAME}. Download it once with: {SBERT_STEP}; "
                  f"then rerun, or pass --sbert <local directory> (env {ENV_SBERT})")


def portable(path):
    """Repository-relative POSIX path when possible (manifests stay valid if the repository moves)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def code_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for item in files:
            if "__pycache__" in item.parts:
                continue
            digest.update(item.relative_to(ROOT).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def cache_key(settings, code_paths):
    payload = json.dumps(settings, sort_keys=True, default=str) + code_digest(code_paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cached(folder):
    manifest = Path(folder) / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("complete"):
            return data
    return None


def write_manifest(folder, manifest):
    folder = Path(folder)
    manifest["complete"] = True
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = folder.parent / "latest.json"
    latest.write_text(json.dumps({"folder": portable(folder), "key": folder.name,
                                  "created": manifest.get("created")}, indent=2), encoding="utf-8")


def ready(folder, manifest, extra_args=(), cached_result=False):
    summary = dict(manifest.get("summary", {}))
    summary["cached"] = bool(cached_result)
    return emit({"ready": True, "server_args": ["scripts/demo_server.py", "--prepared", portable(folder), *extra_args],
                 "summary": summary})


# ----------------------------------------------------------------------------
# BEAT records and windows

def clean_words(words, frames):
    """Ordered, non-overlapping, in-range word spans (15 fps rounding can make neighbours overlap)."""
    out, last = [], 0
    for word, start, end in sorted(words, key=lambda w: (w[1], w[2])):
        start, end = max(int(start), last), min(int(end), frames)
        if end <= start:
            continue
        out.append((str(word), start, end))
        last = end
    return out


def load_takes(descriptors, max_frames=None):
    """Upper-body, per-frame neck-centred 15 fps records with positions in centimetres."""
    beat = ingest()
    records = []
    for d in descriptors:
        record = beat.center(beat.load_take(d, fps=FPS, max_frames=max_frames, joints=beat.UPPER_BODY), "Neck")
        record["positions"] = np.asarray(record["positions"], np.float32) * 100.0
        record["words"] = clean_words(record["words"], len(record["positions"]))
        records.append(record)
    return records


def span_text(words, start, end):
    """Words whose midpoint lies in [start, end) (the shared ingest window rule)."""
    return " ".join(w for w, s, e in words if start <= (s + e) / 2 < end)


def take_windows(records, length=None, min_words=1):
    """Fixed windows with the shared IDs ``<take>:<start>-<end>`` (same rule as ``beat_ingest.windows``)."""
    beat = ingest()
    length = length or int(round(FPS * UNIT_SECONDS))
    out = []
    for record in records:
        out.extend(beat.windows(record, length, min_words=min_words))
    return out


def descriptor_roles(assignment):
    """{take: role} from a ``joint_order.json`` role assignment."""
    return {take: role for role, takes in assignment["takes"].items() for take in takes}


def source_info(record_source):
    keys = ("kind", "version", "motion_url", "alignment_url")
    return {k: record_source.get(k) for k in keys if record_source.get(k) is not None}


# ----------------------------------------------------------------------------
# Prepared demo payloads

def frames_m(clip, length=None, joints=11):
    """Contract motion (cm, flat or [F,J,3]) to viewer frames in metres."""
    clip = np.asarray(clip, np.float32)
    clip = clip.reshape(len(clip), joints, 3) if clip.ndim == 2 else clip
    if length is not None:
        clip = clip[: int(length)]
    return np.round(clip / 100.0, 4).tolist()


def rest_frames(rest_pose_cm, words, fps=FPS):
    count = max(15, int(round(fps * 0.35 * max(1, len(str(words).split())))))
    frame = np.round(np.asarray(rest_pose_cm, np.float32) / 100.0, 4).tolist()
    return [frame] * count


def param(params, name, default=None, cast=str):
    value = params.get(name, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a {cast.__name__}") from None


def idle_slot(text, rest_pose_cm, reason, **detail):
    return {"gesture_id": "idle", "text": text, "frames": rest_frames(rest_pose_cm, text), "route": "idle_no_match",
            "confidence": 0.0, "source": {"kind": "idle"}, "rule_source": {"kind": "idle", "reason": reason},
            "blend_frames": 0, "idle": True, **detail}


def query_result(slots, *, algorithm, data_label, metrics, trace, joints, extra=None):
    routes = [s["route"] for s in slots]
    counts = {}
    for route in routes:
        counts[route] = counts.get(route, 0) + 1
    result = {"ready": True, "prepared": True, "fps": FPS, "joint_order": list(joints), "axisSigns": [1, 1, 1],
              "slots": slots, "no_match": bool(slots) and all(r == "idle_no_match" for r in routes),
              "algorithm": algorithm, "data_label": data_label,
              "trace": {**trace, "routes": routes},
              "metrics": {**metrics, "route_counts": counts}}
    result.update(extra or {})
    return result


def check_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Supply 1-2000 characters of query text")
    return text.strip()


def handle(handler, demo):
    """Serve the browser contract from a prepared demo; returns False for other paths."""
    parsed = urlsplit(handler.path)
    if parsed.path not in API_PATHS:
        return False
    try:
        if parsed.path == "/api/beat-library":
            result = demo.library()
        else:
            params = parse_qs(parsed.query)
            if handler.command == "POST":
                size = int(handler.headers.get("Content-Length", "0") or 0)
                if not 0 < size <= 32_000:
                    raise ValueError("Text query must be under 32 KB")
                body = json.loads(handler.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("POST a JSON object")
                params.update({k: (v if isinstance(v, list) else [v]) for k, v in body.items()})
            text = params.pop("text", [""])
            result = demo.query(check_text(text[0] if isinstance(text, list) else text), params)
        status = 200
    except (ValueError, KeyError, FileNotFoundError) as error:
        result, status = {"error": str(error)}, 400
    except Exception as error:  # report instead of dropping the connection
        result, status = {"error": f"{error.__class__.__name__}: {error}"}, 500
    body = json.dumps(result, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def load_manifest(folder):
    folder = resolve(folder)
    if (folder / "latest.json").is_file() and not (folder / "manifest.json").is_file():
        folder = resolve(json.loads((folder / "latest.json").read_text(encoding="utf-8"))["folder"])
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} is missing; run python scripts/prepare_paper_method.py")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise ValueError(f"{manifest} is incomplete; rerun python scripts/prepare_paper_method.py --force")
    return folder, data
