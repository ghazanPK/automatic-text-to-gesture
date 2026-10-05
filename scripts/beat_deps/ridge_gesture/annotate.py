"""Rule-base phrase extraction (paper Section 3.1).

The paper sends TextGrid transcripts to an LLM with a fixed extraction prompt,
then matches the returned phrases back to the timed transcript. This module
holds that prompt verbatim, TextGrid I/O, OpenAI-compatible and local-command
LLM clients, JSON parsing and validation with provenance.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import urllib.request
from pathlib import Path

from .pipeline import heuristic_phrases, validate_phrases

# Verbatim from the RIDGE paper (Ali, Kim and Hwang, CAVW 2025), Section 3.1.
EXTRACTION_PROMPT = (
    "Task: Extract Key Gesture-Aligned Phrases. "
    "Description: Read the given content of the TextGrid file, format it into a clean and coherent paragraph. "
    "Identify and extract a few key phrases that are most likely to align well with co-speech gestures. "
    "Phrases should have a minimum length of 3 words and a maximum of 10 words. "
    "Focus on phrases that are meaningful, impactful, or highlight significant actions, emotions, or intentions. "
    "Limit the selection to phrases that stand out as the most gesture-relevant, avoiding over-extraction."
)

# Appended after the verbatim prompt so the reply can be parsed and aligned.
OUTPUT_FORMAT = (
    'Return only JSON of the form {"paragraph": "<clean paragraph>", "phrases": ["<phrase>", ...]}. '
    "Copy each phrase word for word from the transcript."
)

PROMPT_SHA256 = hashlib.sha256(EXTRACTION_PROMPT.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# TextGrid


def read_textgrid(path, tier="words"):
    """Word intervals ``[{word, start_seconds, end_seconds}]`` from a Praat
    long-format TextGrid (the format BEAT ships). Empty intervals are skipped."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"item \[\d+\]:", text)[1:] or [text]
    chosen = next((b for b in blocks if re.search(rf'name = "{re.escape(tier)}"', b)), blocks[0])
    words = []
    pattern = r'xmin = ([\d.eE+-]+)\s+xmax = ([\d.eE+-]+)\s+text = "((?:[^"]|"")*)"'
    for xmin, xmax, label in re.findall(pattern, chosen):
        label = label.replace('""', '"').strip()
        if label:
            words.append({"word": label, "start_seconds": float(xmin), "end_seconds": float(xmax)})
    return words


def frame_words(words, fps, frames=None):
    """Convert second timestamps to ordered, non-overlapping frame spans."""
    out, last_end = [], 0
    for w in words:
        s = max(int(round(float(w["start_seconds"]) * fps)), last_end)
        e = max(int(round(float(w["end_seconds"]) * fps)), s + 1)
        if frames is not None and e > frames:
            break
        out.append({"word": str(w["word"]), "start_frame": s, "end_frame": e}); last_end = e
    return out


def render_textgrid(words, fps=15):
    """Long-format TextGrid text for a timed word list (used when no file exists)."""
    end = max((w["end_frame"] for w in words), default=0) / fps
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {end:.3f}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "words"', "        xmin = 0", f"        xmax = {end:.3f}",
             f"        intervals: size = {len(words)}"]
    for i, w in enumerate(words, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {w['start_frame'] / fps:.3f}",
                  f"            xmax = {w['end_frame'] / fps:.3f}", f'            text = "{w["word"]}"']
    return "\n".join(lines) + "\n"


def beat_speaker(stem):
    """BEAT take names look like ``1_wayne_0_1_1``; the speaker is ``1_wayne``."""
    m = re.match(r"^(\d+)_([A-Za-z]+)_", stem)
    return f"{m.group(1)}_{m.group(2)}" if m else ""


def record_from_textgrid(path, fps=15, record_id=None, speaker=None, frames=None):
    path = Path(path)
    words = frame_words(read_textgrid(path), fps, frames)
    if not words:
        raise ValueError(f"{path} has no word intervals")
    return {"record_id": record_id or path.stem, "speaker": speaker if speaker is not None else beat_speaker(path.stem),
            "text": " ".join(w["word"] for w in words), "words": words, "textgrid": str(path), "fps": fps}


# ---------------------------------------------------------------------------
# LLM clients


def build_prompt(record, fps=15):
    source = record.get("textgrid")
    content = Path(source).read_text(encoding="utf-8", errors="replace") if source and Path(source).is_file() \
        else render_textgrid(record["words"], record.get("fps", fps))
    return f"{EXTRACTION_PROMPT}\n\n{OUTPUT_FORMAT}\n\nTextGrid content:\n{content}"


def call_openai_compatible(prompt, endpoint, model, api_key_env="OPENAI_API_KEY", temperature=0.0, timeout=120.0,
                           opener=None):
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(api_key_env) if api_key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers, method="POST")
    with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def call_command(prompt, command, timeout=600.0):
    """Run a local LLM CLI with the prompt on stdin; its stdout is the reply."""
    result = subprocess.run(shlex.split(command), input=prompt, capture_output=True, text=True, timeout=timeout,
                            encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"LLM command failed ({result.returncode}): {result.stderr.strip()[:500]}")
    return result.stdout


def parse_phrases(reply):
    """Phrase strings from an LLM reply: a JSON object with ``phrases``, a JSON
    list of strings or of ``{"phrase": ...}`` objects; code fences are ignored."""
    text = re.sub(r"```(?:json)?", "", reply).strip()
    candidates = [text] + re.findall(r"\{.*\}|\[.*\]", text, flags=re.S)
    for chunk in candidates:
        try:
            value = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(value, dict):
            value = value.get("phrases", value.get("key_phrases", []))
        if isinstance(value, list):
            return [str(v.get("phrase", "")) if isinstance(v, dict) else str(v) for v in value if v]
    raise ValueError("LLM reply contains no JSON phrase list")


# ---------------------------------------------------------------------------
# Annotation driver


def _now():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def annotate_record(record, mode="heuristic", reviewed=None, endpoint=None, model=None, command=None,
                    api_key_env="OPENAI_API_KEY", min_words=3, max_words=10, limit=5, call=None):
    """Annotate one transcript record; returns a JSON-serialisable row.

    ``mode`` is ``heuristic``, ``reviewed`` (``reviewed`` list supplied) or ``llm``
    (``endpoint``+``model`` or ``command``). ``call`` overrides the LLM client
    (tests). Every row records provenance and rejected proposals.
    """
    provenance = {"annotator": mode, "created": _now()}
    if mode == "llm":
        prompt = build_prompt(record)
        if call is None:
            if command:
                call = lambda p: call_command(p, command)
            elif endpoint and model:
                call = lambda p: call_openai_compatible(p, endpoint, model, api_key_env)
            else:
                raise ValueError("LLM annotation needs --llm-endpoint and --model, or --llm-command")
        reply = call(prompt)
        proposed = parse_phrases(reply)
        provenance.update(prompt_source="RIDGE paper Section 3.1 (verbatim) + JSON output format",
                          prompt_sha256=PROMPT_SHA256, model=model, endpoint=endpoint, command=command,
                          raw_response=reply)
    elif mode == "reviewed":
        proposed = list(reviewed or [])
    elif mode == "heuristic":
        proposed = heuristic_phrases(record["text"], min_words, max_words, limit)
        provenance["note"] = "content-word heuristic, not the paper's LLM extraction"
    else:
        raise ValueError("mode must be heuristic, reviewed or llm")
    accepted, rejected = validate_phrases(proposed, record["words"], min_words, max_words)
    provenance["rejected"] = rejected
    return {"record_id": record["record_id"], "speaker": record.get("speaker", ""), "phrases": accepted,
            "annotator": mode, "provenance": provenance}
