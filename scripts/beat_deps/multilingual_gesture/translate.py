"""Pluggable translators for the paper's translate-to-English step.

The paper used a hosted Korean-to-English service (Naver Papago). Any client
that implements :class:`Translator` can stand in for it:

* ``DictTranslator`` - exact source-string to translation map (offline fixture);
* ``HTTPTranslator`` - an OpenAI-compatible ``/chat/completions`` endpoint or a
  LibreTranslate-style ``/translate`` endpoint (self-hosted or hosted);
* ``LocalMTTranslator`` - a local Hugging Face translation model directory
  (for example a user-downloaded ``Helsinki-NLP/opus-mt-ko-en``), loaded with
  ``local_files_only=True`` so nothing is downloaded implicitly.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

LANGUAGE_NAMES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese", "es": "Spanish",
                  "fr": "French", "de": "German"}


@runtime_checkable
class Translator(Protocol):
    def translate(self, text: str, source_language: str, target_language: str = "en") -> str:
        """Return ``text`` translated from ``source_language`` into ``target_language``."""


def _same(a: str, b: str) -> bool:
    return a.lower().split("-")[0] == b.lower().split("-")[0]


class DictTranslator:
    """Exact-match translation table: ``{source_text: translation}``.

    A nested form ``{"ko->en": {...}, "en->ko": {...}}`` selects a direction;
    a flat table is treated as ``source -> en``.
    """

    def __init__(self, table: dict | None = None):
        self.table = table or {}

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")) if path else {})

    def translate(self, text, source_language, target_language="en"):
        if _same(source_language, target_language):
            return text
        nested = all(isinstance(v, dict) for v in self.table.values()) and self.table
        table = self.table.get(f"{source_language}->{target_language}", {}) if nested else (
            self.table if _same(target_language, "en") else {})
        if text not in table:
            raise ValueError(f"no {source_language}->{target_language} translation supplied for this text; "
                             "add it to --translations or choose --translator http/local")
        return str(table[text])


class HTTPTranslator:
    """HTTP translation client.

    ``api="openai"`` posts a chat completion to ``<url>/chat/completions`` (the
    URL may already end in that path) and expects the translation as the reply.
    ``api="libretranslate"`` posts ``{q, source, target}`` to ``<url>/translate``
    and reads ``translatedText``. An API key is read from the environment
    variable named by ``api_key_env`` when it is set.
    """

    def __init__(self, url, model=None, api="openai", api_key_env="OPENAI_API_KEY", timeout=60.0, opener=None):
        if api not in ("openai", "libretranslate"):
            raise ValueError("api must be 'openai' or 'libretranslate'")
        if api == "openai" and not model:
            raise ValueError("an OpenAI-compatible translator needs --translator-model")
        self.url, self.model, self.api, self.timeout = url.rstrip("/"), model, api, timeout
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self._open = opener or urllib.request.urlopen

    def _post(self, url, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers, method="POST")
        with self._open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def translate(self, text, source_language, target_language="en"):
        if _same(source_language, target_language):
            return text
        if self.api == "libretranslate":
            url = self.url if self.url.endswith("/translate") else self.url + "/translate"
            body = self._post(url, {"q": text, "source": source_language, "target": target_language,
                                    "format": "text"})
            out = body.get("translatedText")
        else:
            url = self.url if self.url.endswith("/chat/completions") else self.url + "/chat/completions"
            src = LANGUAGE_NAMES.get(source_language.lower()[:2], source_language)
            dst = LANGUAGE_NAMES.get(target_language.lower()[:2], target_language)
            prompt = (f"Translate the following {src} text into {dst}. "
                      f"Reply with the translation only.\n\n{text}")
            body = self._post(url, {"model": self.model, "temperature": 0,
                                    "messages": [{"role": "user", "content": prompt}]})
            out = body["choices"][0]["message"]["content"]
        if not isinstance(out, str) or not out.strip():
            raise ValueError("translator returned no text")
        return out.strip().strip('"')


class LocalMTTranslator:
    """Local sequence-to-sequence MT model directory via ``transformers``.

    ``model_path`` may be one directory (used for every direction) or a JSON
    mapping such as ``{"ko->en": "/models/opus-mt-ko-en", "en->ko": "..."}``.
    """

    def __init__(self, model_path, device=-1):
        self.paths = model_path if isinstance(model_path, dict) else {"*": str(model_path)}
        self.device, self._pipes = device, {}

    def _pipe(self, key):
        path = self.paths.get(key, self.paths.get("*"))
        if path is None:
            raise ValueError(f"no local MT model configured for {key}")
        if path not in self._pipes:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install transformers to use --translator local") from exc
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
            self._pipes[path] = (tok, model)
        return self._pipes[path]

    def translate(self, text, source_language, target_language="en"):
        if _same(source_language, target_language):
            return text
        tok, model = self._pipe(f"{source_language}->{target_language}")
        batch = tok([text], return_tensors="pt", truncation=True)
        out = model.generate(**batch, max_new_tokens=256)
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def make_translator(kind="dict", translations=None, url=None, model=None, api="openai",
                    api_key_env="OPENAI_API_KEY", model_path=None) -> Translator:
    kind = (kind or "dict").lower()
    if kind == "dict":
        return DictTranslator.from_file(translations)
    if kind == "http":
        if not url:
            raise ValueError("--translator http needs --translator-url")
        return HTTPTranslator(url, model, api, api_key_env)
    if kind == "local":
        if not model_path:
            raise ValueError("--translator local needs --mt-model-path")
        p = Path(model_path)
        mapping = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" and p.is_file() else str(p)
        return LocalMTTranslator(mapping)
    raise ValueError("translator must be dict, http or local")


def add_translator_arguments(parser):
    parser.add_argument("--translator", choices=("dict", "http", "local"), default="dict",
                        help="dict: --translations JSON; http: OpenAI-compatible or LibreTranslate endpoint; "
                             "local: transformers MT model directory")
    parser.add_argument("--translations", help="exact-text translation JSON for --translator dict")
    parser.add_argument("--translator-url", help="endpoint base URL for --translator http")
    parser.add_argument("--translator-model", help="model name for an OpenAI-compatible endpoint")
    parser.add_argument("--translator-api", choices=("openai", "libretranslate"), default="openai")
    parser.add_argument("--translator-api-key-env", default="OPENAI_API_KEY",
                        help="environment variable holding the endpoint key (optional)")
    parser.add_argument("--mt-model-path", help="local MT model directory, or JSON mapping 'src->tgt' to directories")


def translator_from_args(a) -> Translator:
    return make_translator(a.translator, a.translations, a.translator_url, a.translator_model,
                           a.translator_api, a.translator_api_key_env, a.mt_model_path)
