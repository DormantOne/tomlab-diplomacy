"""
llm_providers.py
================

One place to pick the model that drives the Diplomacy agents. The rest of the
codebase only needs a callable `llm_call(prompt: str) -> str`; this module builds
that callable for whichever provider is available/selected:

    - ollama     (local, e.g. gpt-oss:20b / llama3.1)  — no key, needs the daemon
    - anthropic  (Claude Haiku by default)             — ANTHROPIC_API_KEY
    - openai     (gpt-4o-mini by default)              — OPENAI_API_KEY
    - google     (gemini-1.5-flash by default)         — GOOGLE_API_KEY / GEMINI_API_KEY

SDK-free (urllib only), matching the existing callers' style. On any error a
caller returns "{}" so the downstream JSON parsers degrade gracefully rather
than crash — same contract the old anthropic/ollama callers used.

Selection precedence:
    1. explicit kind/model passed in (e.g. from the New Game dialog)
    2. env overrides  DIPLOMACY_LLM_KIND / DIPLOMACY_LLM_MODEL
    3. auto-detect: first available provider in PRIORITY order
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional, Tuple

KINDS = ("ollama", "anthropic", "openai", "google")
# When auto-detecting, prefer a hosted model (historically local was too weak
# to drive these agents) but fall back to local Ollama if that's all there is.
PRIORITY = ("anthropic", "openai", "google", "ollama")

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "ollama": os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------
def _key(kind: str) -> str:
    if kind == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if kind == "openai":
        return os.environ.get("OPENAI_API_KEY", "").strip()
    if kind == "google":
        return (os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", "")).strip()
    return ""


def _ollama_up(timeout: float = 0.6) -> bool:
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def available() -> Dict[str, Dict[str, object]]:
    """Report which providers can be used right now and their default model."""
    out: Dict[str, Dict[str, object]] = {}
    for k in ("anthropic", "openai", "google"):
        out[k] = {"available": bool(_key(k)), "model": DEFAULT_MODELS[k],
                  "detail": "env key set" if _key(k) else "no API key in env"}
    up = _ollama_up()
    out["ollama"] = {"available": up, "model": DEFAULT_MODELS["ollama"],
                     "detail": f"daemon at {OLLAMA_URL}" if up else f"no daemon at {OLLAMA_URL}"}
    return out


def resolve(kind: Optional[str] = None, model: Optional[str] = None
            ) -> Tuple[Optional[str], Optional[str]]:
    """Apply precedence and return (kind, model). kind is None if nothing usable."""
    kind = (kind or os.environ.get("DIPLOMACY_LLM_KIND") or "").strip().lower() or None
    model = (model or os.environ.get("DIPLOMACY_LLM_MODEL") or "").strip() or None
    avail = available()
    if kind not in KINDS:
        kind = None
    if kind is None:
        for k in PRIORITY:
            if avail[k]["available"]:
                kind = k
                break
    if kind and not model:
        model = str(DEFAULT_MODELS.get(kind))
    return kind, model


def label(kind: Optional[str], model: Optional[str]) -> str:
    return f"{kind}:{model}" if kind else "(no provider)"


def describe() -> str:
    """One-line summary for health logs."""
    a = available()
    parts = [f"{k}={'✓' if a[k]['available'] else '—'}" for k in KINDS]
    return "providers: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Provider callers (each returns text; "{}" on error)
# ---------------------------------------------------------------------------
SYSTEM_MSG = ("You are an LLM driving one power in a Diplomacy game. Follow the "
              "user message exactly. When asked for JSON, output JSON only "
              "(no prose, no code fences).")


def _post(url: str, body: dict, headers: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _anthropic(model: str, max_tokens: int, timeout: float) -> Callable[[str], str]:
    key = _key("anthropic")

    def _call(prompt: str) -> str:
        try:
            data = _post("https://api.anthropic.com/v1/messages",
                         {"model": model, "max_tokens": max_tokens, "system": SYSTEM_MSG,
                          "messages": [{"role": "user", "content": prompt}]},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout)
            return "\n".join(b.get("text", "") for b in data.get("content", [])
                             if isinstance(b, dict) and b.get("type") == "text")
        except Exception as e:
            print(f"  [anthropic error] {e}", file=sys.stderr)
            return "{}"
    return _call


def _openai(model: str, max_tokens: int, timeout: float) -> Callable[[str], str]:
    key = _key("openai")

    def _call(prompt: str) -> str:
        try:
            data = _post("https://api.openai.com/v1/chat/completions",
                         {"model": model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": SYSTEM_MSG},
                                       {"role": "user", "content": prompt}]},
                         {"Authorization": f"Bearer {key}"}, timeout)
            return (((data.get("choices") or [{}])[0]).get("message", {}) or {}).get("content", "") or ""
        except Exception as e:
            print(f"  [openai error] {e}", file=sys.stderr)
            return "{}"
    return _call


def _google(model: str, max_tokens: int, timeout: float) -> Callable[[str], str]:
    key = _key("google")

    def _call(prompt: str) -> str:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            data = _post(url, {"system_instruction": {"parts": [{"text": SYSTEM_MSG}]},
                               "contents": [{"parts": [{"text": prompt}]}],
                               "generationConfig": {"maxOutputTokens": max_tokens}},
                         {}, timeout)
            cand = (data.get("candidates") or [{}])[0]
            return "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []))
        except Exception as e:
            print(f"  [google error] {e}", file=sys.stderr)
            return "{}"
    return _call


def _ollama(model: str, timeout: float) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        try:
            data = _post(OLLAMA_URL + "/api/generate",
                         {"model": model, "prompt": prompt, "stream": False,
                          "system": SYSTEM_MSG, "options": {"temperature": 0.7}},
                         {}, timeout)
            return data.get("response", "") or ""
        except Exception as e:
            print(f"  [ollama error] {e}", file=sys.stderr)
            return "{}"
    return _call


def make_llm_call(kind: str, model: str, *, max_tokens: int = 1500,
                  timeout: float = 120.0) -> Callable[[str], str]:
    """Build the prompt->text callable for the chosen provider."""
    if kind == "anthropic":
        return _anthropic(model, max_tokens, timeout)
    if kind == "openai":
        return _openai(model, max_tokens, timeout)
    if kind == "google":
        return _google(model, max_tokens, timeout)
    if kind == "ollama":
        return _ollama(model, max(timeout, 300.0))  # local models are slower
    raise ValueError(f"unknown LLM kind {kind!r}")


def build(kind: Optional[str] = None, model: Optional[str] = None
          ) -> Tuple[Callable[[str], str], str, str]:
    """Resolve + build in one step. Returns (caller, kind, model).
    Raises RuntimeError if no provider is available."""
    k, m = resolve(kind, model)
    if not k:
        raise RuntimeError(
            "No LLM provider available. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY / "
            "GOOGLE_API_KEY), or start a local Ollama daemon, then restart.")
    return make_llm_call(k, m), k, m
