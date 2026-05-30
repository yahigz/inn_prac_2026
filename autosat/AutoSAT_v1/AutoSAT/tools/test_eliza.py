#!/usr/bin/env python3
"""
Quick smoke-test for the Eliza API (Anthropic / Claude endpoint).

Based on official Eliza docs: https://wiki.yandex-team.ru/eliza/vendors/claude/

Uses raw HTTP requests only (no anthropic SDK), exactly as shown in the docs:

    import requests, os
    url = "https://api.eliza.yandex.net/anthropic/v1/messages"
    headers = {
        "authorization": f"OAuth {os.getenv('SOY_TOKEN')} ",
        "content-type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, verify=False)

Usage
-----
    # Token is picked up automatically from the environment or .env file:
    python3 tools/test_eliza.py

    # Override model:
    AUTOSAT_LLM_MODEL=claude-3-5-sonnet-20241022 python3 tools/test_eliza.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Load .env from project root (if present)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # AutoSAT root


def _load_env():
    for candidate in [_HERE / ".env", _HERE.parent / ".env", _HERE.parent.parent / ".env"]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            print(f"[env] Loaded {candidate}", flush=True)
            break


_load_env()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOY_TOKEN = os.getenv("AUTOSAT_API_KEY", "").strip()
# Actual available models (from GET /anthropic/v1/models):
#   claude-sonnet-4-6        (recommended, supports tool_use + structured_outputs)
#   claude-opus-4-7          (most powerful)
#   claude-sonnet-4-5-20250929
#   claude-haiku-4-5-20251001
MODEL = os.getenv("AUTOSAT_LLM_MODEL", "claude-sonnet-4-6").strip()

# Eliza endpoint per official docs
MESSAGES_URL = "https://api.eliza.yandex.net/anthropic/v1/messages"

MAX_TOKENS = 256
TEST_PROMPT = "Say hello and tell me which model you are in one sentence."

if not SOY_TOKEN:
    print(
        "[ERROR] AUTOSAT_API_KEY is not set.\n"
        "  Set it in .env:  AUTOSAT_API_KEY=<your_soy_oauth_token>\n"
        "  Or export it:    export AUTOSAT_API_KEY=<token>",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[Config] model={MODEL}", flush=True)


def _make_headers() -> dict:
    """Auth header per official Eliza docs — trailing space is intentional."""
    return {
        "authorization": f"OAuth {SOY_TOKEN} ",
        "content-type": "application/json",
    }


def _unwrap(data: dict) -> dict:
    """Unwrap Eliza envelope: actual Anthropic response is under 'response' key.

    Eliza returns: {"key": "...", "response": {<actual Anthropic response>}, ...}
    """
    if "response" in data and isinstance(data["response"], dict):
        return data["response"]
    return data


# ---------------------------------------------------------------------------
# Test 1: plain text request (matches official Python example exactly)
# ---------------------------------------------------------------------------
def test_plain():
    print("\n[Test 1] Plain text request via Eliza ...", flush=True)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }
    resp = requests.post(MESSAGES_URL, json=payload, headers=_make_headers(), verify=False, timeout=60)
    print(f"[plain] HTTP {resp.status_code}", flush=True)
    resp.raise_for_status()
    data = _unwrap(resp.json())
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    print(f"[plain] Response: {text}", flush=True)
    print(f"[plain] Usage: input={usage.get('input_tokens')} output={usage.get('output_tokens')}", flush=True)
    return bool(text)


# ---------------------------------------------------------------------------
# Test 2: request with system prompt
# ---------------------------------------------------------------------------
def test_with_system():
    print("\n[Test 2] Request with system prompt via Eliza ...", flush=True)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": "You are a helpful assistant. Always respond in exactly one sentence.",
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }
    resp = requests.post(MESSAGES_URL, json=payload, headers=_make_headers(), verify=False, timeout=60)
    print(f"[system] HTTP {resp.status_code}", flush=True)
    resp.raise_for_status()
    data = _unwrap(resp.json())
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    print(f"[system] Response: {text}", flush=True)
    print(f"[system] Usage: input={usage.get('input_tokens')} output={usage.get('output_tokens')}", flush=True)
    return bool(text)


# ---------------------------------------------------------------------------
# Test 3: tool_use (structured output)
# ---------------------------------------------------------------------------
def test_tool_use():
    print("\n[Test 3] Tool use (structured output) via Eliza ...", flush=True)
    tool_schema = {
        "name": "emit_answer",
        "description": "Emit a structured answer. Always call this tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "greeting": {"type": "string", "description": "A greeting message."},
                "model_name": {"type": "string", "description": "The model name."},
            },
            "required": ["greeting", "model_name"],
        },
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "tools": [tool_schema],
        "tool_choice": {"type": "any"},
    }
    resp = requests.post(MESSAGES_URL, json=payload, headers=_make_headers(), verify=False, timeout=60)
    print(f"[tool_use] HTTP {resp.status_code}", flush=True)
    resp.raise_for_status()
    data = _unwrap(resp.json())
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            tool_input = block.get("input", {})
            print(f"[tool_use] tool_input: {json.dumps(tool_input, ensure_ascii=False)}", flush=True)
            return True
    print(f"[tool_use] No tool_use block. content={json.dumps(data.get('content', []), indent=2)}", flush=True)
    return False


# ---------------------------------------------------------------------------
# Test 4: AutoSAT ElizaCallAPI integration test
# ---------------------------------------------------------------------------
def test_autosat_eliza():
    sys.path.insert(0, str(_HERE))
    try:
        from autosat.llm_api.base_api import ElizaCallAPI
    except ImportError as e:
        print(f"[AutoSAT] Cannot import ElizaCallAPI: {e} – skipping", flush=True)
        return False

    print("\n[Test 4] AutoSAT ElizaCallAPI (structured output) ...", flush=True)

    import tempfile
    import textwrap
    prompt_text = textwrap.dedent("""
        You are optimizing a SAT solver heuristic.
        Provide a trivial C++ bump_var function body that just returns.
        The code must start with 'void Solver::bump_var' and end with '}'.
    """).strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(prompt_text)
        tmp_path = tf.name

    try:
        api = ElizaCallAPI(api_base="", api_key=SOY_TOKEN, model_name=MODEL)
        code, lbd = api.call_api_structured(prompt_file=tmp_path, temperature=0.7)
        print(f"[AutoSAT] code (first 200 chars): {code[:200]!r}", flush=True)
        print(f"[AutoSAT] lbd_queue_size: {lbd!r}", flush=True)
        return True
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}

    for name, fn in [
        ("plain", test_plain),
        ("system", test_with_system),
        ("tool_use", test_tool_use),
        ("autosat", test_autosat_eliza),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            if hasattr(e, "response") and e.response is not None:
                print(f"[{name}] Status: {e.response.status_code}", flush=True)
                print(f"[{name}] Body: {e.response.text[:500]}", flush=True)
            results[name] = False

    print("\n" + "=" * 50, flush=True)
    print("Results:", flush=True)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL/SKIP"
        print(f"  {name:12s} {status}", flush=True)

    if not any(results.values()):
        sys.exit(1)
