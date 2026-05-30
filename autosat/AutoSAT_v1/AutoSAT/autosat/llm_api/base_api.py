import os
import json
import time

import openai


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------
_STRUCTURED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "heuristic_code_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The complete C++ heuristic function body to replace "
                        "the original code between // start and // end markers."
                    ),
                },
                "lbd_queue_size": {
                    "type": "string",
                    "description": (
                        "Integer value for lbd_queue_size if applicable, "
                        "otherwise empty string."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief rationale for the proposed heuristic.",
                },
            },
            "required": ["code", "lbd_queue_size", "explanation"],
            "additionalProperties": False,
        },
    },
}

_JSON_OBJECT_FORMAT = {"type": "json_object"}


def _parse_structured_response(content: str):
    """
    Parse a JSON response from the model.
    Returns (code, lbd_queue_size) strings.
    Falls back to ("", "") on any parse error.
    """
    try:
        data = json.loads(content)
        code = str(data.get("code", "")).strip()
        lbd = str(data.get("lbd_queue_size", "")).strip()
        return code, lbd
    except Exception as exc:
        print(f"[StructuredOutput] JSON parse error: {exc}. Raw content:\n{content[:300]}", flush=True)
        return "", ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseCallAPI:
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.model_name = model_name
        self._cum_prompt_tokens = 0
        self._cum_completion_tokens = 0
        self._cum_total_tokens = 0
        self._cum_calls = 0
        self._retry_sleep_seconds = max(1, int(os.getenv("AUTOSAT_API_RETRY_SECONDS", "10")))
        self._max_retries = int(os.getenv("AUTOSAT_API_MAX_RETRIES", "0"))
        self._structured_output = os.getenv("AUTOSAT_STRUCTURED_OUTPUT", "1").strip() not in ("0", "false", "False", "no")

    def _log_token_usage(self, usage):
        if usage is None:
            return
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
        try:
            prompt_tokens = int(prompt_tokens or 0)
            completion_tokens = int(completion_tokens or 0)
            total_tokens = int(total_tokens or 0)
        except Exception:
            return
        self._cum_calls += 1
        self._cum_prompt_tokens += prompt_tokens
        self._cum_completion_tokens += completion_tokens
        self._cum_total_tokens += total_tokens
        print(
            f"[TokenUsage] model={self.model_name} "
            f"call_prompt={prompt_tokens} call_completion={completion_tokens} call_total={total_tokens} "
            f"cum_calls={self._cum_calls} cum_prompt={self._cum_prompt_tokens} "
            f"cum_completion={self._cum_completion_tokens} cum_total={self._cum_total_tokens}",
            flush=True,
        )

    def load_prompt(self, file_dir):
        with open(file_dir, "r") as file:
            return file.read()

    def _call_with_retries(self, request_fn, description="API call"):
        attempt = 0
        while True:
            try:
                return request_fn()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                attempt += 1
                if self._max_retries > 0 and attempt > self._max_retries:
                    print(f"[{description}] failed after {attempt - 1} retries: {exc}", flush=True)
                    raise
                print(
                    f"[{description}] error on attempt {attempt}: {exc}. "
                    f"Retrying in {self._retry_sleep_seconds} seconds...",
                    flush=True,
                )
                time.sleep(self._retry_sleep_seconds)

    def _build_structured_system_prompt(self):
        return (
            "You are an expert C++ SAT solver engineer. "
            "You MUST respond with a valid JSON object matching the required schema. "
            "The 'code' field must contain only the C++ function body (no markdown fences). "
            "The 'lbd_queue_size' field must be a digit string (e.g. '50') or empty string. "
            "The 'explanation' field is a brief rationale."
        )

    def call_api(self, prompt, temperature):
        pass

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GPT / OpenAI-compatible API
# ---------------------------------------------------------------------------

class GPTCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super().__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file, temperature=0.2):
        with open(prompt_file, "r") as file:
            prompt = file.read()
        if self._structured_output:
            return self._call_structured_raw(prompt, temperature)
        response = self._call_with_retries(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
            ),
            description="GPTCallAPI.call_api",
        )
        self._log_token_usage(response.get("usage", None))
        return response["choices"][0]["message"]["content"]

    def _call_structured_raw(self, prompt: str, temperature: float) -> str:
        system_msg = self._build_structured_system_prompt()

        def _request_with_schema():
            return openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
                response_format=_STRUCTURED_SCHEMA,
            )

        def _request_json_object():
            return openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
                response_format=_JSON_OBJECT_FORMAT,
            )

        for attempt_fn, label in [
            (_request_with_schema, "json_schema"),
            (_request_json_object, "json_object"),
        ]:
            try:
                response = self._call_with_retries(attempt_fn, description=f"GPTCallAPI.structured[{label}]")
                self._log_token_usage(response.get("usage", None))
                content = response["choices"][0]["message"]["content"]
                code, _ = _parse_structured_response(content)
                if code:
                    print(f"[StructuredOutput] Success via {label}", flush=True)
                    return content
                print(f"[StructuredOutput] Empty code via {label}, trying next.", flush=True)
            except Exception as exc:
                print(f"[StructuredOutput] {label} failed: {exc}. Trying next.", flush=True)

        print("[StructuredOutput] Falling back to plain text mode.", flush=True)
        response = self._call_with_retries(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a chatbot"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
            ),
            description="GPTCallAPI.call_api[fallback]",
        )
        self._log_token_usage(response.get("usage", None))
        return response["choices"][0]["message"]["content"]

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0):
        with open(prompt_file, "r") as f:
            prompt = f.read()
        raw = self._call_structured_raw(prompt, temperature)
        return _parse_structured_response(raw)


# ---------------------------------------------------------------------------
# Local / vLLM API
# ---------------------------------------------------------------------------

class LocalCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super().__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file, temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"
        with open(prompt_file, "r") as file:
            prompt = file.read()
        response = self._call_with_retries(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stop=stop_tokens,
            ),
            description="LocalCallAPI.call_api",
        )
        self._log_token_usage(response.get("usage", None))
        return response["choices"][0]["message"]["content"]

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0):
        content = self.call_api(prompt_file, temperature)
        return _parse_structured_response(content)


# ---------------------------------------------------------------------------
# Eliza API  (internal Yandex gateway — native Anthropic Messages protocol)
# Uses raw HTTP requests only, exactly as shown in official Eliza docs.
# https://wiki.yandex-team.ru/eliza/vendors/claude/
# ---------------------------------------------------------------------------

class ElizaCallAPI(BaseCallAPI):
    """
    Client for the Eliza LLM gateway using raw HTTP requests (no anthropic SDK).

    Per official Eliza docs (https://wiki.yandex-team.ru/eliza/vendors/claude/):
      - Endpoint:  https://api.eliza.yandex.net/anthropic/v1/messages
      - Auth:      Authorization: OAuth <SOY_TOKEN>  (trailing space intentional)
      - SSL:       self-signed corp cert — use verify=False
      - No anthropic-version header needed — Eliza handles it internally.

    Activate in config or environment:
        llm_model: "claude-sonnet-4-6"

    Set in .env:
        AUTOSAT_API_TYPE=eliza
        AUTOSAT_API_KEY=<your_soy_oauth_token>
        AUTOSAT_LLM_MODEL=claude-sonnet-4-6   # optional

    Available models (from GET https://api.eliza.yandex.net/anthropic/v1/models):
        claude-sonnet-4-6          (recommended, supports tool_use + structured_outputs)
        claude-opus-4-7            (most powerful)
        claude-opus-4-6
        claude-sonnet-4-5-20250929
        claude-haiku-4-5-20251001
        claude-opus-4-5-20251101
        claude-opus-4-1-20250805
        claude-opus-4-20250514
        claude-sonnet-4-20250514
    """

    _MESSAGES_URL = "https://api.eliza.yandex.net/anthropic/v1/messages"
    _DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_base: str, api_key: str, model_name: str):
        super().__init__(api_base, api_key, model_name)
        self._soy_token = api_key.strip()
        if not self._soy_token:
            raise ValueError(
                "ElizaCallAPI requires a SOY OAuth token. "
                "Set AUTOSAT_API_KEY=<your_token> in .env"
            )
        # Suppress InsecureRequestWarning once at init time
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_headers(self) -> dict:
        """Build request headers per official Eliza docs.

        The trailing space after the token in the Authorization header is
        intentional — it matches the official curl/python examples in the wiki.
        No anthropic-version header is needed; Eliza adds it internally.
        """
        return {
            "authorization": f"OAuth {self._soy_token} ",
            "content-type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        """POST to Eliza Messages endpoint with verify=False (per official docs).

        Eliza wraps the actual Anthropic response in a 'response' key:
            {"key": "...", "response": {<actual Anthropic response>}, ...}
        This method unwraps it so callers always get the raw Anthropic response dict.
        """
        import requests as _requests
        resp = _requests.post(
            self._MESSAGES_URL,
            json=payload,
            headers=self._make_headers(),
            verify=False,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Unwrap Eliza envelope: actual Anthropic response is under "response" key
        if "response" in data and isinstance(data["response"], dict):
            return data["response"]
        return data

    def _extract_text(self, data: dict) -> str:
        """Extract text content from an Anthropic Messages API response dict."""
        content = data.get("content", [])
        parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return "\n".join(parts)

    def _log_usage_from_response(self, data: dict):
        usage = data.get("usage", {})
        self._log_token_usage({
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        })

    # ------------------------------------------------------------------
    # Plain text call
    # ------------------------------------------------------------------

    def _call_plain(self, system_msg: str, user_msg: str, temperature: float) -> str:
        """Call Eliza with a plain text prompt. Returns the response text."""
        payload = {
            "model": self.model_name,
            "max_tokens": self._DEFAULT_MAX_TOKENS,
            "system": system_msg,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": temperature,
        }

        def _request():
            return self._post(payload)

        data = self._call_with_retries(_request, description="ElizaCallAPI[plain]")
        self._log_usage_from_response(data)
        return self._extract_text(data)

    # ------------------------------------------------------------------
    # Structured output via tool_use
    # ------------------------------------------------------------------

    def _call_structured_raw(self, prompt: str, temperature: float) -> str:
        """
        Request structured output from Claude via Eliza using tool_use.
        Returns a JSON string with keys: code, lbd_queue_size, explanation.
        Falls back to plain text if tool_use is not returned.
        """
        tool_schema = {
            "name": "emit_heuristic",
            "description": (
                "Emit the optimized C++ heuristic code and metadata. "
                "Always call this tool with your answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete C++ heuristic function body to replace "
                            "the original code between // start and // end markers."
                        ),
                    },
                    "lbd_queue_size": {
                        "type": "string",
                        "description": "Integer value for lbd_queue_size if applicable, else empty string.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief rationale for the proposed heuristic.",
                    },
                },
                "required": ["code", "lbd_queue_size", "explanation"],
            },
        }

        payload = {
            "model": self.model_name,
            "max_tokens": self._DEFAULT_MAX_TOKENS,
            "system": self._build_structured_system_prompt(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "tools": [tool_schema],
            "tool_choice": {"type": "any"},
        }

        def _request():
            return self._post(payload)

        try:
            data = self._call_with_retries(_request, description="ElizaCallAPI[structured]")
            self._log_usage_from_response(data)
            # Extract tool_use block from content
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    return json.dumps(block.get("input", {}))
            print("[StructuredOutput][Eliza] No tool_use block in response, falling back to plain text.", flush=True)
        except Exception as exc:
            print(f"[StructuredOutput][Eliza] tool_use failed: {exc}. Falling back to plain text.", flush=True)

        # Fallback: plain text
        return self._call_plain(
            system_msg="You are a chatbot",
            user_msg=prompt,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call_api(self, prompt_file: str, temperature: float = 1.0) -> str:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        if self._structured_output:
            return self._call_structured_raw(prompt, temperature)
        return self._call_plain(
            system_msg="You are a chatbot",
            user_msg=prompt,
            temperature=temperature,
        )

    def call_api_structured(self, prompt_file: str, temperature: float = 1.0):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        raw = self._call_structured_raw(prompt, temperature)
        return _parse_structured_response(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_api(args):
    model_name = str(getattr(args, "llm_model", "") or "").strip()
    api_base = str(getattr(args, "api_base", "") or "").strip()
    api_key = str(getattr(args, "api_key", "") or "").strip()
    api_type = str(os.getenv("AUTOSAT_API_TYPE", "") or "").strip().lower()

    if api_type == "eliza":
        resolved_model = model_name.removeprefix("eliza/") if model_name.startswith("eliza/") else model_name
        if not resolved_model:
            raise ValueError(
                "Eliza API requires a model name. "
                "Set AUTOSAT_LLM_MODEL=<model> or llm_model in config"
            )
        if not api_key:
            raise ValueError(
                "Eliza API requires a SOY OAuth token. "
                "Set AUTOSAT_API_KEY=<your_token> in .env"
            )
        print(f"[LLM] Using ElizaCallAPI: model={resolved_model}", flush=True)
        return ElizaCallAPI(api_base="", api_key=api_key, model_name=resolved_model)

    if (not api_base) and model_name.startswith("openai/"):
        api_base = os.getenv("AUTOSAT_API_BASE") or os.getenv("DEEPINFRA_API_BASE") or ""

    if api_base:
        return GPTCallAPI(api_base=api_base, api_key=api_key, model_name=model_name)

    if model_name in ("gpt-4-1106-preview", "gpt-3.5-turbo"):
        return GPTCallAPI(api_base=api_base, api_key=api_key, model_name=model_name)
    if model_name == "Qwen":
        return LocalCallAPI(
            api_base="http://172.26.1.16:31251/v1",
            api_key="sk-",
            model_name="modelscope/qwen/Qwen-72B-Chat",
        )
    if model_name == "llama":
        return LocalCallAPI(
            api_base="http://172.26.1.16:31251/v1",
            api_key="sk-",
            model_name="modelscope/modelscope/Llama-2-70b-chat-ms",
        )
    if model_name == "deepseek":
        return LocalCallAPI(
            api_base="http://172.26.1.16:31251/v1",
            api_key="sk-",
            model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct",
        )

    raise NotImplementedError(
        "Unsupported llm_model without external API endpoint. "
        "Set AUTOSAT_API_BASE/AUTOSAT_API_KEY for OpenAI-compatible models, "
        "set AUTOSAT_API_TYPE=eliza for Eliza, or use one of: Qwen, llama, deepseek."
    )


if __name__ == "__main__":
    llm_api = LocalCallAPI(
        api_base="http://172.26.1.16:31251/v1",
        api_key="sk-",
        model_name="modelscope/modelscope/Llama-2-70b-chat-ms",
    )
    answer = llm_api.call_api(prompt_file="../template/EasySAT/bump_var_function/original_prompt.txt")
    print(answer)
