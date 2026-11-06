"""
LLM CLIENTS: model registry + provider adapters (harness v2)
============================================================

One call = one query record + its blocked candidates -> one LLMResult.

Every adapter:
  * requests structured JSON output ({"match", "confidence", "rationale"})
    instead of free text, so answers never depend on line-prefix parsing;
  * returns a row-level `status` so API failures, unparseable answers,
    refusals and truncations are never silently scored as "no match":
        ok         - model picked a candidate
        no_match   - model answered match = 0
        parse_fail - response arrived but could not be parsed/validated
        truncated  - hit the output-token cap before a complete answer
        refusal    - model/provider declined
        api_error  - request failed (after the SDK's own retries)
  * logs raw response text, stop reason, token usage, cost, served model
    and request id.

Fatal account-level errors (bad key, no credit, unknown model) raise
FatalAPIError so a run stops instead of writing thousands of blank rows.

Sampling parameters (temperature etc.) are NOT sent: current Claude models
reject them, and current OpenAI reasoning models do not support them.
Run-to-run variance must therefore be measured with repeated runs.

No API call happens on import; clients are created lazily.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv

# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {
            "type": "integer",
            "description": "Number of the candidate that is the same entity as the query, or 0 if none is.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the decision, from 0.0 to 1.0.",
        },
        "rationale": {
            "type": "string",
            "description": "One short sentence naming the evidence used.",
        },
    },
    "required": ["match", "confidence", "rationale"],
    "additionalProperties": False,
}

# =============================================================================
# MODEL REGISTRY
# =============================================================================
# Prices are USD per 1M tokens (standard tier, short context), checked 2026-11-06:
#   Claude: claude-api skill model table (cached 2026-10-06)
#   OpenAI: https://developers.openai.com/api/docs/pricing
# Re-check before budgeting. Output tokens include thinking/reasoning tokens.


@dataclass(frozen=True)
class ModelConfig:
    key: str                 # short name used in method names / result dirs
    provider: str            # "anthropic" | "openai" | "openweight"
    model_id: str            # exact API model string
    label: str               # display label for tables/figures
    input_price: float       # $ / 1M input tokens
    output_price: float      # $ / 1M output tokens
    effort: Optional[str]    # default reasoning effort; None = do not send
    max_output_tokens: int   # cap incl. thinking tokens
    tier: str                # "small" | "mid" | "flagship" | "bridge" | "open"
    full_run: bool = True    # False = run on a subsample by default (cost)


MODELS = {
    # --- Anthropic (Claude 5.5 family). Thinking is adaptive by default on all
    # three; Opus 5.5 cannot disable it, so effort is the only control.
    "haiku55": ModelConfig("haiku55", "anthropic", "claude-haiku-5-5", "Claude Haiku 5.5",
                           0.10, 0.50, "medium", 8000, "small"),
    "sonnet55": ModelConfig("sonnet55", "anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5",
                            2.00, 10.00, "medium", 8000, "mid"),
    "opus55": ModelConfig("opus55", "anthropic", "claude-opus-5-5", "Claude Opus 5.5",
                          4.00, 20.00, "medium", 8000, "flagship", full_run=False),
    # --- OpenAI (GPT-6 family; all reasoning models).
    "luna": ModelConfig("luna", "openai", "gpt-6-luna", "GPT-6 Luna",
                        0.10, 0.50, "medium", 8000, "small"),
    "sol": ModelConfig("sol", "openai", "gpt-6.1-sol", "GPT-6.1 Sol",
                       2.00, 10.00, "medium", 8000, "mid"),
    "astra": ModelConfig("astra", "openai", "gpt-6-astra", "GPT-6 Astra",
                         10.00, 50.00, "medium", 8000, "flagship", full_run=False),
    # --- Bridge to the TKDE submission: same model, fixed harness.
    "gpt4omini": ModelConfig("gpt4omini", "openai", "gpt-4o-mini", "GPT-4o-mini (harness v2)",
                             0.15, 0.60, None, 1024, "bridge"),
    # --- Open-weight model behind any OpenAI-compatible server (vLLM, Ollama,
    # OpenRouter, Together...). Configure with env vars:
    #   OPENWEIGHT_BASE_URL, OPENWEIGHT_MODEL, OPENWEIGHT_API_KEY (optional),
    #   OPENWEIGHT_INPUT_PRICE / OPENWEIGHT_OUTPUT_PRICE (optional, $/1M; 0 for local)
    "openweight": ModelConfig("openweight", "openweight",
                              os.environ.get("OPENWEIGHT_MODEL", "UNSET"),
                              f"Open-weight ({os.environ.get('OPENWEIGHT_MODEL', 'unset')})",
                              float(os.environ.get("OPENWEIGHT_INPUT_PRICE", "0")),
                              float(os.environ.get("OPENWEIGHT_OUTPUT_PRICE", "0")),
                              None, 4000, "open"),
}

# =============================================================================
# RESULT TYPE + ERRORS
# =============================================================================


class FatalAPIError(RuntimeError):
    """Account/config-level failure: every further call would fail too."""


@dataclass
class LLMResult:
    status: str
    match_index: Optional[int] = None     # 1-based candidate index, 0 = no match
    confidence: float = 0.0
    rationale: str = ""
    raw_response: str = ""
    stop_reason: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_served: str = ""
    request_id: str = ""
    latency_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def _cost(cfg: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * cfg.input_price + output_tokens / 1e6 * cfg.output_price


# =============================================================================
# PARSING (shared by all providers)
# =============================================================================

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_match_json(text: str, n_candidates: int):
    """Validate a structured answer. Returns (status, match_index, confidence, rationale, error)."""
    if not text or not text.strip():
        return "parse_fail", None, 0.0, "", "empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Lenient fallback for servers that ignore response_format.
        m = _JSON_OBJ.search(text)
        if not m:
            return "parse_fail", None, 0.0, "", "no JSON object"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return "parse_fail", None, 0.0, "", f"invalid JSON: {e}"
    if not isinstance(data, dict) or "match" not in data:
        return "parse_fail", None, 0.0, "", "missing 'match'"
    try:
        match = int(data["match"])
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as e:
        return "parse_fail", None, 0.0, "", f"bad field type: {e}"
    if 1.0 < conf <= 100.0:          # model answered on a 0-100 scale
        conf /= 100.0
    conf = min(max(conf, 0.0), 1.0)
    rationale = str(data.get("rationale", ""))[:500]
    if match < 0 or match > n_candidates:
        return "parse_fail", None, conf, rationale, f"match {match} out of range 0..{n_candidates}"
    return ("no_match" if match == 0 else "ok"), match, conf, rationale, ""


# =============================================================================
# CLIENTS (lazy)
# =============================================================================

_clients = {}


def _client(provider: str):
    if provider in _clients:
        return _clients[provider]
    load_dotenv()
    if provider == "anthropic":
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise FatalAPIError("ANTHROPIC_API_KEY not set (.env or environment)")
        c = anthropic.Anthropic(max_retries=6, timeout=300.0)
    elif provider == "openai":
        import openai
        if not os.getenv("OPENAI_API_KEY"):
            raise FatalAPIError("OPENAI_API_KEY not set (.env or environment)")
        c = openai.OpenAI(max_retries=6, timeout=300.0)
    elif provider == "openweight":
        import openai
        base_url = os.getenv("OPENWEIGHT_BASE_URL")
        if not base_url or os.getenv("OPENWEIGHT_MODEL", "UNSET") == "UNSET":
            raise FatalAPIError("Set OPENWEIGHT_BASE_URL and OPENWEIGHT_MODEL for the open-weight model")
        c = openai.OpenAI(base_url=base_url, api_key=os.getenv("OPENWEIGHT_API_KEY", "none"),
                          max_retries=6, timeout=600.0)
    else:
        raise ValueError(f"unknown provider {provider}")
    _clients[provider] = c
    return c


# =============================================================================
# PROVIDER CALLS
# =============================================================================

_FATAL_HINTS = ("credit balance", "billing", "insufficient_quota", "quota", "invalid x-api-key",
                "incorrect api key", "model_not_found", "does not exist")


def _is_fatal(err: Exception) -> bool:
    name = type(err).__name__
    if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
        return True
    msg = str(err).lower()
    return any(h in msg for h in _FATAL_HINTS)


def _call_anthropic(cfg: ModelConfig, prompt: str, effort: Optional[str], n_candidates: int) -> LLMResult:
    import anthropic
    client = _client("anthropic")
    output_config = {"format": {"type": "json_schema", "schema": MATCH_SCHEMA}}
    if effort:
        output_config["effort"] = effort
    t0 = time.time()
    try:
        # No `thinking` param: adaptive thinking is the default on the 5.5 family.
        # No `fallbacks`: a silent switch to another model would contaminate
        # per-model results, so refusals are logged as status="refusal" instead.
        resp = client.messages.create(
            model=cfg.model_id,
            max_tokens=cfg.max_output_tokens,
            output_config=output_config,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        if _is_fatal(e):
            raise FatalAPIError(f"{type(e).__name__}: {e}") from e
        return LLMResult("api_error", error=f"{type(e).__name__}: {e}"[:1000], latency_s=time.time() - t0)

    text = "".join(b.text for b in resp.content if b.type == "text")
    inp, out = resp.usage.input_tokens, resp.usage.output_tokens
    base = dict(raw_response=text, stop_reason=str(resp.stop_reason), input_tokens=inp,
                output_tokens=out, cost_usd=_cost(cfg, inp, out), model_served=resp.model,
                request_id=str(getattr(resp, "_request_id", "") or ""), latency_s=time.time() - t0)
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return LLMResult("refusal", error=f"refusal category={cat}", **base)
    if resp.stop_reason == "max_tokens":
        return LLMResult("truncated", error="hit max_tokens", **base)
    status, idx, conf, rat, err = parse_match_json(text, n_candidates)
    return LLMResult(status, idx, conf, rat, error=err, **base)


def _call_openai(cfg: ModelConfig, prompt: str, effort: Optional[str], n_candidates: int) -> LLMResult:
    import openai
    client = _client("openai")
    kwargs = dict(
        model=cfg.model_id,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=cfg.max_output_tokens,
        text={"format": {"type": "json_schema", "name": "entity_match",
                         "schema": MATCH_SCHEMA, "strict": True}},
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    t0 = time.time()
    try:
        resp = client.responses.create(**kwargs)
    except openai.APIError as e:
        if _is_fatal(e):
            raise FatalAPIError(f"{type(e).__name__}: {e}") from e
        return LLMResult("api_error", error=f"{type(e).__name__}: {e}"[:1000], latency_s=time.time() - t0)

    usage = resp.usage
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    text = resp.output_text or ""
    base = dict(raw_response=text, stop_reason=str(resp.status), input_tokens=inp, output_tokens=out,
                cost_usd=_cost(cfg, inp, out), model_served=str(resp.model),
                request_id=str(resp.id), latency_s=time.time() - t0)
    # Refusals arrive as a content part of type "refusal" and do not follow the schema.
    for item in resp.output or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", "") == "refusal":
                return LLMResult("refusal", error=str(getattr(part, "refusal", ""))[:500], **base)
    if resp.status == "incomplete":
        reason = getattr(resp.incomplete_details, "reason", "") if resp.incomplete_details else ""
        status = "truncated" if reason == "max_output_tokens" else "refusal"
        return LLMResult(status, error=f"incomplete: {reason}", **base)
    status, idx, conf, rat, err = parse_match_json(text, n_candidates)
    return LLMResult(status, idx, conf, rat, error=err, **base)


def _call_openweight(cfg: ModelConfig, prompt: str, effort: Optional[str], n_candidates: int) -> LLMResult:
    import openai
    client = _client("openweight")
    t0 = time.time()
    try:
        # Chat Completions: the most widely implemented OpenAI-compatible endpoint.
        resp = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.max_output_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "entity_match", "schema": MATCH_SCHEMA, "strict": True}},
        )
    except openai.APIError as e:
        if _is_fatal(e):
            raise FatalAPIError(f"{type(e).__name__}: {e}") from e
        return LLMResult("api_error", error=f"{type(e).__name__}: {e}"[:1000], latency_s=time.time() - t0)
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = resp.usage
    inp = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    out = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    base = dict(raw_response=text, stop_reason=str(choice.finish_reason), input_tokens=inp,
                output_tokens=out, cost_usd=_cost(cfg, inp, out), model_served=str(resp.model),
                request_id=str(resp.id), latency_s=time.time() - t0)
    if choice.finish_reason == "length":
        return LLMResult("truncated", error="hit max_tokens", **base)
    status, idx, conf, rat, err = parse_match_json(text, n_candidates)
    return LLMResult(status, idx, conf, rat, error=err, **base)


_CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai, "openweight": _call_openweight}


def call_model(cfg: ModelConfig, prompt: str, n_candidates: int, effort: Optional[str] = None) -> LLMResult:
    """Single structured call. `effort` overrides the model default; pass "" to send none."""
    eff = cfg.effort if effort is None else (effort or None)
    if cfg.provider == "anthropic" and eff == "none":
        raise FatalAPIError("Claude 5.5 models cannot run with effort 'none' (thinking cannot be "
                            "disabled); use 'low' as the minimum-reasoning condition")
    return _CALLERS[cfg.provider](cfg, prompt, eff, n_candidates)
