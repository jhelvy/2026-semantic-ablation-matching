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

Providers: Anthropic Messages API, OpenAI Responses API, any OpenAI-compatible
Chat Completions server (no model in the current lineup uses it; it was built
for Kimi K3, dropped 2026-11-09), and the Claude Code CLI on a subscription
(pilots only).

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

from .paths import ENV_FILE

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
# Lineup for the reported results (proof of concept, not a model survey;
# revised 2026-11-09: GPT-4o-mini replaced by GPT-6 Luna, Kimi K3 dropped):
#   haiku55  - Claude Haiku 5.5, full grid (small)
#   gpt6luna - GPT-6 Luna, full grid (small; same price tier as Haiku 5.5)
#   sonnet55 - Claude Sonnet 5.5, fixed 30% subsample (mid)
# The subsample is a fixed set of query records (one seed), so Sonnet's
# records are a subset of the Haiku / Luna runs. All three reason at effort
# "medium" by default.
#
# *_cli entries run through the Claude Code CLI on a Claude subscription. They
# are for pilots and prompt iteration only and write to pilot_results/, never
# to results/.
#
# Prices are USD per 1M tokens, checked 2026-11-06; re-check before budgeting.
#   Claude: claude-api skill model table (cached 2026-10-06)
#   OpenAI: https://developers.openai.com/api/docs/pricing (gpt-6-luna re-checked 2026-11-09)
# Output tokens include thinking/reasoning tokens.

SUBSAMPLE_FRAC = 0.3


@dataclass(frozen=True)
class ModelConfig:
    key: str                 # short name used in method names / result dirs
    provider: str            # "anthropic" | "openai" | "openai_compatible" | "claude_cli"
    model_id: str            # exact API model string
    label: str               # display label for tables/figures
    input_price: float       # $ / 1M input tokens
    output_price: float      # $ / 1M output tokens
    effort: Optional[str]    # default reasoning effort; None = do not send
    max_output_tokens: int   # cap incl. thinking tokens
    tier: str                # "small" | "mid" | "pilot"
    default_sample_frac: Optional[float] = None   # None = full grid
    base_url: Optional[str] = None                # openai_compatible only
    api_key_env: Optional[str] = None             # openai_compatible only
    extra_body: Optional[dict] = None             # openai_compatible only
    response_format: str = "json_schema"          # openai_compatible: json_schema | json_object | none


MODELS = {
    # --- Anthropic. Thinking is adaptive by default on the 5.5 family.
    "haiku55": ModelConfig("haiku55", "anthropic", "claude-haiku-5-5", "Claude Haiku 5.5",
                           0.10, 0.50, "medium", 8000, "small"),
    "sonnet55": ModelConfig("sonnet55", "anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5",
                            2.00, 10.00, "medium", 8000, "mid", default_sample_frac=SUBSAMPLE_FRAC),
    # --- OpenAI small model. Reasoning effort matches the Claude models.
    "gpt6luna": ModelConfig("gpt6luna", "openai", "gpt-6-luna", "GPT-6 Luna",
                            0.10, 0.50, "medium", 8000, "small"),
    # --- Pilot-only: Claude Code CLI on a subscription (no API billing). Same
    # prompts and schema, but a different harness (CLI system prompt, CLI
    # defaults), so these numbers are never reported.
    "haiku55_cli": ModelConfig("haiku55_cli", "claude_cli", "claude-haiku-5-5",
                               "Claude Haiku 5.5 (CLI pilot)", 0.0, 0.0, "medium", 8000, "pilot"),
    "sonnet55_cli": ModelConfig("sonnet55_cli", "claude_cli", "claude-sonnet-5-5",
                                "Claude Sonnet 5.5 (CLI pilot)", 0.0, 0.0, "medium", 8000, "pilot"),
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
    load_dotenv(ENV_FILE)
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
    elif provider.startswith("openai_compatible:"):
        import openai
        _, base_url, key_env = provider.split("|")   # "openai_compatible:|<url>|<KEY_ENV>"
        if not os.getenv(key_env):
            raise FatalAPIError(f"{key_env} not set (.env or environment)")
        c = openai.OpenAI(base_url=base_url, api_key=os.getenv(key_env), max_retries=6, timeout=900.0)
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


def _call_openai_compatible(cfg: ModelConfig, prompt: str, effort: Optional[str], n_candidates: int) -> LLMResult:
    import openai
    client = _client(f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}")
    kwargs = dict(model=cfg.model_id, messages=[{"role": "user", "content": prompt}],
                  max_tokens=cfg.max_output_tokens)
    if cfg.response_format == "json_schema":
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "entity_match", "schema": MATCH_SCHEMA, "strict": True}}
    elif cfg.response_format == "json_object":   # schema then rests on the prompt + parser
        kwargs["response_format"] = {"type": "json_object"}
    extra = dict(cfg.extra_body or {})
    if effort:
        extra["reasoning"] = {"effort": effort}   # OpenRouter's unified reasoning field
    if extra:
        kwargs["extra_body"] = extra
    t0 = time.time()
    try:
        # Chat Completions: the most widely implemented OpenAI-compatible endpoint.
        resp = client.chat.completions.create(**kwargs)
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
    if getattr(choice.message, "refusal", None):
        return LLMResult("refusal", error=str(choice.message.refusal)[:500], **base)
    if choice.finish_reason == "length":
        return LLMResult("truncated", error="hit max_tokens", **base)
    if choice.finish_reason == "content_filter":
        return LLMResult("refusal", error="content_filter", **base)
    status, idx, conf, rat, err = parse_match_json(text, n_candidates)
    return LLMResult(status, idx, conf, rat, error=err, **base)


# ---- Claude Code CLI (subscription; PILOT ONLY) -----------------------------
# Isolation: empty temp working directory (no project CLAUDE.md or memory),
# only "project" setting sources (so user-level hooks and plugins are not
# loaded), no tools, no MCP servers, no slash commands, replaced system prompt.
# ANTHROPIC_API_KEY is removed from the child environment so the CLI uses the
# subscription login and can never bill the API.
# UNVERIFIED until the first pilot: the JSON envelope field names
# ("structured_output", "result", "usage", "is_error", "subtype") and whether
# any user-level context is still injected. Compare logged input_tokens with an
# API count of the same prompt; a large excess means extra context got in.

CLI_SYSTEM_PROMPT = "You are an expert at entity matching. Answer only in the requested JSON format."
_CLI_FATAL_HINTS = ("usage limit", "rate limit", "limit reached", "not logged in", "please run /login",
                    "authentication", "invalid api key")


def _call_claude_cli(cfg: ModelConfig, prompt: str, effort: Optional[str], n_candidates: int) -> LLMResult:
    import shutil
    import subprocess
    import tempfile
    exe = shutil.which("claude")
    if not exe:
        raise FatalAPIError("claude CLI not found on PATH")
    cmd = [exe, "-p", "--model", cfg.model_id, "--output-format", "json",
           "--json-schema", json.dumps(MATCH_SCHEMA), "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "project",
           "--system-prompt", CLI_SYSTEM_PROMPT]
    if effort:
        cmd += ["--effort", effort]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  cwd=cwd, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            return LLMResult("api_error", error="CLI timeout", latency_s=time.time() - t0)
    latency = time.time() - t0
    blob = (proc.stdout or "") + (proc.stderr or "")
    try:
        envl = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        envl = None
    if proc.returncode != 0 or envl is None or envl.get("is_error"):
        msg = (envl or {}).get("result") or blob
        if any(h in str(msg).lower() for h in _CLI_FATAL_HINTS):
            raise FatalAPIError(f"claude CLI: {str(msg)[:300]}")
        return LLMResult("api_error", error=f"claude CLI rc={proc.returncode}: {str(msg)[:800]}",
                         latency_s=latency)
    structured = envl.get("structured_output")
    text = json.dumps(structured) if structured is not None else str(envl.get("result", ""))
    usage = envl.get("usage") or {}
    inp = sum(int(usage.get(k, 0) or 0) for k in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    base = dict(raw_response=text, stop_reason=str(envl.get("subtype", "")), input_tokens=inp,
                output_tokens=int(usage.get("output_tokens", 0) or 0), cost_usd=0.0,
                model_served=cfg.model_id, request_id=str(envl.get("session_id", "")),
                latency_s=latency)
    if envl.get("subtype") not in (None, "success"):
        return LLMResult("parse_fail", error=f"CLI subtype {envl.get('subtype')}", **base)
    status, idx, conf, rat, err = parse_match_json(text, n_candidates)
    return LLMResult(status, idx, conf, rat, error=err, **base)


_CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai,
            "openai_compatible": _call_openai_compatible, "claude_cli": _call_claude_cli}


def call_model(cfg: ModelConfig, prompt: str, n_candidates: int, effort: Optional[str] = None) -> LLMResult:
    """Single structured call. `effort` overrides the model default; pass "" to send none."""
    eff = cfg.effort if effort is None else (effort or None)
    if cfg.provider in ("anthropic", "claude_cli") and eff == "none":
        raise FatalAPIError("Claude 5.5 models cannot run with effort 'none' (thinking cannot be "
                            "disabled); use 'low' as the minimum-reasoning condition")
    return _CALLERS[cfg.provider](cfg, prompt, eff, n_candidates)
