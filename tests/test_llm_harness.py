"""
Offline tests for harness v2 (no network, no API keys, no cost).

    uv run tests/test_llm_harness.py        # or: uv run pytest tests/test_llm_harness.py
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# methods.py imports sentence_transformers at module load; stub it if absent so
# these tests run in a light environment. (No embedding code is exercised here.)
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=object)

import pandas as pd  # noqa: E402

from lib import llm_clients  # noqa: E402
from lib import llm_runner  # noqa: E402
from lib.llm_clients import MODELS, LLMResult, FatalAPIError, parse_match_json  # noqa: E402


# ---------------------------------------------------------------- parsing
def test_parse_ok_and_no_match():
    assert parse_match_json('{"match": 3, "confidence": 0.9, "rationale": "x"}', 50)[:3] == ("ok", 3, 0.9)
    assert parse_match_json('{"match": 0, "confidence": 0.7, "rationale": "x"}', 50)[:2] == ("no_match", 0)


def test_parse_failures():
    assert parse_match_json("", 50)[0] == "parse_fail"
    assert parse_match_json("Match: 3\nConfidence: 0.9", 50)[0] == "parse_fail"   # v1 free text
    assert parse_match_json('{"match": 51, "confidence": 0.9}', 50)[0] == "parse_fail"
    assert parse_match_json('{"confidence": 0.9}', 50)[0] == "parse_fail"


def test_parse_lenient_and_scale():
    s, i, c, _, _ = parse_match_json('Sure: {"match": 2, "confidence": 85, "rationale": "y"} done', 50)
    assert (s, i) == ("ok", 2) and abs(c - 0.85) < 1e-9


# ---------------------------------------------------------------- anthropic adapter
class _FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp


def _anthropic_resp(text, stop="end_turn"):
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
              usage=NS(input_tokens=1000, output_tokens=200), stop_reason=stop,
              stop_details=None, model="claude-haiku-5-5", _request_id="req_1")


def test_anthropic_request_shape_and_ok():
    fake = _FakeMessages(_anthropic_resp('{"match": 4, "confidence": 0.8, "rationale": "sku"}'))
    llm_clients._clients["anthropic"] = NS(messages=fake)
    r = llm_clients.call_model(MODELS["haiku55"], "prompt", 50)
    assert r.status == "ok" and r.match_index == 4
    assert "temperature" not in fake.kwargs and "thinking" not in fake.kwargs
    assert fake.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert fake.kwargs["output_config"]["effort"] == "medium"
    assert abs(r.cost_usd - (1000 * 0.10 + 200 * 0.50) / 1e6) < 1e-12


def test_anthropic_truncated_and_refusal():
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(_anthropic_resp("", "max_tokens")))
    assert llm_clients.call_model(MODELS["haiku55"], "p", 50).status == "truncated"
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(_anthropic_resp("", "refusal")))
    assert llm_clients.call_model(MODELS["haiku55"], "p", 50).status == "refusal"


def test_anthropic_fatal_vs_row_error():
    import anthropic
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def err(cls, code, msg):
        return cls(msg, response=httpx.Response(code, request=req), body=None)

    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(
        exc=err(anthropic.BadRequestError, 400, "Your credit balance is too low")))
    try:
        llm_clients.call_model(MODELS["haiku55"], "p", 50)
        raise AssertionError("expected FatalAPIError")
    except FatalAPIError:
        pass
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(
        exc=err(anthropic.InternalServerError, 500, "overloaded")))
    assert llm_clients.call_model(MODELS["haiku55"], "p", 50).status == "api_error"


# ---------------------------------------------------------------- openai adapter
class _FakeResponses(_FakeMessages):
    pass


def _openai_resp(text, status="completed", reason=None, refusal=False):
    part = NS(type="refusal", refusal="no") if refusal else NS(type="output_text", text=text)
    return NS(output=[NS(type="message", content=[part])], output_text=text, status=status,
              incomplete_details=NS(reason=reason) if reason else None,
              usage=NS(input_tokens=1000, output_tokens=300), model="gpt-6-luna", id="resp_1")


def test_openai_request_shape_and_statuses():
    fake = _FakeResponses(_openai_resp('{"match": 0, "confidence": 0.6, "rationale": "none"}'))
    llm_clients._clients["openai"] = NS(responses=fake)
    r = llm_clients.call_model(MODELS["gpt6luna"], "p", 50)
    assert r.status == "no_match"
    assert fake.kwargs["text"]["format"]["strict"] is True
    assert fake.kwargs["reasoning"] == {"effort": "medium"}      # default matches the Claude models
    assert abs(r.cost_usd - (1000 * 0.10 + 300 * 0.50) / 1e6) < 1e-12
    llm_clients.call_model(MODELS["gpt6luna"], "p", 50, effort="low")
    assert fake.kwargs["reasoning"] == {"effort": "low"}         # explicit override is passed through
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(
        _openai_resp("", "incomplete", "max_output_tokens")))
    assert llm_clients.call_model(MODELS["gpt6luna"], "p", 50).status == "truncated"
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(_openai_resp("", refusal=True)))
    assert llm_clients.call_model(MODELS["gpt6luna"], "p", 50).status == "refusal"
    # A prompt flagged by OpenAI's usage-policy check (HTTP 400) is a refusal, not a retryable error;
    # other 400s stay api_error.
    import openai
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")

    def err(msg):
        return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)

    llm_clients._clients["openai"] = NS(responses=_FakeResponses(exc=err(
        "Error code: 400 - Invalid prompt: your prompt was flagged as potentially violating "
        "our usage policy. Please try again with a different prompt")))
    assert llm_clients.call_model(MODELS["gpt6luna"], "p", 50).status == "refusal"
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(exc=err("Error code: 400 - bad schema")))
    assert llm_clients.call_model(MODELS["gpt6luna"], "p", 50).status == "api_error"


# ---------------------------------------------------------------- openai-compatible adapter
class _FakeChat:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None
        self.completions = self

    def create(self, **kw):
        self.kwargs = kw
        return self.resp


def _chat_resp(text, finish="stop"):
    return NS(choices=[NS(message=NS(content=text, refusal=None), finish_reason=finish)],
              usage=NS(prompt_tokens=2000, completion_tokens=900), model="vendor/open-model", id="gen_1")


def test_openai_compatible_request_shape_and_statuses():
    # The adapter's OpenRouter defaults (provider pinning, unified reasoning field), tested
    # with a stand-in config; Kimi K3 uses the Moonshot settings (test_kimik3_* below).
    cfg = llm_clients.ModelConfig(
        "openmodel", "openai_compatible", "vendor/open-model", "Open model", 3.00, 15.00, None,
        16000, "mid", default_sample_frac=0.3, base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body={"provider": {"order": ["vendor"], "allow_fallbacks": False}})
    key = f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}"
    fake = _FakeChat(_chat_resp('{"match": 7, "confidence": 0.7, "rationale": "code"}'))
    llm_clients._clients[key] = NS(chat=fake)
    r = llm_clients.call_model(cfg, "p", 50)
    assert r.status == "ok" and r.match_index == 7
    assert fake.kwargs["response_format"]["type"] == "json_schema"
    assert fake.kwargs["extra_body"]["provider"] == {"order": ["vendor"], "allow_fallbacks": False}
    assert fake.kwargs["max_tokens"] == 16000 and "max_completion_tokens" not in fake.kwargs
    assert abs(r.cost_usd - (2000 * 3.0 + 900 * 15.0) / 1e6) < 1e-12
    assert cfg.default_sample_frac == 0.3
    llm_clients.call_model(cfg, "p", 50, effort="high")
    assert fake.kwargs["extra_body"]["reasoning"] == {"effort": "high"}   # OpenRouter style
    assert "reasoning_effort" not in fake.kwargs
    llm_clients._clients[key] = NS(chat=_FakeChat(_chat_resp("", "length")))
    assert llm_clients.call_model(cfg, "p", 50).status == "truncated"


def test_kimik3_config_and_request_shape():
    cfg = MODELS["kimik3"]
    assert (cfg.provider, cfg.model_id, cfg.tier) == ("openai_compatible", "kimi-k3", "open")
    assert cfg.base_url == "https://api.moonshot.ai/v1" and cfg.api_key_env == "KIMI_API_KEY"
    assert (cfg.input_price, cfg.output_price) == (3.00, 15.00)
    assert cfg.default_sample_frac == 0.3 and cfg.effort == "low"
    key = f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}"
    # Moonshot may wrap JSON in a Markdown fence; the lenient parser must still read it.
    fake = _FakeChat(_chat_resp('```json\n{"match": 4, "confidence": 0.8, "rationale": "model no"}\n```'))
    llm_clients._clients[key] = NS(chat=fake)
    r = llm_clients.call_model(cfg, "p", 50)
    assert r.status == "ok" and r.match_index == 4
    kw = fake.kwargs
    assert kw["model"] == "kimi-k3"
    assert kw["reasoning_effort"] == "low"                       # top-level, not OpenRouter's field
    assert kw["max_completion_tokens"] == 16000 and "max_tokens" not in kw
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert "extra_body" not in kw                                # no provider pinning, no reasoning dict
    for p in ("temperature", "top_p", "thinking"):               # fixed / unsupported on K3
        assert p not in kw
    llm_clients.call_model(cfg, "p", 50, effort="high")
    assert fake.kwargs["reasoning_effort"] == "high"
    llm_clients._clients[key] = NS(chat=_FakeChat(_chat_resp("", "length")))
    assert llm_clients.call_model(cfg, "p", 50).status == "truncated"


def test_kimik3_effort_guard_and_fatal_errors():
    import httpx2 as httpx
    import openai
    cfg = MODELS["kimik3"]
    key = f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}"
    fake = _FakeChat(_chat_resp("{}"))
    llm_clients._clients[key] = NS(chat=fake)
    for bad in ("medium", ""):                                   # "" would mean Moonshot's default, max
        try:
            llm_clients.call_model(cfg, "p", 50, effort=bad)
            raise AssertionError(f"effort {bad!r} should fail before any call")
        except FatalAPIError:
            pass
    assert fake.kwargs is None

    def raising(err):
        class _C:
            completions = None

            def create(self, **kw):
                raise err
        c = _C()
        c.completions = c
        return NS(chat=c)
    req = httpx.Request("POST", "https://api.moonshot.ai/v1/chat/completions")
    # Moonshot reports insufficient balance as HTTP 429 exceeded_current_quota_error: fatal.
    quota = openai.RateLimitError(
        "Error code: 429 - {'error': {'type': 'exceeded_current_quota_error', 'message': "
        "'Account balance is insufficient or the account has been disabled'}}",
        response=httpx.Response(429, request=req), body=None)
    llm_clients._clients[key] = raising(quota)
    try:
        llm_clients.call_model(cfg, "p", 50)
        raise AssertionError("insufficient balance should be fatal")
    except FatalAPIError:
        pass
    # An ordinary rate limit (after the SDK's retries) is a per-row api_error, not fatal.
    rate = openai.RateLimitError(
        "Error code: 429 - {'error': {'type': 'rate_limit_reached_error', 'message': "
        "'Your account reached max request per minute'}}",
        response=httpx.Response(429, request=req), body=None)
    llm_clients._clients[key] = raising(rate)
    assert llm_clients.call_model(cfg, "p", 50).status == "api_error"


# ---------------------------------------------------------------- claude CLI pilot adapter
def test_claude_cli_adapter(monkeypatch=None):
    import shutil
    import subprocess
    seen = {}
    envelope = {"type": "result", "subtype": "success", "is_error": False, "session_id": "s1",
                "result": "", "structured_output": {"match": 2, "confidence": 0.9, "rationale": "sku"},
                "usage": {"input_tokens": 50, "cache_read_input_tokens": 3000, "output_tokens": 120}}

    def fake_run(cmd, input, capture_output, text, cwd, env, timeout):
        seen.update(cmd=cmd, env=env, cwd=cwd, input=input)
        return NS(returncode=0, stdout=json.dumps(envelope), stderr="")

    orig_run, orig_which = subprocess.run, shutil.which
    subprocess.run, shutil.which = fake_run, (lambda name: "/usr/local/bin/claude")
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-should-not-leak"
    try:
        r = llm_clients.call_model(MODELS["haiku55_cli"], "PROMPT", 50)
        assert r.status == "ok" and r.match_index == 2 and r.cost_usd == 0.0
        assert r.input_tokens == 3050                            # total context, to detect injected context
        assert "ANTHROPIC_API_KEY" not in seen["env"]            # never bills the API
        for flag in ("-p", "--json-schema", "--tools", "--strict-mcp-config", "--no-session-persistence",
                     "--system-prompt", "--setting-sources"):
            assert flag in seen["cmd"], flag
        assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""
        assert seen["input"] == "PROMPT"
        envelope.update(is_error=True, result="Claude usage limit reached")
        try:
            llm_clients.call_model(MODELS["haiku55_cli"], "PROMPT", 50)
            raise AssertionError("usage limit should be fatal")
        except FatalAPIError:
            pass
        # Safety-classifier block: CLI exits rc=1 with an AUP message -> refusal, not api_error.
        envelope.update(is_error=True, result="API Error: Haiku 5.5 can't help with this. Start a new "
                        "session to continue.\n\nLearn more: https://www.anthropic.com/legal/aup\n\n"
                        "Details: `[bio]`")
        r = llm_clients.call_model(MODELS["haiku55_cli"], "PROMPT", 50)
        assert r.status == "refusal" and "[bio]" in r.error
    finally:
        subprocess.run, shutil.which = orig_run, orig_which
        os.environ.pop("ANTHROPIC_API_KEY", None)
    assert llm_runner.LLM_METHODS["llm_haiku55_cli"].result_root == "pilot_results"
    assert llm_runner.LLM_METHODS["llm_haiku55"].result_root == "results"
    ck = llm_runner.checkpoint_path_for("pilot_results/abt-buy/llm_haiku55_cli/original.parquet")
    assert ck.parts[:2] == ("checkpoints", "pilot")


# ---------------------------------------------------------------- runner
def _toy_data(n=30):
    left = pd.DataFrame({"id": [f"L{i}" for i in range(n)],
                         "name": [f"sony camera dsc w{i} silver" for i in range(n)]})
    right = pd.DataFrame({"id": [f"R{i}" for i in range(n)],
                          "name": [f"sony dsc w{i} digital camera" for i in range(n)]})
    mapping = pd.DataFrame({"idLeft": left["id"], "idRight": right["id"]})
    return left, right, mapping


def _fake_call(statuses):
    """call_model stand-in: picks the true match when status == 'ok'."""
    it = iter(statuses)

    def call(cfg, prompt, n, effort=None):
        s = next(it, "ok")
        if s == "fatal":
            raise FatalAPIError("no credit")
        if s == "ok":
            q = prompt.split("QUERY:\n")[1].split("\n")[0]
            w = q.split()[3]                                   # "w12"
            cands = prompt.split("CANDIDATES:\n")[1].split("\n\n")
            idx = next(i for i, c in enumerate(cands, 1) if f" {w} " in f" {c} ")
            return LLMResult("ok", idx, 0.9, "sku", raw_response="{}", input_tokens=10, output_tokens=5)
        return LLMResult(s, None, 0.0, error=s)
    return call


def _run(tmp, statuses, workers=1):
    llm_runner.call_model = _fake_call(statuses)
    left, right, mapping = _toy_data()
    out = Path(tmp) / "results" / "toy" / "llm_haiku55" / "original.parquet"
    return llm_runner.run_llm(left, right, mapping, "haiku55", output_path=str(out), workers=workers), out


def test_build_requests_blocking_and_sampling():
    left, right, mapping = _toy_data()
    reqs, k = llm_runner.build_requests(left, right, mapping, top_k=5)
    assert k == 5 and all(r["n_candidates"] == 5 for r in reqs)
    assert all(r["true_rank_in_candidates"] >= 1 for r in reqs)
    a, _ = llm_runner.build_requests(left, right, mapping, sample_frac=0.5)
    b, _ = llm_runner.build_requests(left, right, mapping, sample_frac=0.5)
    assert [r["id_left"] for r in a] == [r["id_left"] for r in b] and len(a) == 15
    s1, _ = llm_runner.build_requests(left, right, mapping, shuffle_candidates=True, shuffle_seed=1)
    s2, _ = llm_runner.build_requests(left, right, mapping, shuffle_candidates=True, shuffle_seed=1)
    assert [r["candidate_ids"] for r in s1] == [r["candidate_ids"] for r in s2]


def test_runner_happy_path_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        df, out = _run(tmp, ["ok"] * 30)
        for col in ("id_left", "true_id_right", "pred_id_right", "similarity_score", "predicted_match",
                    "is_correct", "status", "raw_response", "true_rank_in_candidates", "total_cost_usd"):
            assert col in df.columns, col
        assert df["is_correct"].sum() == 30
        ckpt = llm_runner.checkpoint_path_for(out)
        assert "checkpoints" in ckpt.parts and "results" not in ckpt.parts


def test_runner_aborts_on_consecutive_errors_then_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(tmp, ["ok"] * 3 + ["api_error"] * 5)
            raise AssertionError("expected abort")
        except RuntimeError as e:
            assert "consecutive" in str(e)
        df, _ = _run(tmp, ["ok"] * 30)                         # resume: only missing rows re-called
        assert len(df) == 30 and (df["status"] == "ok").all()


def test_runner_fatal_and_leftover_errors_never_final():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(tmp, ["ok", "fatal"])
            raise AssertionError("expected abort")
        except RuntimeError as e:
            assert "fatal" in str(e)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _run(tmp, ["ok"] * 29 + ["api_error"])
            raise AssertionError("expected leftover-error failure")
        except RuntimeError as e:
            assert "api_error rows remain" in str(e)


def test_parse_fail_retried_once():
    with tempfile.TemporaryDirectory() as tmp:
        df, _ = _run(tmp, ["parse_fail"] * 30)                 # first pass: all parse_fail
        assert (df["status"] == "parse_fail").all()
        df2, _ = _run(tmp, ["ok"] * 30)                        # second pass retries them
        assert (df2["status"] == "ok").all() and (df2["attempts"] == 2).all()
        df3, _ = _run(tmp, ["api_error"] * 30)                 # nothing left to retry
        assert (df3["status"] == "ok").all()


def test_method_registry_and_default_subsample():
    assert {"llm_haiku55", "llm_haiku55_tf", "llm_sonnet55", "llm_gpt6luna", "llm_kimik3",
            "llm_haiku55_cli", "llm_sonnet55_cli"} <= set(llm_runner.LLM_METHODS)
    assert llm_runner.LLM_METHODS["llm_kimik3"].result_root == "results"
    with tempfile.TemporaryDirectory() as tmp:
        llm_runner.call_model = _fake_call(["ok"] * 30)
        out = Path(tmp) / "results" / "toy" / "llm_sonnet55" / "original.parquet"
        df = llm_runner.run_llm(*_toy_data(), "sonnet55", output_path=str(out))
        assert len(df) == 9                                     # 30% of 30 query records
        df_h = llm_runner.run_llm(*_toy_data(), "haiku55",
                                  output_path=str(Path(tmp) / "results" / "toy" / "h" / "o.parquet"))
        assert set(df["id_left"]) <= set(df_h["id_left"]) and len(df_h) == 30


# ---------------------------------------------- previous-generation models (Part D)
def test_haiku45_config_and_request_shape():
    cfg = MODELS["haiku45"]
    assert cfg.model_id == "claude-haiku-4-5-20251001" and cfg.tier == "small"
    assert (cfg.input_price, cfg.output_price) == (1.00, 5.00)
    assert cfg.effort is None and cfg.thinking_budget == 1024
    assert 1024 <= cfg.thinking_budget < cfg.max_output_tokens          # API rules for budget_tokens
    fake = _FakeMessages(_anthropic_resp('{"match": 2, "confidence": 0.9, "rationale": "sku"}'))
    llm_clients._clients["anthropic"] = NS(messages=fake)
    r = llm_clients.call_model(cfg, "prompt", 50)
    assert r.status == "ok" and r.match_index == 2
    kw = fake.kwargs
    assert kw["model"] == "claude-haiku-4-5-20251001"
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "effort" not in kw["output_config"]                         # Haiku 4.5 rejects effort
    assert kw["output_config"]["format"]["type"] == "json_schema"       # structured output kept
    assert "temperature" not in kw and "top_p" not in kw and "fallbacks" not in kw
    assert kw["max_tokens"] > kw["thinking"]["budget_tokens"]
    assert abs(r.cost_usd - (1000 * 1.00 + 200 * 5.00) / 1e6) < 1e-12
    # the 5.5 models are unchanged: effort sent, no thinking param
    fake55 = _FakeMessages(_anthropic_resp('{"match": 1, "confidence": 0.5, "rationale": "x"}'))
    llm_clients._clients["anthropic"] = NS(messages=fake55)
    llm_clients.call_model(MODELS["haiku55"], "p", 50)
    assert "thinking" not in fake55.kwargs and fake55.kwargs["output_config"]["effort"] == "medium"


def test_haiku45_statuses_and_effort_guard():
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(_anthropic_resp("", "max_tokens")))
    assert llm_clients.call_model(MODELS["haiku45"], "p", 50).status == "truncated"
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(_anthropic_resp("", "refusal")))
    assert llm_clients.call_model(MODELS["haiku45"], "p", 50).status == "refusal"
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(_anthropic_resp("not json")))
    assert llm_clients.call_model(MODELS["haiku45"], "p", 50).status == "parse_fail"
    # --effort on Haiku 4.5 would be a 400 on every call: fail before any request is sent
    fake = _FakeMessages(_anthropic_resp("{}"))
    llm_clients._clients["anthropic"] = NS(messages=fake)
    try:
        llm_clients.call_model(MODELS["haiku45"], "p", 50, effort="low")
        raise AssertionError("expected FatalAPIError")
    except FatalAPIError:
        pass
    assert fake.kwargs is None
    # "" means "send none" and is allowed
    llm_clients._clients["anthropic"] = NS(messages=_FakeMessages(
        _anthropic_resp('{"match": 0, "confidence": 0.2, "rationale": "none"}')))
    assert llm_clients.call_model(MODELS["haiku45"], "p", 50, effort="").status == "no_match"


def test_gpt56luna_config_and_request_shape():
    cfg = MODELS["gpt56luna"]
    assert cfg.provider == "openai" and cfg.model_id == "gpt-5.6-luna" and cfg.tier == "small"
    assert (cfg.input_price, cfg.output_price) == (0.20, 1.20) and cfg.thinking_budget is None
    fake = _FakeResponses(_openai_resp('{"match": 3, "confidence": 0.7, "rationale": "code"}'))
    llm_clients._clients["openai"] = NS(responses=fake)
    r = llm_clients.call_model(cfg, "p", 50)
    assert r.status == "ok" and r.match_index == 3
    assert fake.kwargs["model"] == "gpt-5.6-luna"
    assert fake.kwargs["reasoning"] == {"effort": "medium"}             # same setting as GPT-6 Luna
    assert fake.kwargs["text"]["format"]["strict"] is True
    assert "temperature" not in fake.kwargs
    assert abs(r.cost_usd - (r.input_tokens * 0.20 + r.output_tokens * 1.20) / 1e6) < 1e-12


def test_previous_generation_method_registry():
    for k in ("llm_haiku45", "llm_haiku45_tf", "llm_gpt56luna", "llm_gpt56luna_tf"):
        assert k in llm_runner.LLM_METHODS and llm_runner.LLM_METHODS[k].result_root == "results"
    # Redesign 2026-11-21: the new models run on the fixed 30% subsample, the same query
    # records as Sonnet 5.5, so all models compare on one shared query set.
    with tempfile.TemporaryDirectory() as tmp:
        saved = llm_runner.call_model
        try:
            dfs = {}
            for mk in ("sonnet55", "haiku45", "gpt56luna", "kimik3"):
                llm_runner.call_model = _fake_call(["ok"] * 30)
                out = Path(tmp) / "results" / "toy" / f"llm_{mk}" / "original.parquet"
                dfs[mk] = llm_runner.run_llm(*_toy_data(), mk, output_path=str(out))
        finally:
            llm_runner.call_model = saved
        for mk in ("haiku45", "gpt56luna", "kimik3"):
            assert len(dfs[mk]) == 9                                    # 30% of 30 query records
            assert set(dfs[mk]["id_left"]) == set(dfs["sonnet55"]["id_left"]), mk
        assert set(dfs["haiku45"]["model_id"]) == {"claude-haiku-4-5-20251001"}
        assert set(dfs["haiku45"]["effort"]) == {""}
        assert set(dfs["kimik3"]["model_id"]) == {"kimi-k3"} and set(dfs["kimik3"]["effort"]) == {"low"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
