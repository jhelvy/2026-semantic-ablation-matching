"""
Offline tests for harness v2 (no network, no API keys, no cost).

    python tests/test_llm_harness.py        # or: pytest tests/test_llm_harness.py
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
              usage=NS(input_tokens=1000, output_tokens=300), model="gpt-4o-mini", id="resp_1")


def test_openai_request_shape_and_statuses():
    fake = _FakeResponses(_openai_resp('{"match": 0, "confidence": 0.6, "rationale": "none"}'))
    llm_clients._clients["openai"] = NS(responses=fake)
    r = llm_clients.call_model(MODELS["gpt4omini"], "p", 50)
    assert r.status == "no_match"
    assert fake.kwargs["text"]["format"]["strict"] is True
    assert "reasoning" not in fake.kwargs                        # non-reasoning bridge model
    llm_clients.call_model(MODELS["gpt4omini"], "p", 50, effort="low")
    assert fake.kwargs["reasoning"] == {"effort": "low"}         # explicit override is passed through
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(
        _openai_resp("", "incomplete", "max_output_tokens")))
    assert llm_clients.call_model(MODELS["gpt4omini"], "p", 50).status == "truncated"
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(_openai_resp("", refusal=True)))
    assert llm_clients.call_model(MODELS["gpt4omini"], "p", 50).status == "refusal"


# ---------------------------------------------------------------- kimi (openai-compatible)
class _FakeChat:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None
        self.completions = self

    def create(self, **kw):
        self.kwargs = kw
        return self.resp


def _chat_resp(text, finish="stop"):
    return NS(choices=[NS(message=NS(content=text, refusal=None), finish_reason=finish)],
              usage=NS(prompt_tokens=2000, completion_tokens=900), model="moonshotai/kimi-k3", id="gen_1")


def test_kimi_request_shape_and_statuses():
    cfg = MODELS["kimik3"]
    key = f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}"
    fake = _FakeChat(_chat_resp('{"match": 7, "confidence": 0.7, "rationale": "code"}'))
    llm_clients._clients[key] = NS(chat=fake)
    r = llm_clients.call_model(cfg, "p", 50)
    assert r.status == "ok" and r.match_index == 7
    assert fake.kwargs["response_format"]["type"] == "json_schema"
    if "openrouter.ai" in cfg.base_url:                          # default: provider pinned, no fallback
        assert fake.kwargs["extra_body"]["provider"] == {"order": ["moonshotai"], "allow_fallbacks": False}
    assert abs(r.cost_usd - (2000 * 3.0 + 900 * 15.0) / 1e6) < 1e-12
    assert cfg.default_sample_frac == 0.3
    llm_clients._clients[key] = NS(chat=_FakeChat(_chat_resp("", "length")))
    assert llm_clients.call_model(cfg, "p", 50).status == "truncated"


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
    assert {"llm_haiku55", "llm_haiku55_tf", "llm_sonnet55", "llm_gpt4omini", "llm_kimik3",
            "llm_haiku55_cli", "llm_sonnet55_cli"} <= set(llm_runner.LLM_METHODS)
    with tempfile.TemporaryDirectory() as tmp:
        llm_runner.call_model = _fake_call(["ok"] * 30)
        out = Path(tmp) / "results" / "toy" / "llm_sonnet55" / "original.parquet"
        df = llm_runner.run_llm(*_toy_data(), "sonnet55", output_path=str(out))
        assert len(df) == 9                                     # 30% of 30 query records
        df_h = llm_runner.run_llm(*_toy_data(), "haiku55",
                                  output_path=str(Path(tmp) / "results" / "toy" / "h" / "o.parquet"))
        assert set(df["id_left"]) <= set(df_h["id_left"]) and len(df_h) == 30


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
