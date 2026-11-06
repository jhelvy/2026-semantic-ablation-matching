"""
Offline tests for harness v2 (no network, no API keys, no cost).

    python scripts/test_llm_harness.py        # or: pytest scripts/test_llm_harness.py
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.append(str(Path(__file__).parent))

# methods.py imports sentence_transformers at module load; stub it if absent so
# these tests run in a light environment. (No embedding code is exercised here.)
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=object)

import pandas as pd  # noqa: E402

import llm_clients  # noqa: E402
import llm_runner  # noqa: E402
from llm_clients import MODELS, LLMResult, FatalAPIError, parse_match_json  # noqa: E402


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
    r = llm_clients.call_model(MODELS["luna"], "p", 50)
    assert r.status == "no_match"
    assert fake.kwargs["text"]["format"]["strict"] is True
    assert fake.kwargs["reasoning"] == {"effort": "medium"}
    llm_clients.call_model(MODELS["gpt4omini"], "p", 50)        # bridge model: no reasoning param
    assert "reasoning" not in llm_clients._clients["openai"].responses.kwargs
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(
        _openai_resp("", "incomplete", "max_output_tokens")))
    assert llm_clients.call_model(MODELS["luna"], "p", 50).status == "truncated"
    llm_clients._clients["openai"] = NS(responses=_FakeResponses(_openai_resp("", refusal=True)))
    assert llm_clients.call_model(MODELS["luna"], "p", 50).status == "refusal"


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


def test_method_registry():
    assert {"llm_haiku55", "llm_haiku55_tf", "llm_opus55", "llm_luna", "llm_gpt4omini"} <= set(llm_runner.LLM_METHODS)
    try:
        llm_runner.run_llm(*_toy_data(), "opus55")
        raise AssertionError("opus55 full run should require --sample-frac")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
