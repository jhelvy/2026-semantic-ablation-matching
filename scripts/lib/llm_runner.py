"""
LLM RUNNER (harness v2)
=======================

TF-IDF blocking (top-k candidates per query) -> one structured LLM call per
query -> per-row status/diagnostics -> F1-optimal confidence threshold.

Differences from the v1 pipeline in methods.py (`llm`, `_run_llm_method`):
  * all models share one code path (llm_clients.call_model);
  * structured JSON output; row-level `status`; raw responses logged;
  * parallel requests (`workers`), checkpoints under checkpoints/ (never
    under results/, so analysis scripts cannot pick up partial runs);
  * fails loudly: fatal account errors abort at once; the run aborts if
    5 consecutive calls or >max_error_rate of calls end in api_error; a
    run with unresolved api_error rows is never written as a final result;
  * parse_fail rows are retried once on resume;
  * diagnostics for instance-level analysis: rank of the true match in the
    blocked candidate list, candidate ids, prompt size.

Output keeps the v1 columns used by 06_summarize_results.py / 05_cv_threshold.py
(id_left, true_id_right, pred_id_right, similarity_score, predicted_match,
is_correct, total_cost_usd, wall_time_seconds).
"""

from __future__ import annotations

import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm_clients import MODELS, FatalAPIError, call_model
from . import methods as _methods          # read _methods.TITLE_ONLY_MODE at call time (set per dataset)
from .methods import create_ground_truth_set, get_display_column, prepare_text_column
from .paths import CHECKPOINTS, PILOT_RESULTS, RESULTS

# =============================================================================
# PROMPTS
# =============================================================================
# Same task framing and guidelines as v1; the free-text RESPONSE FORMAT block is
# replaced by a JSON instruction (the schema itself is enforced by the API).

_JSON_INSTRUCTION = (
    "Respond with JSON only: \"match\" (the candidate number, or 0 if no candidate matches), "
    "\"confidence\" (0.0 to 1.0), and \"rationale\" (one short sentence naming the evidence)."
)

PROMPTS = {
    "standard": """You are an expert at entity matching. Determine if any candidate matches the query.

QUERY:
{query_text}

CANDIDATES:
{candidates_text}
GUIDELINES:
- Entities are SAME if they refer to the exact same real-world item/entity
- Consider: all available attributes (names, identifiers, descriptions, metadata)
- Account for: abbreviations, different word orders, minor variations, formatting differences
- Be strict: only match if you're confident they're the same entity

""" + _JSON_INSTRUCTION,

    # v1 told the model the words were "no longer readable English" even on the
    # Original condition; this wording is valid for every condition.
    "token_fallback": """You are an expert at entity matching. The text below may have been transformed so that words are not readable English. Any transformation is consistent: the same original word always maps to the same token in both the query and the candidates. Find which candidate refers to the same entity as the query by comparing shared tokens, character patterns, numbers, and structure, rather than by relying on what the words mean.

QUERY:
{query_text}

CANDIDATES:
{candidates_text}
GUIDELINES:
- Focus on overlapping tokens, shared substrings, matching numbers, and structural patterns
- The same original word always becomes the same token across query and candidates
- Distinctive tokens (model numbers, codes, identifiers) are the strongest signal
- Be strict: only match if token overlap and structure clearly indicate the same entity

""" + _JSON_INSTRUCTION,
}

RESULT_STATUSES = ("ok", "no_match", "parse_fail", "truncated", "refusal", "api_error")
_DONE_STATUSES = {"ok", "no_match", "truncated", "refusal"}   # model behaviour: keep on resume


def _adaptive_top_k(n_right: int) -> int:
    """Same rule as v1."""
    if n_right < 2000:
        return 50
    if n_right < 10000:
        return 100
    return 200


# =============================================================================
# REQUEST CONSTRUCTION (no API calls; also used for dry-run cost estimates)
# =============================================================================

def build_requests(df_left, df_right, df_mapping, prompt_key="standard", top_k=None,
                   max_text_length=2000, sample_frac=None, sample_seed=42,
                   shuffle_candidates=False, shuffle_seed=0):
    """Return (requests, top_k). One request per (sampled) left record."""
    df_left = prepare_text_column(df_left.copy(), title_only=_methods.TITLE_ONLY_MODE)
    df_right = prepare_text_column(df_right.copy(), title_only=_methods.TITLE_ONLY_MODE)
    id_l, id_r = df_left.columns[0], df_right.columns[0]
    name_l, name_r = get_display_column(df_left), get_display_column(df_right)

    true_matches = create_ground_truth_set(df_mapping)
    truth_by_left = {}
    for la, rb in true_matches:
        truth_by_left.setdefault(la, []).append(rb)
    for v in truth_by_left.values():
        v.sort()                                   # first truth independent of PYTHONHASHSEED

    if sample_frac:
        rng = np.random.default_rng(sample_seed)
        ids = np.array(sorted(df_left[id_l].astype(str).unique()))
        keep = set(rng.choice(ids, size=max(1, int(round(len(ids) * sample_frac))), replace=False))
        df_left = df_left[df_left[id_l].astype(str).isin(keep)]

    top_k = top_k or _adaptive_top_k(len(df_right))
    vec = TfidfVectorizer()
    vec.fit(pd.concat([df_left["text"], df_right["text"]]))
    right_m = vec.transform(df_right["text"])
    right_ids = df_right[id_r].astype(str).values
    right_names = df_right[name_r].values
    right_text = df_right["text"].values
    template = PROMPTS[prompt_key]

    def _trunc(s):
        return s[:max_text_length] + "..." if max_text_length > 0 and len(s) > max_text_length else s

    reqs = []
    for _, row in df_left.iterrows():
        lid = str(row[id_l])
        sims = cosine_similarity(vec.transform([row["text"]]), right_m).flatten()
        cand_idx = list(sims.argsort()[::-1][:top_k])
        if shuffle_candidates:
            # deterministic per query, so every model sees the same order
            np.random.default_rng([shuffle_seed, zlib.crc32(lid.encode())]).shuffle(cand_idx)
        cand_ids = [right_ids[i] for i in cand_idx]
        truths = truth_by_left.get(lid, [])
        ranks = [cand_ids.index(t) + 1 for t in truths if t in cand_ids]
        cands_text = "".join(f"{n}. {_trunc(right_text[i])}\n\n" for n, i in enumerate(cand_idx, 1))
        prompt = template.format(query_text=_trunc(row["text"]), candidates_text=cands_text)
        reqs.append(dict(
            id_left=lid, left_name=row[name_l], true_id_right=truths[0] if truths else "",
            prompt=prompt, prompt_chars=len(prompt), n_candidates=len(cand_idx),
            candidate_ids=cand_ids, candidate_names=[right_names[i] for i in cand_idx],
            true_rank_in_candidates=min(ranks) if ranks else 0,   # 0 = blocked out
        ))
    return reqs, top_k


# =============================================================================
# CHECKPOINTS
# =============================================================================

def checkpoint_path_for(output_path):
    """results/... -> checkpoints/...; pilot_results/... -> checkpoints/pilot/..."""
    p = Path(output_path)
    # Paths under the repo's results trees map into the repo's checkpoints/ tree.
    for root, ck in ((RESULTS, CHECKPOINTS), (PILOT_RESULTS, CHECKPOINTS / "pilot")):
        if p.is_absolute() and p.is_relative_to(root):
            return ck / p.relative_to(root)
    # Any other path: swap the first "results"/"pilot_results" component.
    parts = list(p.parts)
    for root, repl in (("results", ["checkpoints"]), ("pilot_results", ["checkpoints", "pilot"])):
        if root in parts:
            i = parts.index(root)
            return Path(*parts[:i], *repl, *parts[i + 1:])
    return p.with_suffix(".checkpoint.parquet")


def _save(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["candidate_ids"] = df["candidate_ids"].apply(lambda x: x if isinstance(x, str) else json.dumps(list(x)))
    df.to_parquet(path, index=False)


# =============================================================================
# THRESHOLD (identical logic to v1 _run_llm_method)
# =============================================================================

def _apply_best_threshold(rows, true_matches):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        tp = fp = fn = 0
        for r in rows:
            pred = r["pred_id_right"] != "" and r["similarity_score"] >= t
            hit = (r["id_left"], r["pred_id_right"]) in true_matches
            tp += pred and hit
            fp += pred and not hit
            fn += (r["true_id_right"] != "") and not (pred and hit)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    for r in rows:
        pred = r["pred_id_right"] != "" and r["similarity_score"] >= best_t
        r["predicted_match"] = int(pred)
        r["is_correct"] = int(pred and (r["id_left"], r["pred_id_right"]) in true_matches)
    return best_t, best_f1


# =============================================================================
# MAIN ENTRY
# =============================================================================

def run_llm(df_left, df_right, df_mapping, model_key, prompt_key="standard", output_path=None,
            effort=None, sample_frac=None, sample_seed=42, workers=8, max_error_rate=0.01,
            shuffle_candidates=False, shuffle_seed=0, top_k=None, max_text_length=2000):
    cfg = MODELS[model_key]
    if sample_frac is None:
        sample_frac = cfg.default_sample_frac      # e.g. Sonnet / Kimi: fixed 30% subsample
    eff = cfg.effort if effort is None else (effort or None)
    print(f"\n{cfg.label} [{cfg.model_id}] prompt={prompt_key} effort={eff} "
          f"sample_frac={sample_frac} workers={workers}")

    reqs, top_k = build_requests(df_left, df_right, df_mapping, prompt_key, top_k=top_k,
                                 max_text_length=max_text_length, sample_frac=sample_frac,
                                 sample_seed=sample_seed, shuffle_candidates=shuffle_candidates,
                                 shuffle_seed=shuffle_seed)
    true_matches = create_ground_truth_set(df_mapping)
    print(f"  {len(reqs)} queries, top_k={top_k}, blocking recall="
          f"{np.mean([r['true_rank_in_candidates'] > 0 for r in reqs]):.3f}")

    # ---- resume
    ckpt = checkpoint_path_for(output_path) if output_path else None
    done = {}
    if ckpt and ckpt.exists():
        prev = pd.read_parquet(ckpt)
        for r in prev.to_dict("records"):
            if r["status"] in _DONE_STATUSES or (r["status"] == "parse_fail" and r.get("attempts", 1) >= 2):
                done[r["id_left"]] = r
        print(f"  resumed {len(done)} finished rows from {ckpt}")
    attempts_prev = {}
    if ckpt and ckpt.exists():
        attempts_prev = dict(zip(prev["id_left"], prev.get("attempts", pd.Series(1, index=prev.index))))

    todo = [r for r in reqs if r["id_left"] not in done]
    rows = list(done.values())
    t0 = time.time()
    n_err = consecutive_err = n_calls = 0

    def _row(req, res):
        idx = res.match_index
        picked = idx is not None and idx > 0
        return dict(
            id_left=req["id_left"], left_name=req["left_name"], true_id_right=req["true_id_right"],
            pred_id_right=req["candidate_ids"][idx - 1] if picked else "",
            pred_right_name=req["candidate_names"][idx - 1] if picked else "",
            similarity_score=float(res.confidence),
            status=res.status, match_index=idx if idx is not None else -1,
            rationale=res.rationale, raw_response=res.raw_response, error=res.error,
            stop_reason=res.stop_reason, input_tokens=res.input_tokens,
            output_tokens=res.output_tokens, cost_usd=res.cost_usd, model_id=cfg.model_id,
            model_served=res.model_served, request_id=res.request_id, latency_s=res.latency_s,
            effort=eff or "", prompt_key=prompt_key,
            attempts=int(attempts_prev.get(req["id_left"], 0)) + 1,
            true_rank_in_candidates=req["true_rank_in_candidates"], n_candidates=req["n_candidates"],
            candidate_ids=json.dumps(req["candidate_ids"]), prompt_chars=req["prompt_chars"],
        )

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(call_model, cfg, r["prompt"], r["n_candidates"], effort): r for r in todo}
    aborted = None
    try:
        for fut in as_completed(futures):
            req = futures[fut]
            res = fut.result()                        # FatalAPIError propagates
            rows.append(_row(req, res))
            n_calls += 1
            if res.status == "api_error":
                n_err += 1
                consecutive_err += 1
            else:
                consecutive_err = 0
            if consecutive_err >= 5:
                aborted = f"5 consecutive api_error rows; last error: {res.error}"
            elif n_calls >= 100 and n_err / n_calls > max_error_rate:
                aborted = f"api_error rate {n_err}/{n_calls} > {max_error_rate:.0%}; last: {res.error}"
            if aborted:
                break
            if ckpt and n_calls % 25 == 0:
                _save(rows, ckpt)
    except FatalAPIError as e:
        aborted = f"fatal: {e}"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if ckpt and rows:
            _save(rows, ckpt)

    if aborted:
        raise RuntimeError(f"{cfg.label} run aborted ({aborted}). {len(rows)} rows checkpointed "
                           f"at {ckpt}; rerun the same command to resume.")
    n_api = sum(r["status"] == "api_error" for r in rows)
    if n_api:
        raise RuntimeError(f"{n_api} api_error rows remain; rerun the same command to retry them "
                           f"(checkpoint: {ckpt}). Not writing a final result.")

    wall = time.time() - t0
    status_counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    total_cost = float(sum(r["cost_usd"] for r in rows))
    best_t, best_f1 = _apply_best_threshold(rows, true_matches)
    print(f"  status: {status_counts}")
    print(f"  cost=${total_cost:.2f} wall={wall:.0f}s best threshold={best_t:.2f} F1={best_f1:.3f}")

    out = pd.DataFrame(rows)
    out["total_cost_usd"] = total_cost
    out["wall_time_seconds"] = wall
    return out


# =============================================================================
# METHOD REGISTRY (merged into 04_run_experiments' METHODS)
# =============================================================================

def make_llm_method(model_key, prompt_key):
    def _method(df_left, df_right, df_mapping, output_path=None, llm_options=None):
        return run_llm(df_left, df_right, df_mapping, model_key, prompt_key,
                       output_path=output_path, **(llm_options or {}))
    _method.__name__ = f"llm_{model_key}" + ("_tf" if prompt_key == "token_fallback" else "")
    _method.__doc__ = f"{MODELS[model_key].label}, TF-IDF blocking, {prompt_key} prompt (harness v2)."
    # Subscription (CLI) pilots never write to results/, so they cannot leak into reported numbers.
    _method.result_root = "pilot_results" if MODELS[model_key].provider == "claude_cli" else "results"
    return _method


LLM_METHODS = {}
for _k in MODELS:
    LLM_METHODS[f"llm_{_k}"] = make_llm_method(_k, "standard")
    LLM_METHODS[f"llm_{_k}_tf"] = make_llm_method(_k, "token_fallback")
