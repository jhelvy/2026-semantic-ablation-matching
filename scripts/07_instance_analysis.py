"""
INSTANCE ANALYSIS (harness-v2 LLM runs)
=======================================

Reads the saved per-query results (no API calls) and writes five tables.

1. results/instance_analysis.csv: one row per dataset x LLM run x transformation.
   * status rates (ok, no_match, refusal, parse_fail, truncated);
   * shortlist recall: share of queries whose true match is in the TF-IDF shortlist;
   * TF-IDF top-1 accuracy on the same queries (the shortlist's first candidate);
   * LLM accuracy (correct top pick; refusals count as misses), and how its answers
     relate to TF-IDF's top-1: picked_top1 (share of answered queries where the LLM
     chose candidate 1), and the 2x2 of LLM vs TF-IDF top-1 correct/incorrect
     (both, llm_only, tfidf_only, neither; shares of all queries);
   * CV F1 on all queries and on queries whose true match made the shortlist;
   * unique shared identifier (R2 D2): uid_share = share of queries where the query
     and its true match share an identifier piece that no other shortlisted candidate
     has; llm_correct_uid / tfidf_top1_correct_uid = accuracy on those queries;
     miss_uid_share = share of LLM misses that had such a piece (misses_answered_uid_share:
     the same, refusals left out);
   * prompt size: mean input tokens, and the Spearman correlation between input
     tokens and a correct pick within the run.

   The shortlist's first candidate agrees with the standalone TF-IDF method on
   96-98% of queries; it is used here because it is defined row by row.

   Identifier pieces are the factorial identifier set (`id_map` keys in
   data/<ds>/data_test/factorial_manifest.json: model numbers and SKU pieces),
   found in the original text the LLM sees (Amazon-Google: title only). Every
   transform preserves token identity across catalogs, so a piece shared on
   Original is shared, in transformed form, under every condition. The LLM
   prompt truncates long Abt-Buy descriptions; pieces found only past the cut
   are still counted.

2. results/f1_answered.csv: CV F1 with refusals excluded. For each reference model
   and transformation, keep the queries that model did not refuse and score every
   main method (TF-IDF, embeddings, the three LLMs) on exactly those queries. Rows
   with subset_of == method give the model's own "F1 when it answers". A method
   that did not run on every query in the subset (Sonnet's 30% sample) is skipped.

3. results/shortlist_recall.csv: recall of the TF-IDF shortlist at k = 1, 5, 10,
   20, 50, per dataset and transformation (from the k=50 Luna runs; the shortlist
   does not depend on the model).

4. results/prompt_size.csv: accuracy by input-token bin (R2 D1c). Bins are the
   quintiles of input tokens pooled over a run's conditions, so Original and the
   ciphers can be compared at the same prompt size.

5. results/bootstrap_ci.csv: paired bootstrap 95% CIs (2,000 resamples of query
   ids, seed 42). Three comparisons, each on the queries both arms share:
   * drop:   Original minus the condition, same LLM run;
   * vs_tfidf_top1: LLM minus TF-IDF top-1 (accuracy only), same condition;
   * model_gap: one main LLM minus another, same condition.
   Metrics: accuracy (correct top pick) and F1 at a fixed threshold (the
   F1-optimal tau on the full set for each arm). The reported F1 values stay the
   CV F1 from 05; the fixed-tau F1 is only used to put a CI on differences.

Usage:
    uv run scripts/07_instance_analysis.py
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cv import CFG, GRID, TRANS, cv_f1, prf, true_and_score, truth_set  # noqa: E402
from lib.datasets import DATASETS, load_csv  # noqa: E402
from lib.paths import DATA, RESULTS as RES  # noqa: E402

STATUSES = ["ok", "no_match", "refusal", "parse_fail", "truncated", "api_error"]
REF_MODELS = ["llm_haiku55", "llm_sonnet55", "llm_gpt6luna"]
COMPARE = ["tfidf", "sentence_transformer", "openai_embeddings"] + REF_MODELS
RECALL_K = [1, 5, 10, 20, 50]
N_BINS, N_BOOT, SEED = 5, 2000, 42
PIECE_RE = re.compile(r"\w+")


# =============================================================================
# INPUTS
# =============================================================================

def read(ds, method, t):
    pq = RES / ds / method / f"{t}.parquet"
    return pd.read_parquet(pq) if pq.exists() else None


def llm_text(df, title_only):
    """Text the LLM sees, as in lib.methods.prepare_text_column (lowercased)."""
    if title_only:
        cols = [next(c for c in df.columns[1:] if c.lower() in ("title", "name"))]
    else:
        cols = [c for c in df.columns[1:] if pd.api.types.is_string_dtype(df[c])
                and c.lower() not in ("year", "price", "id")]
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def identifier_pieces(ds):
    """{side: {record id: set of identifier pieces}} for one dataset (original text)."""
    cfg = DATASETS[ds]
    manifest = json.loads((DATA / ds / "data_test" / "factorial_manifest.json").read_text())
    ids = set(manifest["id_map"])
    out = {}
    for side in ("left", "right"):
        df = load_csv(cfg["dir"] / "data_original" / f"{cfg[side + '_name']}.csv")
        text = llm_text(df, cfg["title_only"])
        out[side] = {str(i): set(PIECE_RE.findall(t)) & ids for i, t in zip(df.iloc[:, 0].astype(str), text)}
    return out


def unique_id_flags(df, pieces):
    """Per row: True if the query and its true match share an identifier piece that
    no other shortlisted candidate has (False if the true match is not shortlisted)."""
    L, R = pieces["left"], pieces["right"]
    flags = []
    for lid, tid, cands in zip(df["id_left"].astype(str), df["true_id_right"].astype(str), df["candidate_ids"]):
        cand = json.loads(cands) if cands else []
        if tid not in cand:
            flags.append(False)
            continue
        shared = L.get(lid, set()) & R.get(tid, set())
        others = set().union(*(R.get(c, set()) for c in cand if c != tid)) if shared else set()
        flags.append(bool(shared - others))
    return np.array(flags)


# =============================================================================
# TABLE 1: PER-RUN INSTANCE STATS
# =============================================================================

def instance_row(df, truth, uid):
    row = {"n": len(df)}
    for s in STATUSES:
        row[f"rate_{s}"] = (df["status"] == s).mean()
    rank = df["true_rank_in_candidates"].values
    llm_ok = df["is_correct"].astype(bool).values
    tf_ok = rank == 1
    picked = df["match_index"].fillna(0).astype(int).values
    answered = (df["status"] == "ok").values
    refused = (df["status"] == "refusal").values
    miss = ~llm_ok
    in_sl = rank >= 1
    tokens = df["input_tokens"].astype(float)
    row.update(
        n_candidates=int(df["n_candidates"].max()),
        shortlist_recall=in_sl.mean(),
        tfidf_top1_correct=tf_ok.mean(),
        llm_correct=llm_ok.mean(),
        picked_top1=(picked[answered] == 1).mean() if answered.any() else np.nan,
        both=(llm_ok & tf_ok).mean(),
        llm_only=(llm_ok & ~tf_ok).mean(),
        tfidf_only=(~llm_ok & tf_ok).mean(),
        neither=(~llm_ok & ~tf_ok).mean(),
        f1_all=cv_f1(df, truth)[0],
        f1_in_shortlist=cv_f1(df[in_sl], truth)[0],
        uid_share=uid.mean(),
        llm_correct_uid=llm_ok[uid].mean() if uid.any() else np.nan,
        tfidf_top1_correct_uid=tf_ok[uid].mean() if uid.any() else np.nan,
        miss_uid_share=uid[miss].mean() if miss.any() else np.nan,
        misses_answered_uid_share=uid[miss & ~refused].mean() if (miss & ~refused).any() else np.nan,
        mean_input_tokens=tokens.mean(),
        spearman_tokens_correct=tokens.corr(pd.Series(llm_ok.astype(float)), method="spearman")
        if llm_ok.std() > 0 else np.nan,
    )
    return row


# =============================================================================
# TABLE 4: PROMPT SIZE
# =============================================================================

def prompt_size_rows(ds, method, runs):
    """runs: {transformation: df}. Quintile bins of input tokens pooled over conditions."""
    pooled = pd.concat([d["input_tokens"] for d in runs.values()]).astype(float)
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, N_BINS + 1)))
    out = []
    for t, d in runs.items():
        b = np.clip(np.searchsorted(edges, d["input_tokens"].astype(float), side="right") - 1, 0, len(edges) - 2)
        for i in range(len(edges) - 1):
            m = b == i
            if m.any():
                out.append(dict(dataset=ds, method=method, transformation=t, bin=i + 1,
                                tokens_lo=edges[i], tokens_hi=edges[i + 1], n=int(m.sum()),
                                mean_tokens=d["input_tokens"][m].mean(),
                                llm_correct=d["is_correct"][m].mean(),
                                rate_refusal=(d["status"][m] == "refusal").mean()))
    return out


# =============================================================================
# TABLE 5: PAIRED BOOTSTRAP
# =============================================================================

def best_tau(true, score):
    return max(GRID, key=lambda t: prf(true, score, t)[0])


def arm(df, truth):
    """Per-query vectors for one arm, indexed by id_left."""
    df = df.drop_duplicates("id_left").set_index(df.drop_duplicates("id_left")["id_left"].astype(str))
    true, score = true_and_score(df, truth)
    pred = score >= best_tau(true, score)
    return pd.DataFrame({"acc": df["is_correct"].astype(float).values,
                         "tp": (pred & true).astype(float), "fp": (pred & ~true).astype(float),
                         "top1": (df["true_rank_in_candidates"] == 1).astype(float).values
                         if "true_rank_in_candidates" in df else np.nan}, index=df.index)


def boot(a, b, metric, rng):
    """Paired bootstrap of metric(a) - metric(b) over shared queries. Returns (est, lo, hi, n)."""
    ids = a.index.intersection(b.index)
    A, B = a.loc[ids], b.loc[ids]
    n = len(ids)
    W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)

    col_a, col_b = ("acc", "top1") if metric == "acc_vs_top1" else ("acc", "acc")
    use_f1 = metric == "f1"

    def stat(X, col, w):
        if not use_f1:
            return (w @ X[col].values) / w.sum(axis=-1)
        tp, fp = w @ X["tp"].values, w @ X["fp"].values
        return 2 * tp / (tp + fp + w.sum(axis=-1))

    est = float(stat(A, col_a, np.ones(n)) - stat(B, col_b, np.ones(n)))
    diffs = stat(A, col_a, W) - stat(B, col_b, W)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return est, float(lo), float(hi), n


# =============================================================================
# MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(SEED)
    inst, ans, recall, psize, ci = [], [], [], [], []
    for ds in CFG:
        truth = truth_set(ds)
        pieces = identifier_pieces(ds)
        llm_dirs = sorted(d.name for d in (RES / ds).iterdir() if d.is_dir() and d.name.startswith("llm_"))
        arms = {}   # (method, t) -> arm vectors

        for m in llm_dirs:
            runs = {}
            for t in TRANS:
                df = read(ds, m, t)
                if df is None or "status" not in df.columns:
                    continue
                runs[t] = df
                uid = unique_id_flags(df, pieces)
                inst.append({"dataset": ds, "method": m, "transformation": t, **instance_row(df, truth, uid)})
                arms[(m, t)] = arm(df, truth)
            if runs:
                psize += prompt_size_rows(ds, m, runs)

        for t in TRANS:
            # shortlist recall at each k (Luna main run: k=50, and present for every condition)
            df = read(ds, "llm_gpt6luna", t)
            if df is not None:
                r = df["true_rank_in_candidates"]
                recall.append({"dataset": ds, "transformation": t, "n": len(df),
                               **{f"recall_at_{k}": ((r >= 1) & (r <= k)).mean() for k in RECALL_K}})

            # F1 with refusals excluded
            tables = {m: read(ds, m, t) for m in COMPARE}
            for ref in REF_MODELS:
                rdf = tables[ref]
                if rdf is None:
                    continue
                keep = set(rdf.loc[rdf["status"] != "refusal", "id_left"].astype(str))
                for m, df in tables.items():
                    if df is None:
                        continue
                    ids = df["id_left"].astype(str)
                    if not keep <= set(ids):
                        continue   # method did not run on every query in the subset
                    f1, sd, P, R = cv_f1(df[ids.isin(keep)], truth)
                    ans.append(dict(dataset=ds, subset_of=ref, transformation=t, n_subset=len(keep),
                                    share_answered=len(keep) / len(rdf), method=m,
                                    f1_mean=f1, f1_std=sd, precision=P, recall=R))

        # bootstrap CIs
        def add(kind, a_name, b_name, t, a, b, metrics):
            for metric in metrics:
                est, lo, hi, n = boot(a, b, metric, rng)
                ci.append(dict(dataset=ds, comparison=kind, transformation=t, method_a=a_name,
                               method_b=b_name, metric="acc" if metric.startswith("acc") else "f1_fixed_tau",
                               n=n, estimate=est, ci_lo=lo, ci_hi=hi, excludes_zero=lo > 0 or hi < 0))

        for (m, t), a in arms.items():
            add("vs_tfidf_top1", m, "tfidf_top1", t, a, a, ["acc_vs_top1"])
            if t != "original" and (m, "original") in arms:
                add("drop", f"{m}:original", f"{m}:{t}", t, arms[(m, "original")], a, ["acc", "f1"])
        for t in TRANS:
            for x, y in combinations(REF_MODELS, 2):
                if (x, t) in arms and (y, t) in arms:
                    add("model_gap", x, y, t, arms[(x, t)], arms[(y, t)], ["acc", "f1"])

    outputs = {
        "instance_analysis.csv": pd.DataFrame(inst),
        "f1_answered.csv": pd.DataFrame(ans),
        "shortlist_recall.csv": pd.DataFrame(recall),
        "prompt_size.csv": pd.DataFrame(psize),
        "bootstrap_ci.csv": pd.DataFrame(ci),
    }
    for name, df in outputs.items():
        df.to_csv(RES / name, index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    inst = outputs["instance_analysis.csv"]
    main_runs = inst[inst["method"].isin(REF_MODELS)]
    print(main_runs[["dataset", "method", "transformation", "rate_refusal", "tfidf_top1_correct", "llm_correct",
                     "llm_only", "tfidf_only", "f1_all", "f1_in_shortlist", "uid_share", "llm_correct_uid",
                     "tfidf_top1_correct_uid", "miss_uid_share", "mean_input_tokens", "spearman_tokens_correct"]]
          .round(3).to_string(index=False))
    print("\nShortlist recall:")
    print(outputs["shortlist_recall.csv"].round(3).to_string(index=False))
    print("\nPaired bootstrap, main models:")
    c = outputs["bootstrap_ci.csv"]
    c = c[c["method_a"].str.split(":").str[0].isin(REF_MODELS)]
    print(c.round(3).to_string(index=False))
    for name in outputs:
        print(f"saved -> {RES / name}")


if __name__ == "__main__":
    main()
