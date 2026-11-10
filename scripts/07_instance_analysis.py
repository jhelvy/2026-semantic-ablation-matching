"""
INSTANCE ANALYSIS (harness-v2 LLM runs)
=======================================

Two tables, both read from the saved per-query results (no API calls):

1. results/instance_analysis.csv: one row per dataset x LLM run x transformation.
   * status rates (ok, no_match, refusal, parse_fail, truncated);
   * shortlist recall: share of queries whose true match is in the TF-IDF shortlist;
   * TF-IDF top-1 accuracy on the same queries (the shortlist's first candidate);
   * LLM accuracy, and how its answers relate to TF-IDF's top-1:
     picked_top1 (share of answered queries where the LLM chose candidate 1),
     and the 2x2 of correct/incorrect for LLM vs TF-IDF top-1
     (both, llm_only, tfidf_only, neither; shares of all queries).

   The shortlist's first candidate agrees with the standalone TF-IDF method on
   96-98% of queries; it is used here because it is defined row by row.

2. results/f1_answered.csv: CV F1 with refusals excluded. For each reference model
   and transformation, keep the queries that model did not refuse and score every
   main method (TF-IDF, embeddings, the three LLMs) on exactly those queries. Rows
   with subset_of == method give the model's own "F1 when it answers". A method
   that did not run on every query in the subset (Sonnet's 30% sample) is skipped.

Usage:
    uv run scripts/07_instance_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cv import CFG, TRANS, cv_f1, truth_set  # noqa: E402
from lib.paths import RESULTS as RES  # noqa: E402

STATUSES = ["ok", "no_match", "refusal", "parse_fail", "truncated", "api_error"]
REF_MODELS = ["llm_haiku55", "llm_sonnet55", "llm_gpt6luna"]
COMPARE = ["tfidf", "sentence_transformer", "openai_embeddings"] + REF_MODELS


def instance_row(df):
    n = len(df)
    row = {"n": n}
    for s in STATUSES:
        row[f"rate_{s}"] = (df["status"] == s).mean()
    rank = df["true_rank_in_candidates"]
    llm_ok = df["is_correct"].astype(bool)
    tf_ok = rank == 1
    picked = df["match_index"].fillna(0).astype(int)
    answered = df["status"] == "ok"
    row.update(
        n_candidates=int(df["n_candidates"].max()),
        shortlist_recall=(rank >= 1).mean(),
        tfidf_top1_correct=tf_ok.mean(),
        llm_correct=llm_ok.mean(),
        picked_top1=(picked[answered] == 1).mean() if answered.any() else float("nan"),
        both=(llm_ok & tf_ok).mean(),
        llm_only=(llm_ok & ~tf_ok).mean(),
        tfidf_only=(~llm_ok & tf_ok).mean(),
        neither=(~llm_ok & ~tf_ok).mean(),
    )
    return row


def read(ds, method, t):
    pq = RES / ds / method / f"{t}.parquet"
    return pd.read_parquet(pq) if pq.exists() else None


def main():
    inst, ans = [], []
    for ds in CFG:
        truth = truth_set(ds)
        llm_dirs = sorted(d.name for d in (RES / ds).iterdir() if d.is_dir() and d.name.startswith("llm_"))
        for t in TRANS:
            for m in llm_dirs:
                df = read(ds, m, t)
                if df is not None and "status" in df.columns:
                    inst.append({"dataset": ds, "method": m, "transformation": t, **instance_row(df)})

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

    inst = pd.DataFrame(inst)
    ans = pd.DataFrame(ans)
    inst.to_csv(RES / "instance_analysis.csv", index=False)
    ans.to_csv(RES / "f1_answered.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    main_runs = inst[inst["method"].isin(REF_MODELS)]
    print(main_runs[["dataset", "method", "transformation", "rate_refusal", "shortlist_recall",
                     "tfidf_top1_correct", "llm_correct", "picked_top1", "llm_only", "tfidf_only"]]
          .round(3).to_string(index=False))
    own = ans[ans["subset_of"] == ans["method"]]
    print("\nF1 on the queries each model answered (refusals excluded):")
    print(own[["dataset", "method", "transformation", "share_answered", "f1_mean"]].round(3).to_string(index=False))
    print(f"\nsaved -> {RES / 'instance_analysis.csv'}\nsaved -> {RES / 'f1_answered.csv'}")


if __name__ == "__main__":
    main()
