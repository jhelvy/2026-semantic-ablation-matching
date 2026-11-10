"""
5-fold cross-validated threshold selection: choose tau on 4 folds, evaluate on the
held-out fold, report mean +/- std F1/P/R. Removes the 'threshold tuned on the test
set' bias and yields error bars. No experiments re-run: every method, Soft TF-IDF
included, uses the saved per-record scores in results/ (and results/v1_llm/).
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import RESULTS as RES, V1_LLM_RESULTS  # noqa: E402
from lib.cv import CFG, TRANS, cv, true_and_score, truth_set  # noqa: E402

# Scale check (PLAN.md, "Method lineup"): Sonnet runs on a fixed 30% of the queries, so
# the small models and TF-IDF are also scored on exactly those queries. Those rows are
# written as "{method}__sonnet30", next to each method's full-set row.
SCALE_REF = "llm_sonnet55"
SCALE_COMPARE = ["llm_haiku55", "llm_gpt6luna", "tfidf"]
SCALE_SUFFIX = "__sonnet30"

def scale_ref_ids(ds):
    """Query ids in each of Sonnet's runs, by transformation (empty if Sonnet has not run)."""
    ids = {}
    for t in TRANS:
        pq = RES/ds/SCALE_REF/f"{t}.parquet"
        if pq.exists():
            ids[t] = set(pd.read_parquet(pq, columns=["id_left"])["id_left"].astype(str))
    return ids

out=[]
print(f"{'dataset':14s}{'method':24s}{'trans':16s}{'F1 mean':>9s}{'std':>7s}{'P':>7s}{'R':>7s}")
for ds in CFG:
    truth=truth_set(ds); ref_ids=scale_ref_ids(ds)
    # v2 + non-LLM method dirs, plus the superseded v1 LLM dirs (results/v1_llm/), in one name-sorted pass
    mdirs=[d for root in (RES/ds, V1_LLM_RESULTS/ds) if root.exists() for d in root.iterdir()]
    for mdir in sorted(mdirs, key=lambda d: d.name):
        if not mdir.is_dir(): continue
        for t in TRANS:
            pq=mdir/f"{t}.parquet"
            if not pq.exists(): continue
            df=pd.read_parquet(pq)
            runs=[(mdir.name, df)]
            if mdir.parent==RES/ds and mdir.name in SCALE_COMPARE and t in ref_ids:
                runs.append((mdir.name+SCALE_SUFFIX, df[df["id_left"].astype(str).isin(ref_ids[t])]))
            for name, d in runs:
                true,score=true_and_score(d, truth)
                m,s,P,R=cv(true,score); out.append((ds,name,t,m,s,P,R))
                print(f"{ds:14s}{name:24s}{t:16s}{m:9.3f}{s:7.3f}{P:7.3f}{R:7.3f}")

_out = RES / "cv_results.csv"
pd.DataFrame(out,columns=["dataset","method","transformation","f1_mean","f1_std","precision","recall"]).to_csv(
    _out, index=False)
print(f"\nsaved -> {_out}")
