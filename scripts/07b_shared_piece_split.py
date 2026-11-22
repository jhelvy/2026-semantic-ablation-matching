"""
SHARED-PIECE SPLIT (no API calls)
=================================

Why do LLMs do better on all_ciphered (factorial, both factors) than on v1
ciphered_words, when both remove meaning? The two transforms treat two kinds of
evidence differently:

  * identifier pieces ("pslx350h"): all_ciphered keeps their code shape
    ("qbzt817m"); ciphered_words maps the whole token to a dictionary word;
  * quantity pieces (factorial descriptive pieces with a digit, "4gb"):
    all_ciphered keeps the digits ("4ku"); ciphered_words maps the token to a word.

Each query is put in one group by what it shares with its true match in the
original text the LLM sees: "id" (an identifier piece), else "qty" (a quantity
piece), else "neither". For every LLM run on the shared 30% queries (Sonnet's ids)
and both conditions, the table gives LLM accuracy minus TF-IDF top-1 on the
model's own shortlist, on all queries and on answered queries (status ok).
Descriptive only: the answered set is chosen by the model, and some cells are small.

Output: results/shared_piece_split.csv

Usage:
    uv run scripts/07b_shared_piece_split.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.datasets import DATASETS, load_csv  # noqa: E402
from lib.paths import DATA, RESULTS as RES  # noqa: E402

_spec = importlib.util.spec_from_file_location("ia", HERE / "07_instance_analysis.py")
ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ia)   # llm_text, PIECE_RE: the text the LLM sees, as in 07

DATASETS_RUN = ["abt-buy", "amazon-google", "wdc-products"]
MODELS = ["llm_gpt6luna", "llm_gpt56luna", "llm_haiku55", "llm_haiku45", "llm_sonnet55", "llm_kimik3"]
CONDS = ["ciphered_words", "all_ciphered"]
SUBSAMPLE_REF = "llm_sonnet55"   # its query ids define the shared 30% set


def pieces_by_record(ds):
    cfg = DATASETS[ds]
    out = {}
    for side in ("left", "right"):
        df = load_csv(cfg["dir"] / "data_original" / f"{cfg[side + '_name']}.csv")
        text = ia.llm_text(df, cfg["title_only"])
        out[side] = {str(i): set(ia.PIECE_RE.findall(t)) for i, t in zip(df.iloc[:, 0].astype(str), text)}
    return out


def main():
    rows = []
    for ds in DATASETS_RUN:
        man = json.loads((DATA / ds / "data_test" / "factorial_manifest.json").read_text())
        id_pieces = set(man["id_map"])
        qty_pieces = {k for k in man["desc_map"] if any(c.isdigit() for c in k)}
        P = pieces_by_record(ds)
        ref = RES / ds / SUBSAMPLE_REF / "original.parquet"
        ids30 = set(pd.read_parquet(ref, columns=["id_left"])["id_left"].astype(str))
        for m in MODELS:
            for t in CONDS:
                pq = RES / ds / m / f"{t}.parquet"
                if not pq.exists():
                    continue
                d = pd.read_parquet(pq)
                d = d[d["id_left"].astype(str).isin(ids30)]
                shared = [P["left"].get(l, set()) & P["right"].get(r, set())
                          for l, r in zip(d["id_left"].astype(str), d["true_id_right"].astype(str))]
                group = np.array(["id" if s & id_pieces else "qty" if s & qty_pieces else "neither"
                                  for s in shared])
                acc = d["is_correct"].astype(float).values
                top1 = (d["true_rank_in_candidates"] == 1).astype(float).values
                answered = (d["status"] == "ok").values
                for g in ("id", "qty", "neither"):
                    in_g = group == g
                    a = in_g & answered
                    rows.append(dict(
                        dataset=ds, method=m, transformation=t, group=g, n=int(in_g.sum()),
                        n_answered=int(a.sum()),
                        llm_correct=acc[in_g].mean() if in_g.any() else np.nan,
                        tfidf_top1_correct=top1[in_g].mean() if in_g.any() else np.nan,
                        gap=(acc - top1)[in_g].mean() if in_g.any() else np.nan,
                        gap_answered=(acc - top1)[a].mean() if a.any() else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "shared_piece_split.csv", index=False)

    piv = out.pivot_table(index=["dataset", "group"], columns="transformation", values="gap_answered")
    piv["all_minus_cw"] = piv["all_ciphered"] - piv["ciphered_words"]
    piv["n"] = out[out.transformation == "all_ciphered"].groupby(["dataset", "group"])["n"].first()
    w = out.pivot_table(index=["dataset", "group", "method"], columns="transformation", values="gap_answered")
    pos = (w["all_ciphered"] > w["ciphered_words"]).groupby(level=[0, 1])
    piv["models_all_gt_cw"] = pos.sum().astype(int).astype(str) + "/" + pos.size().astype(str)
    print("Answered-query gap (LLM minus TF-IDF top-1), mean over models, points:")
    print((piv[["all_ciphered", "ciphered_words", "all_minus_cw"]] * 100).round(1)
          .join(piv[["n", "models_all_gt_cw"]]).to_string())
    print(f"saved -> {RES / 'shared_piece_split.csv'}")


if __name__ == "__main__":
    main()
