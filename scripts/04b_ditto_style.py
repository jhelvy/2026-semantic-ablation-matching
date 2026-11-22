"""
DITTO-STYLE SUPERVISED BASELINE (D.8; local, no API calls)
==========================================================

A Ditto-style cross-encoder (Li et al. 2020): each (query, candidate) pair is serialized
as "COL name VAL ... COL description VAL ...", the two records are joined as a RoBERTa
sentence pair, and RoBERTa-base is fine-tuned for binary match classification. None of
Ditto's optimizations (domain-knowledge injection, TF-IDF summarization, augmentation)
are used, so the paper calls it "Ditto-style".

Task shape, the same as the LLMs: TF-IDF gives each query a top-50 shortlist
(llm_runner.build_requests, all queries), the model scores every candidate, and the
argmax is the prediction; its match probability is the score 05_cv_threshold.py
thresholds. 5-fold cross-validation over queries (seed 42): each query is predicted by
the model that held it out. Training pairs are each training query's true match(es)
plus 9 non-matches drawn from its shortlist; 10% of the training queries are held out
to choose the epoch (top-1 accuracy over their full shortlists). AdamW, lr 3e-5, linear
warmup over the first 10% of steps, then linear decay (with no warmup, the fits on
Ciphered Words stayed at the class-prior loss).

Two regimes:
  A  ditto_style          train on Original, test on all seven conditions.
  B  ditto_style_matched  train and test on the same condition (ciphered_words,
                          all_ciphered); its original row is copied from regime A.

Per-fold predictions checkpoint to checkpoints/ditto/, so a run resumes by fold.
Output: results/{dataset}/ditto_style{,_matched}/{transformation}.parquet, with the
usual columns (is_correct = top-1 correct; predicted_match = probability >= 0.5).

Usage:
    uv run scripts/04b_ditto_style.py --dataset abt-buy --regime A --folds 0   # time one fold
    uv run scripts/04b_ditto_style.py --dataset abt-buy,amazon-google,wdc-products
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import methods  # noqa: E402
from lib.datasets import DATASETS, FACTORIAL_CELLS, TRANSFORMATIONS, load_dataset  # noqa: E402
from lib.ditto import query_folds, serialize, split_validation, training_pairs  # noqa: E402
from lib.llm_runner import build_requests  # noqa: E402
from lib.paths import CHECKPOINTS, RESULTS  # noqa: E402

SEED, K, N_NEG, TOP_K = 42, 5, 9, 50
MAX_LEN, LR, BATCH, EVAL_BATCH = 256, 3e-5, 32, 128
WARMUP = 0.1   # share of steps with linear warmup; without it, fits on ciphered text collapsed
CONDS = TRANSFORMATIONS + FACTORIAL_CELLS
MATCHED = ["ciphered_words", "all_ciphered"]
METHOD_A, METHOD_B = "ditto_style", "ditto_style_matched"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_condition(ds, t):
    """Shortlists, serialized records and truths for one (dataset, condition)."""
    title_only = DATASETS[ds]["title_only"]
    methods.set_title_only(title_only)
    df_l, df_r, mp = load_dataset(ds, t)
    reqs, _ = build_requests(df_l, df_r, mp, top_k=TOP_K)
    truths = {}
    for a, b in methods.create_ground_truth_set(mp):
        truths.setdefault(a, []).append(b)
    return dict(
        reqs={r["id_left"]: r for r in reqs},
        L=dict(zip(df_l.iloc[:, 0].astype(str), serialize(df_l, title_only))),
        R=dict(zip(df_r.iloc[:, 0].astype(str), serialize(df_r, title_only))),
        truths={k: sorted(v) for k, v in truths.items()})


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tok, a, b, order, size, device, labels=None):
    for s in range(0, len(order), size):
        idx = order[s:s + size]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in idx], device=device)
        yield idx, enc


@torch.no_grad()
def score_queries(model, tok, C, qids, device):
    """{query id: match probability for each shortlist candidate, in shortlist order}."""
    model.eval()
    a, b, owner = [], [], []
    for q in qids:
        for c in C["reqs"][q]["candidate_ids"]:
            a.append(C["L"][q]); b.append(C["R"][c]); owner.append(q)
    lens = np.array([len(x) + len(y) for x, y in zip(a, b)])
    order = list(np.argsort(lens, kind="stable"))        # similar lengths per batch: less padding
    prob = np.zeros(len(a))
    for idx, enc in batches(tok, a, b, order, EVAL_BATCH, device):
        prob[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
    out, pos = {}, 0
    for q in qids:
        n = len(C["reqs"][q]["candidate_ids"])
        out[q] = prob[pos:pos + n]; pos += n
    return out


def top1(C, probs):
    hits = [C["reqs"][q]["candidate_ids"][int(np.argmax(p))] in C["truths"][q] for q, p in probs.items()]
    return float(np.mean(hits))


def train(C, train_q, val_q, tok, model_name, device, epochs, seed):
    """Fine-tune on train_q's pairs; keep the epoch with the best validation top-1."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    a, b, y = [], [], []
    for q in train_q:
        for r, lab in training_pairs(q, C["reqs"][q]["candidate_ids"], C["truths"][q], N_NEG, SEED):
            a.append(C["L"][q]); b.append(C["R"][r]); y.append(lab)
    steps = epochs * -(-len(a) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    rng = np.random.default_rng(seed)
    best, best_acc, hist = None, -1.0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); loss_sum = 0.0
        for _, enc in batches(tok, a, b, list(rng.permutation(len(a))), BATCH, device, y):
            loss = model(**enc).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            loss_sum += loss.item()
        acc = top1(C, score_queries(model, tok, C, val_q, device))
        hist.append(dict(epoch=ep, loss=loss_sum / -(-len(a) // BATCH), val_top1=acc,
                         seconds=round(time.time() - t0, 1)))
        print(f"      epoch {ep}: loss {hist[-1]['loss']:.4f}  val top-1 {acc:.3f}  "
              f"({hist[-1]['seconds']:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best_ep = ep
    model.load_state_dict(best)
    return model, dict(best_epoch=best_ep, n_train_pairs=len(a), n_val=len(val_q), history=hist)


# ---------------------------------------------------------------------------
# rows and files
# ---------------------------------------------------------------------------

def rows_for(C, probs, ds, t, method, fold, info):
    out = []
    for q, p in probs.items():
        r = C["reqs"][q]; j = int(np.argmax(p)); pred = r["candidate_ids"][j]
        out.append(dict(
            id_left=q, left_name=r["left_name"], true_id_right=r["true_id_right"],
            pred_id_right=pred, pred_right_name=r["candidate_names"][j],
            similarity_score=float(p[j]), predicted_match=int(p[j] >= 0.5),
            is_correct=int(pred in C["truths"][q]),
            true_rank_in_candidates=r["true_rank_in_candidates"], n_candidates=r["n_candidates"],
            fold=fold, best_epoch=info["best_epoch"], method=method, transformation=t, dataset=ds))
    return out


def ck_path(ds, method, t, fold):
    return CHECKPOINTS / "ditto" / ds / method / t / f"fold{fold}.parquet"


def assemble(ds, method, t, wall):
    """Write results/ once all K folds of (method, t) exist."""
    parts = [ck_path(ds, method, t, f) for f in range(K)]
    if not all(p.exists() for p in parts):
        return False
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["timestamp"] = datetime.now().isoformat()
    df["wall_time_seconds"] = wall
    out = RESULTS / ds / method / f"{t}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  saved {out.relative_to(RESULTS.parent)}  top-1 {df['is_correct'].mean():.3f}  n={len(df)}")
    return True


def run_fold(ds, method, train_t, test_ts, fold, folds, cache, tok, args, device, log):
    todo = [t for t in test_ts if not ck_path(ds, method, t, fold).exists()]
    if not todo:
        return
    C = cache(train_t)
    tr_all = [q for q in C["reqs"] if folds[q] != fold]
    te = sorted(q for q in C["reqs"] if folds[q] == fold)
    tr, val = split_validation(tr_all, 0.1, SEED + fold)
    print(f"  {ds} {method} fold {fold}: train {len(tr)} queries, val {len(val)}, test {len(te)}"
          f" (trained on {train_t})", flush=True)
    t0 = time.time()
    model, info = train(C, tr, val, tok, args.model, device, args.epochs, SEED + fold)
    info["train_seconds"] = round(time.time() - t0, 1)
    for t in todo:
        t1 = time.time()
        Ct = cache(t)
        probs = score_queries(model, tok, Ct, te, device)
        p = ck_path(ds, method, t, fold); p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_for(Ct, probs, ds, t, method, fold, info)).to_parquet(p, index=False)
        print(f"    test {t:18s} top-1 {top1(Ct, probs):.3f}  ({time.time() - t1:.0f}s)", flush=True)
    log.append(dict(dataset=ds, method=method, train_on=train_t, fold=fold,
                    seconds=round(time.time() - t0, 1), **info))
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="comma list")
    ap.add_argument("--regime", default="A,B")
    ap.add_argument("--folds", default=",".join(map(str, range(K))))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--matched", default=",".join(MATCHED), help="regime B conditions")
    a = ap.parse_args()
    regimes, fold_ids = a.regime.split(","), [int(f) for f in a.folds.split(",")]
    device = device_auto()
    tok = AutoTokenizer.from_pretrained(a.model)
    print(f"device {device}, model {a.model}, epochs {a.epochs}, max_len {MAX_LEN}, batch {BATCH}, lr {LR}, warmup {WARMUP}")

    for ds in a.dataset.split(","):
        assert ds in DATASETS, ds
        memo = {}

        def cache(t):
            if t not in memo:
                memo[t] = load_condition(ds, t)
            return memo[t]

        folds = query_folds(cache("original")["reqs"].keys(), K, SEED)
        log, t0 = [], time.time()
        for f in fold_ids:
            if "A" in regimes:
                run_fold(ds, METHOD_A, "original", CONDS, f, folds, cache, tok, a, device, log)
            if "B" in regimes:
                for t in a.matched.split(","):
                    run_fold(ds, METHOD_B, t, [t], f, folds, cache, tok, a, device, log)
        wall = round(time.time() - t0, 1)
        if log:
            lp = CHECKPOINTS / "ditto" / ds / f"log_{datetime.now():%Y%m%dT%H%M%S}.json"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(json.dumps(log, indent=1))
        for t in CONDS:
            assemble(ds, METHOD_A, t, wall)
        for t in MATCHED:
            assemble(ds, METHOD_B, t, wall)
        src = RESULTS / ds / METHOD_A / "original.parquet"
        if src.exists() and all((RESULTS / ds / METHOD_B / f"{t}.parquet").exists() for t in MATCHED):
            d = pd.read_parquet(src); d["method"] = METHOD_B
            d.to_parquet(RESULTS / ds / METHOD_B / "original.parquet", index=False)
            print(f"  copied regime A original -> {METHOD_B}/original.parquet")


if __name__ == "__main__":
    main()
