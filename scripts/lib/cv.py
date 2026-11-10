"""
Cross-validated threshold F1, shared by 05_cv_threshold.py and 07_instance_analysis.py.

5-fold CV: choose tau on 4 folds, evaluate on the held-out fold, report mean +/- std
F1 and mean P/R. Every reported F1 comes from these functions.
"""
import numpy as np
import pandas as pd

from .paths import DATA

CFG = {
    "abt-buy": dict(left="abt", right="buy", mp="abt-buy/data_original/abt_buy_perfect_mapping.csv"),
    "amazon-google": dict(left="amazon", right="google",
                          mp="amazon-google/data_original/amazon_googleproducts_perfectmapping.csv"),
}
TRANS = ["original", "ciphered_letters", "ciphered_words", "scrambled",
         "desc_ciphered", "ids_ciphered", "all_ciphered"]   # factorial cells: used only if run
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
SEED, K = 42, 5


def load(p):
    for e in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(p, encoding=e)
        except Exception:
            pass


def truth_set(ds):
    mp = load(DATA / CFG[ds]["mp"])
    return set(zip(mp.iloc[:, 0].astype(str), mp.iloc[:, 1].astype(str)))


def prf(true_sub, score_sub, tau):
    pred = score_sub >= tau
    tp = int((pred & true_sub).sum()); fp = int((pred & ~true_sub).sum()); fn = len(true_sub) - tp
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * R / (P + R) if P + R else 0.0), P, R


def cv(true, score):
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(true)); folds = np.array_split(idx, K)
    F = []; P = []; R = []
    for f in range(K):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(K) if j != f])
        tau = max(GRID, key=lambda t: prf(true[tr], score[tr], t)[0])
        fF, fP, fR = prf(true[te], score[te], tau); F.append(fF); P.append(fP); R.append(fR)
    return np.mean(F), np.std(F), np.mean(P), np.mean(R)


def true_and_score(df, truth):
    pid = df["pred_id_right"].astype(str).values; lid = df["id_left"].astype(str).values
    score = df["similarity_score"].fillna(0).values
    score = np.where((pid != "") & (pid != "None"), score, 0.0)
    true = np.array([(lid[i], pid[i]) in truth for i in range(len(df))])
    return true, score


def cv_f1(df, truth):
    """(f1_mean, f1_std, precision, recall) for one result table."""
    return cv(*true_and_score(df, truth))
