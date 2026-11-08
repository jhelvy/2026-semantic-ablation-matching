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
from lib.paths import DATA, RESULTS as RES, V1_LLM_RESULTS  # noqa: E402
CFG = {
 "abt-buy": dict(left="abt", right="buy", mp="abt-buy/data_original/abt_buy_perfect_mapping.csv"),
 "amazon-google": dict(left="amazon", right="google", mp="amazon-google/data_original/amazon_googleproducts_perfectmapping.csv"),
}
TRANS = ["original", "ciphered_letters", "ciphered_words", "scrambled",
         "desc_ciphered", "ids_ciphered", "all_ciphered"]   # factorial cells: used only if run
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
SEED, K = 42, 5

def load(p):
    for e in ["utf-8","latin-1","cp1252"]:
        try: return pd.read_csv(p, encoding=e)
        except Exception: pass

def truth_set(ds):
    mp = load(DATA / CFG[ds]["mp"]); return set(zip(mp.iloc[:,0].astype(str), mp.iloc[:,1].astype(str)))

def prf(true_sub, score_sub, tau):
    pred = score_sub >= tau
    tp = int((pred & true_sub).sum()); fp = int((pred & ~true_sub).sum()); fn = len(true_sub) - tp
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    return (2*P*R/(P+R) if P+R else 0.0), P, R

def cv(true, score):
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(true)); folds = np.array_split(idx, K)
    F=[];P=[];R=[]
    for f in range(K):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(K) if j!=f])
        tau = max(GRID, key=lambda t: prf(true[tr], score[tr], t)[0])
        fF,fP,fR = prf(true[te], score[te], tau); F.append(fF);P.append(fP);R.append(fR)
    return np.mean(F),np.std(F),np.mean(P),np.mean(R)

out=[]
print(f"{'dataset':14s}{'method':20s}{'trans':16s}{'F1 mean':>9s}{'std':>7s}{'P':>7s}{'R':>7s}")
for ds in CFG:
    truth=truth_set(ds)
    # v2 + non-LLM method dirs, plus the superseded v1 LLM dirs (results/v1_llm/), in one name-sorted pass
    mdirs=[d for root in (RES/ds, V1_LLM_RESULTS/ds) if root.exists() for d in root.iterdir()]
    for mdir in sorted(mdirs, key=lambda d: d.name):
        if not mdir.is_dir(): continue
        for t in TRANS:
            pq=mdir/f"{t}.parquet"
            if not pq.exists(): continue
            df=pd.read_parquet(pq)
            pid=df["pred_id_right"].astype(str).values; lid=df["id_left"].astype(str).values
            score=df["similarity_score"].fillna(0).values
            score=np.where((pid!="")&(pid!="None"),score,0.0)
            true=np.array([(lid[i],pid[i]) in truth for i in range(len(df))])
            m,s,P,R=cv(true,score); out.append((ds,mdir.name,t,m,s,P,R))
            print(f"{ds:14s}{mdir.name:20s}{t:16s}{m:9.3f}{s:7.3f}{P:7.3f}{R:7.3f}")

_out = RES / "cv_results.csv"
pd.DataFrame(out,columns=["dataset","method","transformation","f1_mean","f1_std","precision","recall"]).to_csv(
    _out, index=False)
print(f"\nsaved -> {_out}")
