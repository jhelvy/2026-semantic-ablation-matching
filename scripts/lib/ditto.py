"""
Pure helpers for the Ditto-style baseline (04b_ditto_style.py): record serialization,
query folds, and training-pair sampling. No torch import here, so the offline tests
run without loading a model.
"""
import zlib

import numpy as np
import pandas as pd


def text_columns(df, title_only):
    """The columns prepare_text_column joins, in the same order (methods.py)."""
    if title_only:
        for col in df.columns[1:]:
            if col.lower() in ("title", "name"):
                return [col]
        return []
    return [c for c in df.columns[1:]
            if pd.api.types.is_string_dtype(df[c]) and c.lower() not in ("year", "price", "id")]


def serialize(df, title_only):
    """Ditto serialization, 'COL name VAL ... COL description VAL ...', one string per row.

    Values are lowercased and stripped, as in prepare_text_column, so the model sees the
    same text as every other method; empty values keep their COL tag, as in Ditto.
    """
    cols = text_columns(df, title_only)
    parts = [("COL " + c + " VAL " + df[c].fillna("").astype(str).str.lower().str.strip())
             for c in cols]
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.strip()


def query_folds(ids, k=5, seed=42):
    """{query id: fold} from a seeded permutation of the sorted unique ids."""
    ids = np.array(sorted({str(i) for i in ids}))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[j]: f for f, chunk in enumerate(np.array_split(perm, k)) for j in chunk}


def split_validation(train_ids, frac=0.1, seed=42):
    """Hold out a seeded share of the training queries to choose the epoch."""
    ids = np.array(sorted(train_ids))
    n_val = max(1, int(round(len(ids) * frac)))
    val = set(np.random.default_rng(seed).choice(ids, size=n_val, replace=False))
    return [i for i in ids if i not in val], sorted(val)


def training_pairs(qid, candidate_ids, truths, n_neg=9, seed=42):
    """[(right id, label)]: every true match, plus n_neg non-matches from the shortlist.

    Negatives are drawn uniformly without replacement from the shortlist's non-matches,
    with a stream seeded by (seed, query id), so the draw does not depend on run order.
    True matches are positives even when TF-IDF blocked them out of the shortlist.
    """
    truths = sorted(set(truths))
    pool = [c for c in candidate_ids if c not in truths]
    rng = np.random.default_rng([seed, zlib.crc32(str(qid).encode())])
    neg = list(rng.choice(pool, size=min(n_neg, len(pool)), replace=False)) if pool else []
    return [(t, 1) for t in truths] + [(str(c), 0) for c in neg]
