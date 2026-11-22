"""Offline tests for the Ditto-style helpers (lib/ditto.py). No model is loaded."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lib.ditto import query_folds, serialize, split_validation, text_columns, training_pairs  # noqa: E402


def test_serialization_matches_text_columns():
    df = pd.DataFrame({"id": [1, 2], "name": ["Sony PS-LX350H", "Bose AM5"],
                       "description": ["Belt Drive", ""], "price": ["$99", "$1"]})
    assert text_columns(df, title_only=False) == ["name", "description"]
    assert text_columns(df, title_only=True) == ["name"]
    s = serialize(df, title_only=False)
    assert s.iloc[0] == "COL name VAL sony ps-lx350h COL description VAL belt drive"
    assert s.iloc[1] == "COL name VAL bose am5 COL description VAL"   # empty value keeps its tag
    assert serialize(df, title_only=True).iloc[0] == "COL name VAL sony ps-lx350h"


def test_column_with_missing_values_is_dropped_as_in_prepare_text_column():
    # pandas 2.2: an object column holding None/NaN is not a string dtype, so
    # prepare_text_column skips it (Buy's description on Abt-Buy). Mirror that.
    from lib.methods import prepare_text_column
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "description": ["x", None]})
    assert text_columns(df, title_only=False) == ["name"]
    assert prepare_text_column(df)["text"].tolist() == ["a", "b"]


def test_folds_partition_and_are_order_free():
    ids = [str(i) for i in range(103)]
    f1 = query_folds(ids, 5, 42)
    f2 = query_folds(list(reversed(ids)) + ids[:10], 5, 42)    # order and duplicates ignored
    assert f1 == f2
    sizes = pd.Series(f1).value_counts()
    assert set(sizes.index) == set(range(5)) and sizes.max() - sizes.min() <= 1
    assert query_folds(ids, 5, 7) != f1


def test_validation_split():
    tr, val = split_validation([str(i) for i in range(50)], 0.1, 1)
    assert len(val) == 5 and len(tr) == 45 and not set(tr) & set(val)
    assert split_validation([str(i) for i in range(50)], 0.1, 1) == (tr, val)


def test_training_pairs():
    cands = [f"c{i}" for i in range(50)]
    p = training_pairs("q1", cands, ["c3"], 9, 42)
    assert p[0] == ("c3", 1) and len(p) == 10
    negs = [c for c, y in p if y == 0]
    assert len(set(negs)) == 9 and "c3" not in negs and set(negs) <= set(cands)
    assert training_pairs("q1", cands, ["c3"], 9, 42) == p           # seeded per query
    assert training_pairs("q2", cands, ["c3"], 9, 42) != p
    # one-to-many: every truth is a positive and none is a negative
    p2 = training_pairs("q1", cands, ["c3", "c7"], 9, 42)
    assert [c for c, y in p2 if y == 1] == ["c3", "c7"]
    assert not {"c3", "c7"} & {c for c, y in p2 if y == 0}
    # truth blocked out of the shortlist is still a positive
    p3 = training_pairs("q1", cands, ["zz"], 9, 42)
    assert ("zz", 1) in p3 and len(p3) == 10
    # short shortlist: take what there is
    assert len(training_pairs("q1", ["c0", "c1", "c2"], ["c0"], 9, 42)) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
