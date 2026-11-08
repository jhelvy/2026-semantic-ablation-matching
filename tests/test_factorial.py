"""
Offline tests for scripts/lib/factorial.py (no data files, no API).

    python tests/test_factorial.py
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from lib import factorial as pf  # noqa: E402

ENGLISH = {"sony", "camera", "black", "digital", "pixel", "year", "warranty", "with"}


def _toy():
    left = pd.DataFrame({"id": ["a1", "a2", "a3"],
                         "name": ["Sony DSC-W55 Digital Camera Black",
                                  "Sony Bravia KDL-46V3000 46in LCD",
                                  "Canon 12-pixel camera with 1-year warranty 4GB"]})
    right = pd.DataFrame({"id": ["b1", "b2", "b3"],
                          "name": ["sony dsc-w55 camera", "kdl-46v3000 sony tv",
                                   "canon camera 1024x768 4gb"]})
    return left, right


def _maps(left, right):
    vocab, id_p, kept, desc, rdf = pf.classify_pieces(left, right, max_id_df=20, english=ENGLISH)
    rng = random.Random("test")
    used = set()
    dm, _ = pf.build_map(desc, pf._nonce, rng, vocab, ENGLISH, used)
    im, _ = pf.build_map(id_p, pf._code, rng, vocab, ENGLISH, used)
    return vocab, id_p, kept, desc, dm, im


def test_classification():
    vocab, id_p, kept, desc, _, _ = _maps(*_toy())
    assert {"w55", "46v3000", "dsc", "kdl"} <= id_p          # codes and short glued prefixes
    assert "pixel" not in id_p and "warranty" not in id_p    # English words glued to digits
    assert "4gb" not in id_p and "1024x768" not in id_p      # quantities / dimensions
    assert {"sony", "camera", "bravia", "4gb"} <= desc
    assert {"12", "1"} <= kept
    assert id_p.isdisjoint(desc) and id_p.isdisjoint(kept) and desc.isdisjoint(kept)


def test_maps_injective_and_disjoint_from_vocab():
    vocab, _, _, _, dm, im = _maps(*_toy())
    outs = list(dm.values()) + list(im.values())
    assert len(outs) == len(set(outs))
    assert not set(outs) & vocab
    for p, c in im.items():                                   # shape-preserving code cipher
        assert len(c) == len(p)
        assert all(a.isdigit() == b.isdigit() for a, b in zip(p, c))
    for p, c in dm.items():                                   # digits kept in descriptive pieces
        assert [ch for ch in p if ch.isdigit()] == [ch for ch in c if ch.isdigit()]


def test_cells():
    left, right = _toy()
    *_, dm, im = _maps(left, right)
    d = pf.apply_cell(left, dm, im, True, False)["name"][0]
    i = pf.apply_cell(left, dm, im, False, True)["name"][0]
    a = pf.apply_cell(left, dm, im, True, True)["name"][0]
    assert d.split()[1] == "dsc-w55" and "sony" not in d       # meaning gone, codes intact
    assert i.startswith("sony ") and "w55" not in i            # meaning intact, codes ciphered
    assert "sony" not in a and "w55" not in a
    assert i.split()[1] == a.split()[1]                        # same code map in both cells
    # identity across catalogs preserved: the shared code maps to the same string on both sides
    r = pf.apply_cell(right, dm, im, True, True)["name"][0]
    assert a.split()[1] == r.split()[1]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
