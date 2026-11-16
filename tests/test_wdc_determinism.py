"""
Determinism tests for the WDC build (no API; needs /usr/share/dict/words, and the cached
zip data/wdc-products/download/80multi.zip for the full-build test, which is skipped if absent).

    uv run tests/test_wdc_determinism.py
    uv run pytest tests/test_wdc_determinism.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ZIP = ROOT / "data" / "wdc-products" / "download" / "80multi.zip"
DICT = Path("/usr/share/dict/words")

# Words of many lengths, so several length classes exceed the dictionary's supply or not.
_PROBE = r'''
import json, random, sys
sys.path.insert(0, sys.argv[1])
import pandas as pd
from lib.transforms import generate_ciphered_words, generate_ciphered_letters
words = [f"w{i}x{i*7}" for i in range(300)] + [f"word{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}" for i in range(400)]
left = pd.DataFrame({"id": range(200), "title": [" ".join(words[i::7][:6]) for i in range(200)]})
right = pd.DataFrame({"id": range(100), "title": [" ".join(words[i::5][:6]) for i in range(100)]})
random.seed(42)
cl = generate_ciphered_letters(left, right, {})
cw = generate_ciphered_words(left, right, {}, sort_words=(sys.argv[2] == "1"))
print("@@" + json.dumps([cl[0].title.tolist(), cw[0].title.tolist(), cw[1].title.tolist()]))
'''


def _probe(hashseed, sort_words):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, "-c", _PROBE, str(SCRIPTS), "1" if sort_words else "0"],
                         env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out.split("@@")[1])


def test_sort_words_is_hashseed_independent():
    if not DICT.exists():
        print("skip: no /usr/share/dict/words")
        return
    a, b = _probe(1, True), _probe(98765, True)
    assert a == b


def test_default_path_still_depends_on_hashseed():
    # Documents why the committed Abt-Buy / Amazon-Google CSVs cannot be regenerated.
    if not DICT.exists():
        print("skip: no /usr/share/dict/words")
        return
    a, b = _probe(1, False), _probe(98765, False)
    assert a[0] == b[0]          # letter cipher does not depend on hash order
    assert a[1] != b[1]          # word map does


def _build(out_dir, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    subprocess.run([sys.executable, str(SCRIPTS / "01b_prepare_wdc.py"), "--zip", str(ZIP),
                    "--out-dir", str(out_dir)], env=env, capture_output=True, text=True, check=True)


def test_01b_byte_identical_across_hashseeds():
    if not (ZIP.exists() and DICT.exists()):
        print("skip: cached zip or dictionary missing")
        return
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        _build(t1, 1)
        _build(t2, 98765)
        files1 = sorted(p.relative_to(t1) for p in Path(t1).rglob("*.csv"))
        files2 = sorted(p.relative_to(t2) for p in Path(t2).rglob("*.csv"))
        assert files1 == files2 and len(files1) == 9, files1    # 3 original + 6 transformed
        for rel in files1:
            assert (Path(t1) / rel).read_bytes() == (Path(t2) / rel).read_bytes(), rel


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
