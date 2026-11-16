"""
BUILD THE WDC PRODUCTS DATASET (third dataset)
==============================================
Downloads WDC Products (80% corner cases, multi-class, 0% unseen), checks the sha256,
writes data/wdc-products/data_original/{offers,catalog}.csv and wdc_mapping.csv, then
the three v1 transforms (ciphered letters, ciphered words, scrambled) to data_test/.

Differences from 01_prepare_datasets.py (which refuses wdc-products):
  * the Ciphered Words map is built with sort_words=True, so the output does not depend
    on PYTHONHASHSEED. Two runs give byte-identical CSVs (tests/test_wdc_determinism.py).
  * the transforms run in the v1 order (letters, words, scrambled) in a fresh process with
    the global random seed reset to 42, so the letter cipher is the same on every run.
The Ciphered Words map also depends on /usr/share/dict/words (first 1,500 words per
length); its sha256 is recorded in data_test/wdc_build.json.

Data files are gitignored (licence unsettled); rebuild them with this script.

Usage:
    uv run scripts/01b_prepare_wdc.py
    uv run scripts/01b_prepare_wdc.py --zip path/to/80multi.zip --out-dir /tmp/wdc_check
"""

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.transforms import (DATASETS, generate_ciphered_letters, generate_ciphered_words,  # noqa: E402
                            generate_scrambled)

URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip"
SHA256 = "5385e2491e82c7f6c208d8eac7ec777a4a44180769f3a339c99803ff2b068981"
TEST_MEMBER = "wdcproductsmulti80cc20rnd000un_gs.json.gz"
CATALOG_MEMBER = "wdcproductsmulti80cc20rnd000un_train_large.json.gz"
DICT_PATH = Path("/usr/share/dict/words")
NAME = "wdc-products"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_zip(zip_arg, cache_dir):
    if zip_arg:
        path = Path(zip_arg)
    else:
        path = cache_dir / "80multi.zip"
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"downloading {URL}")
            with urllib.request.urlopen(URL) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
    got = sha256_of(path)
    if got != SHA256:
        raise SystemExit(f"sha256 mismatch for {path}: {got} (expected {SHA256})")
    print(f"zip ok: {path} ({got[:12]}...)")
    return path


def read_member(zf, member):
    with zf.open(member) as raw:
        return [json.loads(line) for line in gzip.open(raw, "rt", encoding="utf-8")]


def to_frame(rows):
    return pd.DataFrame({
        "id": [r["id"] for r in rows],
        "title": [r["title"] for r in rows],
        "description": [r["description"] for r in rows],
        "manufacturer": [r["brand"] for r in rows],
        "price": [r["price"] for r in rows],
    })


def build_original(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        queries = read_member(zf, TEST_MEMBER)
        offers = read_member(zf, CATALOG_MEMBER)

    # catalog: lowest-id offer of each product (deterministic, blind to text)
    first = {}
    for r in offers:
        c = r["cluster_id"]
        if c not in first or r["id"] < first[c]["id"]:
            first[c] = r
    catalog_rows = sorted(first.values(), key=lambda r: r["id"])

    missing = {q["cluster_id"] for q in queries} - set(first)
    if missing:
        raise SystemExit(f"{len(missing)} test products have no catalog offer")
    mapping = pd.DataFrame({
        "id_offers": [q["id"] for q in queries],
        "id_catalog": [first[q["cluster_id"]]["id"] for q in queries],
    })
    return to_frame(queries), to_frame(catalog_rows), mapping


def main():
    ap = argparse.ArgumentParser(description="Build the WDC Products dataset and its v1 transforms")
    ap.add_argument("--zip", help="path to 80multi.zip (default: download to <out-dir>/download/)")
    ap.add_argument("--out-dir", help="default: data/wdc-products")
    args = ap.parse_args()

    cfg = DATASETS[NAME]
    out = Path(args.out_dir) if args.out_dir else cfg["dir"]
    if not DICT_PATH.exists():
        raise SystemExit(f"{DICT_PATH} not found: Ciphered Words would fall back to generated words")

    zip_path = get_zip(args.zip, out / "download")
    offers, catalog, mapping = build_original(zip_path)
    print(f"offers {len(offers)}, catalog {len(catalog)}, mapping {len(mapping)}")
    assert len(offers) == 1000 and len(catalog) == 500 and len(mapping) == 1000
    assert offers["id"].is_unique and catalog["id"].is_unique

    orig = out / "data_original"
    test = out / "data_test"
    orig.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)
    offers.to_csv(orig / "offers.csv", index=False)
    catalog.to_csv(orig / "catalog.csv", index=False)
    mapping.to_csv(orig / "wdc_mapping.csv", index=False)

    # Re-read from disk so the transforms see exactly what the experiments load.
    df_l = pd.read_csv(orig / "offers.csv")
    df_r = pd.read_csv(orig / "catalog.csv")

    random.seed(42)  # same state as a fresh import; makes the letter cipher run-independent
    transformed = {}
    transformed["ciphered_letters"] = generate_ciphered_letters(df_l, df_r, cfg)
    transformed["ciphered_words"] = generate_ciphered_words(df_l, df_r, cfg, sort_words=True)
    transformed["scrambled"] = generate_scrambled(df_l, df_r, cfg)
    for name, (left, right) in transformed.items():
        left.to_csv(test / f"{cfg['left_name']}_{name}.csv", index=False)
        right.to_csv(test / f"{cfg['right_name']}_{name}.csv", index=False)
        print(f"wrote {name}")

    (test / "wdc_build.json").write_text(json.dumps(dict(
        zip_sha256=SHA256, dict_words_sha256=sha256_of(DICT_PATH),
        offers=len(offers), catalog=len(catalog), mapping=len(mapping),
        ciphered_words_sort_words=True, random_seed=42), indent=1) + "\n")


if __name__ == "__main__":
    main()
