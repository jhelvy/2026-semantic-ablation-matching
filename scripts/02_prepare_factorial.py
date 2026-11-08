"""
2x2 FACTORIAL ABLATION: descriptive words x identifiers
=======================================================
Writes the desc_ciphered, ids_ciphered, and all_ciphered cells plus a manifest
to data/<dataset>/data_test/. Design, token classes, and substitution rules are
documented in scripts/lib/factorial.py.

Usage:
    python scripts/02_prepare_factorial.py --dataset all
    python scripts/02_prepare_factorial.py --dataset abt-buy --seed 7     # extra seed -> *_s7.csv
    python scripts/02_prepare_factorial.py --dataset all --verify         # + TF-IDF invariance check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.factorial import DATASETS, process  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate the 2x2 descriptive-words x identifiers ablation")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS) + ["all"])
    ap.add_argument("--seed", type=int, default=42, help="non-default seeds write *_s<seed>.csv")
    ap.add_argument("--max-id-df", type=int, default=20,
                    help="max number of records an identifier piece may appear in")
    ap.add_argument("--verify", action="store_true", help="also run v1 TF-IDF on every cell")
    args = ap.parse_args()
    for ds in (DATASETS if args.dataset == "all" else [args.dataset]):
        if not (DATASETS[ds]["dir"] / DATASETS[ds]["files"]["left"]).exists():
            print(f"skip {ds}: data not present")
            continue
        process(ds, args.seed, args.max_id_df, args.verify)


if __name__ == "__main__":
    main()
