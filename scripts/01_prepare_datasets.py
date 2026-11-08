"""
MASTER DATASET TRANSFORMATION GENERATOR
========================================
Generates the three v1 transformations for any dataset:
1. Ciphered Letters (character substitution)
2. Ciphered Words (word replacement)
3. Scrambled (letter scrambling within words)

Writes data/<dataset>/data_test/<side>_<transformation>.csv. Transform code is in
scripts/lib/transforms.py.

Do not rerun this to check the committed CSVs: the Ciphered Words map depends on
PYTHONHASHSEED, so a rerun overwrites the map the paper used.

Usage:
    uv run scripts/01_prepare_datasets.py --dataset abt-buy
    uv run scripts/01_prepare_datasets.py --dataset amazon-google
    uv run scripts/01_prepare_datasets.py --dataset all  # Generate for all datasets
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.transforms import DATASETS, process_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description='Generate all dataset transformations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/01_prepare_datasets.py --dataset abt-buy
  uv run scripts/01_prepare_datasets.py --dataset all
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()) + ['all'],
        help='Dataset to process (or "all" for all datasets)'
    )

    args = parser.parse_args()

    print("="*80)
    print("MASTER DATASET TRANSFORMATION GENERATOR")
    print("="*80)
    print("\nGenerates 3 transformations:")
    print("  1. Ciphered Letters (character substitution)")
    print("  2. Ciphered Words (word replacement)")
    print("  3. Scrambled (letter scrambling)")

    if args.dataset == 'all':
        print(f"\nProcessing ALL {len(DATASETS)} datasets...")
        for dataset_name in DATASETS.keys():
            process_dataset(dataset_name)
    else:
        print(f"\nProcessing dataset: {args.dataset}")
        process_dataset(args.dataset)

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)

if __name__ == "__main__":
    main()
