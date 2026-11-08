"""
Dataset configuration and loading, shared by every script.

Merges the two DATASETS dicts that used to live in prepare_datasets.py
('files' = source CSVs for the transforms) and run_experiments.py
('mapping_file' = ground truth). File names are kept exactly as before; some
differ in case from the files on disk (e.g. amazon.csv vs Amazon.csv), which
works on case-insensitive file systems (macOS default) only.
"""

import pandas as pd

from .paths import DATA

DATASETS = {
    'abt-buy': {
        'dir': DATA / 'abt-buy',
        'files': {
            'left': 'data_original/abt.csv',
            'right': 'data_original/buy.csv'
        },
        'left_name': 'abt',
        'right_name': 'buy',
        'mapping_file': 'data_original/abt_buy_perfect_mapping.csv'
    },
    'amazon-google': {
        'dir': DATA / 'amazon-google',
        'files': {
            'left': 'data_original/amazon.csv',
            'right': 'data_original/google.csv'
        },
        'left_name': 'amazon',
        'right_name': 'google',
        'mapping_file': 'data_original/Amazon_GoogleProducts_perfectMapping.csv'
    },
    # The two DBLP datasets are configured but their data is not in this repo.
    'dblp-acm': {
        'dir': DATA / 'dblp-acm',
        'files': {
            'left': 'data_original/dblp.csv',
            'right': 'data_original/acm.csv'
        },
        'left_name': 'dblp',
        'right_name': 'acm',
        'mapping_file': 'data_original/DBLP-ACM_perfectMapping.csv'
    },
    'dblp-scholar': {
        'dir': DATA / 'dblp-scholar',
        'files': {
            'left': 'data_original/dblp.csv',
            'right': 'data_original/scholar.csv'
        },
        # run_experiments used 'dblp1', prepare_datasets used 'dblp'; 'dblp1' kept
        # so the runner finds the files the transforms write.
        'left_name': 'dblp1',
        'right_name': 'scholar',
        'mapping_file': 'data_original/DBLP-Scholar_perfectMapping.csv'
    }
}

TRANSFORMATIONS = ['original', 'ciphered_letters', 'ciphered_words', 'scrambled']
# 2x2 descriptive-words x identifiers cells (scripts/02_prepare_factorial.py); 'original' is the 4th cell.
# Not part of "all": select with --transformations factorial (or list cells explicitly).
FACTORIAL_CELLS = ['desc_ciphered', 'ids_ciphered', 'all_ciphered']
TRANSFORMATION_GROUPS = {
    'all': TRANSFORMATIONS,
    'factorial': ['original'] + FACTORIAL_CELLS,
}


def load_csv(filepath):
    """Load CSV with encoding detection"""
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except:
            continue
    raise ValueError(f"Cannot load {filepath}")


def get_text_columns(df):
    """Get text columns for transformation"""
    common_cols = ['name', 'title', 'description', 'manufacturer', 'authors', 'venue', 'year']
    return [col for col in common_cols if col in df.columns]


def load_dataset(dataset_name, transformation):
    """Load dataset files"""
    config = DATASETS[dataset_name]
    base_dir = config['dir']

    if transformation == 'original':
        left_file = base_dir / f"data_original/{config['left_name']}.csv"
        right_file = base_dir / f"data_original/{config['right_name']}.csv"
    else:
        left_file = base_dir / f"data_test/{config['left_name']}_{transformation}.csv"
        right_file = base_dir / f"data_test/{config['right_name']}_{transformation}.csv"

    # Load with encoding detection
    for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df_left = pd.read_csv(left_file, encoding=enc)
            df_right = pd.read_csv(right_file, encoding=enc)
            break
        except:
            continue

    # Load ground truth
    mapping_file = base_dir / config['mapping_file']
    df_mapping = pd.read_csv(mapping_file)

    # FILTER TO ONLY MATCHED RECORDS (Professor's requirement)
    print(f"  Original: {len(df_left)} left, {len(df_right)} right")

    # Get column names
    left_col = df_left.columns[0]
    right_col = df_right.columns[0]
    mapping_left_col = df_mapping.columns[0]
    mapping_right_col = df_mapping.columns[1]

    # Get unique IDs from mapping
    matched_left_ids = set(df_mapping[mapping_left_col].astype(str).unique())
    matched_right_ids = set(df_mapping[mapping_right_col].astype(str).unique())

    # Filter datasets to only matched records
    df_left = df_left[df_left[left_col].astype(str).isin(matched_left_ids)].copy()
    df_right = df_right[df_right[right_col].astype(str).isin(matched_right_ids)].copy()

    print(f"  Filtered: {len(df_left)} left, {len(df_right)} right (only matched records)")

    return df_left, df_right, df_mapping
