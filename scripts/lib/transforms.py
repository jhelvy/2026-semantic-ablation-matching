"""
v1 dataset transformations (library code for scripts/01_prepare_datasets.py):
1. Ciphered Letters (character substitution)
2. Ciphered Words (word replacement)
3. Scrambled (letter scrambling within words)

Importing this module seeds Python's global `random` with 42, as the original
prepare_datasets.py did. Ciphered Words builds its map by iterating over a
Python set, so its output depends on PYTHONHASHSEED; the committed CSVs in
data/*/data_test/ are the only copy of the map the paper used. Do not
regenerate them as a check.
"""

import pandas as pd
import random
import re
from pathlib import Path
from collections import defaultdict

from .datasets import DATASETS, get_text_columns, load_csv  # noqa: F401  (re-exported)

random.seed(42)

# ============================================================================
# TRANSFORMATION 1: CIPHERED LETTERS
# ============================================================================

def generate_ciphered_letters(df_left, df_right, dataset_config):
    """Generate ciphered letters transformation (character substitution)"""
    
    print("\n" + "="*80)
    print("TRANSFORMATION 1: CIPHERED LETTERS")
    print("="*80)
    
    # Create consistent character cipher
    letters = 'abcdefghijklmnopqrstuvwxyz'
    shuffled = list(letters)
    random.shuffle(shuffled)
    cipher = str.maketrans(letters + letters.upper(), 
                          ''.join(shuffled) + ''.join(shuffled).upper())
    
    print("\nCipher mapping (first 10):")
    for i in range(10):
        print(f"  {letters[i]} → {shuffled[i]}")
    
    def apply_cipher(df):
        df_new = df.copy()
        text_cols = get_text_columns(df)
        
        for col in text_cols:
            if col in df_new.columns:
                df_new[col] = df_new[col].apply(
                    lambda x: x.translate(cipher) if pd.notna(x) else x
                )
        
        return df_new
    
    print("\nApplying cipher...")
    df_left_new = apply_cipher(df_left)
    df_right_new = apply_cipher(df_right)
    
    # Verify
    print("\nVerification:")
    verify_transformation(df_left, df_left_new, "Left (ciphered letters)")
    verify_transformation(df_right, df_right_new, "Right (ciphered letters)")
    
    return df_left_new, df_right_new

# ============================================================================
# TRANSFORMATION 2: CIPHERED WORDS
# ============================================================================

def generate_ciphered_words(df_left, df_right, dataset_config):
    """Generate ciphered words transformation (word replacement)"""
    
    print("\n" + "="*80)
    print("TRANSFORMATION 2: CIPHERED WORDS")
    print("="*80)
    
    # Load dictionary
    print("\nLoading English dictionary...")
    dict_by_length = load_english_dictionary()
    
    # Extract unique words
    print("\nExtracting unique words...")
    words_by_length = extract_unique_words(df_left, df_right)
    
    # Create mappings
    print("\nCreating word mappings...")
    mapping = create_word_mappings(words_by_length, dict_by_length)
    
    # Apply cipher
    print("\nApplying word cipher...")
    df_left_new = apply_word_cipher(df_left, mapping)
    df_right_new = apply_word_cipher(df_right, mapping)
    
    # Verify
    print("\nVerification:")
    verify_transformation(df_left, df_left_new, "Left (ciphered words)")
    verify_transformation(df_right, df_right_new, "Right (ciphered words)")
    
    return df_left_new, df_right_new

def load_english_dictionary():
    """Load and filter English dictionary"""
    
    dict_path = '/usr/share/dict/words'
    
    if not Path(dict_path).exists():
        print("System dictionary not found, using generated words")
        return {}
    
    by_length = defaultdict(list)
    
    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.strip().lower()
            
            if not word or len(word) < 2 or len(word) > 15:
                continue
            if not word.isalpha():
                continue
            if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', word):
                continue
            
            length = len(word)
            if len(by_length[length]) < 1500:
                by_length[length].append(word)
    
    total = sum(len(words) for words in by_length.values())
    print(f"   Loaded {total:,} words")
    
    return by_length

def extract_unique_words(df_left, df_right):
    """Extract unique words from both datasets"""
    
    text_parts = []
    for df in [df_left, df_right]:
        text_cols = get_text_columns(df)
        for col in text_cols:
            if col in df.columns:
                text_parts.extend(df[col].fillna('').astype(str).tolist())
    
    all_text = ' '.join(text_parts).lower()
    words = set(re.findall(r'\w+', all_text))
    words = {w for w in words if not w.isdigit() and len(w) >= 1}
    
    by_length = defaultdict(list)
    for w in words:
        by_length[len(w)].append(w)
    
    print(f"   Found {len(words):,} unique words")
    return by_length

def create_word_mappings(words_by_length, dict_by_length):
    """Create word-to-word mappings"""
    
    mapping = {}
    consonants = 'bcdfghjklmnprstvwxyz'
    vowels = 'aeiou'
    
    for length in sorted(words_by_length.keys()):
        words_list = words_by_length[length]
        available = dict_by_length.get(length, [])
        
        needed = len(words_list)
        have = len(available)
        
        if have >= needed:
            random.shuffle(available)
            for i, word in enumerate(words_list):
                mapping[word] = available[i]
        else:
            random.shuffle(available)
            
            for i in range(have):
                mapping[words_list[i]] = available[i]
            
            for i in range(have, needed):
                gen_word = ''
                for j in range(length):
                    gen_word += random.choice(consonants if j % 2 == 0 else vowels)
                mapping[words_list[i]] = gen_word
    
    return mapping

def apply_word_cipher(df, mapping):
    """Apply word cipher to dataframe"""
    
    df_new = df.copy()
    text_cols = get_text_columns(df)
    
    def cipher_text(text):
        if pd.isna(text):
            return text
        
        text_str = str(text).lower()
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text_str)
        
        result = []
        for token in tokens:
            if re.match(r'^\w+$', token):
                result.append(token if token.isdigit() else mapping.get(token, token))
            else:
                result.append(token)
        
        return ''.join(result)
    
    for col in text_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].apply(cipher_text)
    
    return df_new

# ============================================================================
# TRANSFORMATION 3: SCRAMBLED
# ============================================================================

def generate_scrambled(df_left, df_right, dataset_config):
    """Generate scrambled transformation (letter scrambling) - CONSISTENT"""
    
    print("\n" + "="*80)
    print("TRANSFORMATION 3: SCRAMBLED (CONSISTENT)")
    print("="*80)
    
    # Create consistent word scrambling mapping
    print("\nCreating consistent scrambling mapping...")
    
    # Extract all unique words from both datasets
    all_words = set()
    text_cols_left = get_text_columns(df_left)
    text_cols_right = get_text_columns(df_right)
    
    for df, cols in [(df_left, text_cols_left), (df_right, text_cols_right)]:
        for col in cols:
            if col in df.columns:
                text = ' '.join(df[col].fillna('').astype(str))
                words = re.findall(r'\w+', text.lower())
                all_words.update([w for w in words if not w.isdigit()])
    
    print(f"   Found {len(all_words):,} unique words")
    
    # Create CONSISTENT scrambling mapping
    scramble_map = {}
    
    for word in all_words:
        if len(word) <= 3:
            # Don't scramble short words
            scramble_map[word] = word
        else:
            # Scramble consistently using word as seed
            middle = list(word[1:-1])
            # Use word itself as seed for consistency
            random.Random(word).shuffle(middle)
            scramble_map[word] = word[0] + ''.join(middle) + word[-1]
    
    print(f"   Created {len(scramble_map):,} consistent scrambling mappings")
    
    # Apply scrambling using the consistent mapping
    def apply_scramble(df, cols):
        df_new = df.copy()
        
        for col in cols:
            if col in df_new.columns:
                def scramble_text(text):
                    if pd.isna(text):
                        return text
                    
                    tokens = re.findall(r'\w+|[^\w\s]|\s+', str(text).lower())
                    result = []
                    
                    for token in tokens:
                        if re.match(r'^\w+$', token):
                            if token.isdigit():
                                result.append(token)
                            else:
                                # Use consistent mapping
                                result.append(scramble_map.get(token, token))
                        else:
                            result.append(token)
                    
                    return ''.join(result)
                
                df_new[col] = df_new[col].apply(scramble_text)
        
        return df_new
    
    print("\nExample consistent scrambling (first 10):")
    sample_words = sorted(list(scramble_map.keys()))[:10]
    for word in sample_words:
        if len(word) > 3:
            print(f"   {word:15} → {scramble_map[word]}")
    
    print("\nApplying consistent scrambling...")
    df_left_new = apply_scramble(df_left, text_cols_left)
    df_right_new = apply_scramble(df_right, text_cols_right)
    
    # Verify
    print("\nVerification:")
    verify_transformation(df_left, df_left_new, "Left (scrambled)")
    verify_transformation(df_right, df_right_new, "Right (scrambled)")
    
    return df_left_new, df_right_new

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_transformation(df_orig, df_new, name):
    """Verify exact length preservation"""
    
    text_cols = get_text_columns(df_orig)
    
    orig_len = sum(df_orig[col].fillna('').str.len() for col in text_cols if col in df_orig.columns)
    new_len = sum(df_new[col].fillna('').str.len() for col in text_cols if col in df_new.columns)
    
    matches = (orig_len == new_len).sum()
    total = len(df_orig)
    pct = 100 * matches / total
    
    status = "PERFECT!" if matches == total else f"{total-matches} errors"
    print(f"  {name:30} {matches}/{total} ({pct:.1f}%) {status}")
    
    return matches == total

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_dataset(dataset_name):
    """Process a single dataset - generate all transformations"""
    
    print("\n" + "="*80)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*80)
    
    config = DATASETS[dataset_name]
    base_dir = config['dir']
    
    # Load original data
    print("\nLoading original data...")
    df_left = load_csv(base_dir / config['files']['left'])
    df_right = load_csv(base_dir / config['files']['right'])
    print(f"   Left:  {len(df_left):,} records")
    print(f"   Right: {len(df_right):,} records")
    
    # Create output directory
    output_dir = base_dir / 'data_test'
    output_dir.mkdir(exist_ok=True)
    
    # Generate transformations
    transformations = {}
    
    # 1. Ciphered Letters
    left_cl, right_cl = generate_ciphered_letters(df_left, df_right, config)
    transformations['ciphered_letters'] = (left_cl, right_cl)
    
    # 2. Ciphered Words
    left_cw, right_cw = generate_ciphered_words(df_left, df_right, config)
    transformations['ciphered_words'] = (left_cw, right_cw)
    
    # 3. Scrambled
    left_sc, right_sc = generate_scrambled(df_left, df_right, config)
    transformations['scrambled'] = (left_sc, right_sc)
    
    # Save all transformations
    print("\n" + "="*80)
    print("SAVING DATASETS")
    print("="*80)
    
    for trans_name, (left_df, right_df) in transformations.items():
        left_file = output_dir / f"{config['left_name']}_{trans_name}.csv"
        right_file = output_dir / f"{config['right_name']}_{trans_name}.csv"
        
        left_df.to_csv(left_file, index=False)
        right_df.to_csv(right_file, index=False)
        
        print(f"{trans_name}:")
        print(f"{left_file}")
        print(f"{right_file}")
    
    print("\n" + "="*80)
    print(f"All transformations generated for {dataset_name}")
    print("="*80)
