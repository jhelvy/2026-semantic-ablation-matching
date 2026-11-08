"""
2x2 FACTORIAL ABLATION: descriptive words x identifiers
=======================================================

Separates two kinds of evidence a matcher can use:
  * descriptive words (brand, product type, attributes) - carry meaning;
  * identifiers (model numbers / SKUs such as "pslx350h", "kdl-46v3000") -
    arbitrary codes whose only use is exact string identity.

Cells (Original is the D0/I0 cell and is not regenerated):

                         identifiers intact     identifiers ciphered
  descriptions intact    original               ids_ciphered
  descriptions ciphered  desc_ciphered          all_ciphered

Reading the cells:
  desc_ciphered - meaning removed, exact identifiers still in plain view.
                  A matcher that can fall back on identifiers should hold up.
  ids_ciphered  - meaning intact, identifiers replaced by unfamiliar codes
                  (still identical across catalogs). Tests reliance on
                  recognising known product codes vs. matching strings.
  all_ciphered  - no meaning anywhere. Unlike v1 ciphered_words, words map to
                  pronounceable NONCE words that are not English words, so
                  no new semantic associations are introduced (R2 D1).

Token classes (assigned once per unique \\w+ piece, globally across both
catalogs, so every mapping is consistent between catalogs):
  * identifier piece - a piece containing both a letter and a digit ("w55",
    "pslx350h"), or a letters-only piece glued to one in the same whitespace
    token that is short (<= 4 letters) and not an English word or unit ("dsc"
    in "dsc-w55"); excluding quantity-like pieces ("4gb", "1080p", "30pk",
    "1024x768") and pieces found in more than --max-id-df records (e.g. "mp3");
  * kept piece - pure digits and single characters: unchanged in every cell
    (as in the v1 transforms; numbers are quantities, 1-char pieces carry
    almost no meaning and have no collision-free substitutes);
  * descriptive piece - everything else.

Substitutions:
  * descriptive piece -> letters replaced run by run with a pronounceable
    pseudoword of the same length; digits inside the piece are kept ("4gb" ->
    "4ku"). Candidates are rejected if they are an English word (length >= 3,
    /usr/share/dict/words when present), any original piece, or already used.
  * identifier piece -> every letter replaced by a random letter and every
    digit by a random digit ("pslx350h" -> "qbzt817m"); same rejection rules.
  Both maps are injective and never produce an original piece, so token
  identity across catalogs is exactly preserved and no false identities are
  created. TF-IDF scores are therefore invariant across all cells up to
  floating-point summation order (~1e-16), which can flip argmax between
  exactly tied candidates and move F1 in the third decimal. --verify checks
  the score matrices and reports v1 tfidf F1 per cell.

Text is lowercased (as in v1 ciphered_words/scrambled; all matchers lowercase
anyway). Output: data/<dataset>/data_test/<side>_<cell>[_s<seed>].csv plus a
manifest JSON with class counts, examples, identifier coverage of the
ground-truth pairs, and the full mappings.

Library code for scripts/02_prepare_factorial.py (see that file for usage).
"""

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .datasets import DATASETS, get_text_columns, load_csv

CELLS = {  # cell -> (cipher descriptive pieces?, cipher identifier pieces?)
    "desc_ciphered": (True, False),
    "ids_ciphered": (False, True),
    "all_ciphered": (True, True),
}

PIECE_RE = re.compile(r"\w+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+")          # same tokenizer as v1 transforms
UNITS = {"gb", "mb", "tb", "kb", "ghz", "mhz", "hz", "mp", "mm", "cm", "in", "ft", "oz", "lb", "lbs",
         "w", "v", "mah", "ma", "ah", "k", "p", "i", "x", "pk", "pc", "pcs", "ct", "ch", "th", "st", "nd",
         "rd", "d", "bit", "mbps", "gbps", "kbps", "rpm", "rms", "db", "ms", "hr", "hrs", "yr", "yrs", "min",
         "ml", "l", "g", "kg", "fps", "dpi", "ppm", "btu", "cu", "qt", "gal", "sec", "mins", "watt", "watts",
         "volt", "inch", "lcd", "led", "hd", "usb", "cd", "dvd", "mpeg", "ata"}
UNIT_RE = re.compile(r"^\d+(?:\.\d+)?(?:" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")$"
                     r"|^\d+x\d+$")                  # quantities ("4gb", "1080p") and dimensions ("1024x768")
MAX_GLUED_LEN = 4      # letters-only pieces glued to a code ("dsc" in "dsc-w55") must be this short
CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"
MANIFEST_MAP_LIMIT = None    # None = store full mappings in the manifest


# =============================================================================
# CLASSIFICATION
# =============================================================================

def _is_mixed(p):
    return any(c.isalpha() for c in p) and any(c.isdigit() for c in p)


def classify_pieces(df_left, df_right, max_id_df, english):
    """Return (vocab, id_pieces, kept_pieces, desc_pieces, record_df).

    Identifier piece = (a) a piece containing both a letter and a digit
    ("w55", "pslx350h", "cs2"), or (b) a letters-only piece that sits in the same
    whitespace token as an (a)-piece and is neither an English word nor a unit
    and has at most MAX_GLUED_LEN letters ("dsc" in "dsc-w55", "kdl" in
    "kdl-46v3000"; not "pixel" in "12-pixel" or "widescreen" in "16x9-widescreen").
    Either way it must not look like a quantity ("4gb", "1080p", "1024x768") and
    must occur in at most max_id_df records."""
    vocab = set()
    code_candidates = set()
    record_df = Counter()
    for df in (df_left, df_right):
        cols = get_text_columns(df)
        for _, row in df[cols].iterrows():
            rec_pieces = set()
            for v in row.dropna().astype(str):
                v = v.lower()
                for ws_tok in v.split():
                    pieces = PIECE_RE.findall(ws_tok)
                    rec_pieces.update(pieces)
                    mixed = [p for p in pieces if _is_mixed(p) and not UNIT_RE.match(p)]
                    if mixed:
                        code_candidates.update(mixed)
                        code_candidates.update(p for p in pieces if p.isalpha() and len(p) <= MAX_GLUED_LEN
                                               and p not in english and p not in UNITS)
            vocab |= rec_pieces
            record_df.update(rec_pieces)

    def is_kept(p):
        return p.isdigit() or len(p) == 1

    id_pieces = {p for p in code_candidates if not is_kept(p) and record_df[p] <= max_id_df}
    kept = {p for p in vocab if is_kept(p)}
    desc = vocab - id_pieces - kept
    return vocab, id_pieces, kept, desc, record_df


# =============================================================================
# SUBSTITUTION MAPS
# =============================================================================

def _english_words():
    p = Path("/usr/share/dict/words")
    if not p.exists():
        print("   /usr/share/dict/words not found: nonce words are not screened against English")
        return set()
    with open(p, encoding="utf-8", errors="ignore") as f:
        return {w.strip().lower() for w in f if w.strip().isalpha()}


def _pseudo_run(n, rng):
    if n <= 3:                                     # short runs: CV pattern space too small
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
    start_vowel = rng.random() < 0.3
    return "".join(rng.choice(VOWELS if (i % 2 == 0) == start_vowel else CONSONANTS)
                   for i in range(n))


def _nonce(piece, rng, extra=0):
    """Replace each letter run with a pseudoword run; keep digits."""
    out, runs = [], re.findall(r"[^\W\d_]+|\d+|_+", piece)
    for i, run in enumerate(runs):
        if run[0].isdigit() or run[0] == "_":
            out.append(run)
        else:
            out.append(_pseudo_run(len(run) + (extra if i == len(runs) - 1 else 0), rng))
    return "".join(out)


def _code(piece, rng, extra=0):
    """Random letter for each letter, random digit for each digit."""
    out = []
    for c in piece:
        if c.isdigit():
            out.append(rng.choice(string.digits))
        elif c.isalpha():
            out.append(rng.choice(string.ascii_lowercase))
        else:
            out.append(c)
    out.extend(rng.choice(string.ascii_lowercase) for _ in range(extra))
    return "".join(out)


def build_map(pieces, gen, rng, forbidden, english, used, tries=200):
    """Injective map; outputs avoid original pieces, English words, and each other."""
    mapping, lengthened = {}, 0
    for p in sorted(pieces):                       # sorted: reproducible for a given seed
        extra = 0
        while True:
            for _ in range(tries):
                cand = gen(p, rng, extra)
                letters_only = cand.isalpha()
                if (cand not in forbidden and cand not in used
                        and not (letters_only and len(cand) >= 3 and cand in english)):
                    break
            else:
                extra += 1                         # space exhausted at this length
                continue
            break
        mapping[p] = cand
        used.add(cand)
        lengthened += extra > 0
    return mapping, lengthened


# =============================================================================
# APPLY
# =============================================================================

def apply_cell(df, desc_map, id_map, cipher_desc, cipher_ids):
    df_new = df.copy()

    def tr(text):
        if pd.isna(text):
            return text
        out = []
        for tok in TOKEN_RE.findall(str(text).lower()):
            if not tok[0].isspace() and PIECE_RE.fullmatch(tok):
                if tok in id_map:
                    out.append(id_map[tok] if cipher_ids else tok)
                elif tok in desc_map:
                    out.append(desc_map[tok] if cipher_desc else tok)
                else:
                    out.append(tok)                # kept piece (digits / single char)
            else:
                out.append(tok)
        return "".join(out)

    for col in get_text_columns(df):
        df_new[col] = df_new[col].apply(tr)
    return df_new


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def identifier_coverage(df_left, df_right, mapping_df, id_pieces, record_df):
    """Share of ground-truth pairs that share >=1 identifier piece, and >=1 piece
    found in no other record (decisive evidence for a pure string matcher)."""
    def pieces_by_id(df):
        cols = get_text_columns(df)
        out = {}
        for _, row in df.iterrows():
            ps = set()
            for v in row[cols].dropna().astype(str):
                ps.update(PIECE_RE.findall(v.lower()))
            out[str(row[df.columns[0]])] = ps & id_pieces
        return out

    L, R = pieces_by_id(df_left), pieces_by_id(df_right)
    lc, rc = mapping_df.columns[0], mapping_df.columns[1]
    n = shared = unique = 0
    for a, b in zip(mapping_df[lc].astype(str), mapping_df[rc].astype(str)):
        if a not in L or b not in R:
            continue
        n += 1
        common = L[a] & R[b]
        shared += bool(common)
        unique += any(record_df[p] == 2 for p in common)
    return dict(pairs=n, share_identifier=round(shared / n, 4) if n else None,
                share_unique_identifier=round(unique / n, 4) if n else None)


def verify_tfidf(dataset, cells):
    """Check TF-IDF cosine scores are identical (to 1e-9) across cells, and run
    the v1 tfidf method on each; F1 can differ only through tie-breaking."""
    from .methods import prepare_text_column, tfidf  # heavy import, only when asked
    from .datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def scores(cell):
        l, r, _ = load_dataset(dataset, cell)
        l, r = prepare_text_column(l), prepare_text_column(r)
        vec = TfidfVectorizer().fit(pd.concat([l["text"], r["text"]]))
        return cosine_similarity(vec.transform(l["text"]), vec.transform(r["text"]))

    base = scores("original")
    for cell in cells:
        diff = float(abs(scores(cell) - base).max())
        if diff > 1e-9:
            raise AssertionError(f"{dataset}/{cell}: TF-IDF scores differ from original by {diff:.2e}")

    def f1(df):
        has_gt = df["true_id_right"] != ""
        tp = ((df["predicted_match"] == 1) & (df["is_correct"] == 1)).sum()
        fp = ((df["predicted_match"] == 1) & (df["is_correct"] == 0)).sum()
        fn = (has_gt & (df["is_correct"] == 0)).sum()
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        return 2 * p * r / (p + r) if p + r else 0

    out = {}
    for cell in ["original"] + cells:
        out[cell] = round(float(f1(tfidf(*load_dataset(dataset, cell)))), 4)
    return out


# =============================================================================
# MAIN
# =============================================================================

def process(dataset, seed, max_id_df, verify):
    cfg = DATASETS[dataset]
    base = cfg["dir"]
    df_l = load_csv(base / cfg["files"]["left"])
    df_r = load_csv(base / cfg["files"]["right"])
    print(f"\n=== {dataset}: {len(df_l)} left, {len(df_r)} right records")

    english = _english_words()
    vocab, id_p, kept_p, desc_p, rdf = classify_pieces(df_l, df_r, max_id_df, english)
    print(f"   pieces: {len(vocab):,} total | identifier {len(id_p):,} | descriptive {len(desc_p):,} "
          f"| kept (digits/1-char) {len(kept_p):,}")

    rng = random.Random(f"factorial-{dataset}-{seed}")
    used = set()
    desc_map, d_long = build_map(desc_p, _nonce, rng, vocab, english, used)
    id_map, i_long = build_map(id_p, _code, rng, vocab, english, used)
    assert len(set(desc_map.values()) | set(id_map.values())) == len(desc_map) + len(id_map)
    assert not (set(desc_map.values()) | set(id_map.values())) & vocab
    print(f"   lengthened (no same-length substitute left): descriptive {d_long}, identifier {i_long}")

    out_dir = base / "data_test"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if seed == 42 else f"_s{seed}"
    for cell, (cd, ci) in CELLS.items():
        for side in ("left", "right"):
            df = df_l if side == "left" else df_r
            path = out_dir / f"{cfg[side + '_name']}_{cell}{suffix}.csv"
            apply_cell(df, desc_map, id_map, cd, ci).to_csv(path, index=False)
        print(f"   wrote {cell}{suffix}")

    # ground-truth identifier coverage
    mp = pd.read_csv(base / cfg["mapping_file"])
    coverage = identifier_coverage(df_l, df_r, mp, id_p, rdf)
    print(f"   ground-truth pairs sharing an identifier piece: {coverage['share_identifier']}, "
          f"sharing a piece unique to the pair: {coverage['share_unique_identifier']}")

    sample = sorted(id_p, key=lambda p: -rdf[p])[:15]
    manifest = dict(
        dataset=dataset, seed=seed, max_id_df=max_id_df, cells={c: dict(cipher_descriptive=d, cipher_identifiers=i)
                                                                for c, (d, i) in CELLS.items()},
        counts=dict(vocab=len(vocab), identifier=len(id_p), descriptive=len(desc_p), kept=len(kept_p),
                    lengthened_descriptive=d_long, lengthened_identifier=i_long),
        identifier_coverage=coverage,
        examples=dict(
            most_frequent_identifier_pieces=sample,
            identifier=[[p, id_map[p]] for p in sorted(id_p)[:15]],
            descriptive=[[p, desc_map[p]] for p in sorted(desc_p, key=lambda p: -rdf[p])[:15]],
        ),
        desc_map=desc_map, id_map=id_map,
    )
    if verify:
        manifest["tfidf_f1"] = verify_tfidf(dataset, [c + suffix for c in CELLS])
        print(f"   TF-IDF F1 by cell: {manifest['tfidf_f1']}")
    (out_dir / f"factorial_manifest{suffix}.json").write_text(json.dumps(manifest, indent=1))
    print(f"   manifest: {out_dir / f'factorial_manifest{suffix}.json'}")
