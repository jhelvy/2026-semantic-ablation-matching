# Matching Without Meaning: LLM Entity Matchers Rely on Non-Essential Semantic Priors

**Systematic comparison of 8 entity matching methods across controlled semantic ablations**

---

## Research Question

**Does an LLM entity matcher rely on the features that actually distinguish entities, or on semantic priors the task does not require?**

We use *controlled semantic ablation*: structure-preserving transformations that remove meaning while keeping the discriminative tokens (and their cross-catalog identity) fully intact. If a method depends only on token distinctiveness, it should be unaffected; if it depends on meaning, it should collapse.

1. **Original** - Baseline
2. **Ciphered Letters** - Consistent character substitution applied to both catalogs (e.g., 'a'->'x'). Preserves token identity exactly.
3. **Ciphered Words** - Consistent vocabulary replacement (e.g., 'Sony'->'Goat' in both catalogs).
4. **Scrambled** - Deterministic per-word letter scrambling (first/last letter kept), identical across catalogs.

---

## Key Findings

**The LLM's clean-text advantage depends on semantic content the task does not require.** Once token-based methods are properly thresholded, the LLM's edge on clean data is modest (Abt-Buy) or negligible (Amazon-Google), and under ablation the LLM collapses while token methods stay flat. A method's fragility tracks how much it depends on pre-trained meaning, not its architecture or scale -- the smallest neural model (SentenceTransformer) is the *most* fragile, and the far larger LLM is no more robust.

### F1 by transformation (all 8 methods)

**Abt-Buy** (rich title + description):
| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|---|---|---|---|---|
| LLM (GPT-4o-mini) | **0.928** | **0.779** | 0.511 | 0.603 |
| OpenAI Embeddings | 0.799 | 0.409 | 0.422 | 0.508 |
| TF-IDF | 0.761 | 0.747 | **0.751** | 0.746 |
| Soft TF-IDF | 0.742 | 0.731 | 0.743 | **0.763** |
| SentenceTransformer | 0.628 | 0.164 | 0.376 | 0.362 |
| Monge-Elkan | 0.381 | 0.381 | 0.289 | 0.382 |
| Levenshtein | 0.164 | 0.164 | 0.120 | 0.173 |
| Jaro-Winkler | 0.085 | 0.085 | 0.052 | 0.080 |

**Amazon-Google** (short title only):
| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|---|---|---|---|---|
| LLM (GPT-4o-mini) | **0.822** | 0.585 | 0.576 | 0.556 |
| TF-IDF | 0.808 | **0.808** | **0.807** | **0.808** |
| OpenAI Embeddings | 0.802 | 0.639 | 0.701 | 0.725 |
| SentenceTransformer | 0.755 | 0.619 | 0.683 | 0.676 |
| Monge-Elkan | 0.740 | 0.740 | 0.700 | 0.735 |
| Soft TF-IDF | 0.650 | 0.650 | 0.678 | 0.683 |
| Levenshtein | 0.563 | 0.563 | 0.535 | 0.569 |
| Jaro-Winkler | 0.488 | 0.488 | 0.483 | 0.489 |

### Precision / Recall / F1 on original (clean) text

**Abt-Buy:**
| Method | Precision | Recall | F1 |
|---|---|---|---|
| LLM (GPT-4o-mini) | 0.967 | 0.892 | **0.928** |
| OpenAI Embeddings | 0.799 | 0.798 | 0.799 |
| TF-IDF | 0.762 | 0.759 | 0.761 |
| Soft TF-IDF | 0.742 | 0.742 | 0.742 |
| SentenceTransformer | 0.629 | 0.628 | 0.628 |
| Monge-Elkan | 0.525 | 0.299 | 0.381 |
| Levenshtein | 0.171 | 0.156 | 0.164 |
| Jaro-Winkler | 0.266 | 0.051 | 0.085 |

**Amazon-Google:**
| Method | Precision | Recall | F1 |
|---|---|---|---|
| LLM (GPT-4o-mini) | 0.876 | 0.774 | **0.822** |
| TF-IDF | 0.808 | 0.808 | 0.808 |
| OpenAI Embeddings | 0.803 | 0.801 | 0.802 |
| SentenceTransformer | 0.759 | 0.751 | 0.755 |
| Monge-Elkan | 0.793 | 0.694 | 0.740 |
| Soft TF-IDF | 0.650 | 0.650 | 0.650 |
| Levenshtein | 0.684 | 0.478 | 0.563 |
| Jaro-Winkler | 0.564 | 0.430 | 0.488 |

### Failure is in reasoning, not retrieval

TF-IDF blocking keeps the true match in the candidate set for ~99% of records *even under corruption*, and a purely lexical scorer still solves those cases -- yet the LLM cannot.

- **Blocking recall** (top-50 candidates): Abt-Buy **99.7%** (1,078/1,081), Amazon-Google **99.0%** (1,102/1,113). Stays high under every ablation (99.5-99.7% / 98.5-99.0%). `k = 50` chosen by sweep (Abt-Buy: k=25 -> 97.2%, k=50 -> 99.7%, k=100 -> 99.8%).
- **LLM recall under Ciphered Words** drops to **0.418** (Abt-Buy) and **0.456** (Amazon-Google) despite blocking recall staying near 0.99 -- the candidates are present; the LLM fails to select them once words are no longer meaningful.

### Robustness holds across models (clean-text F1)

Larger/stronger models score slightly higher on clean text but are *no more robust* under ablation.

| Model | Abt-Buy | Amazon-Google |
|---|---|---|
| GPT-4o-mini (primary) | 0.928 | 0.822 |
| GPT-4o | 0.955 | 0.904 |
| Claude Haiku 4.5 | 0.961 | 0.895 |
| Claude Sonnet 4.6 | 0.954 | -- (run abandoned, billing) |

### Prompt sensitivity (token-fallback prompt)

A prompt that explicitly instructs the model to match on identifier tokens gives mixed results and never closes the gap to TF-IDF:
- **GPT-4o-mini** improves on short titles (Amazon-Google Scrambled 0.555 -> 0.707) and helps unevenly on Abt-Buy (Ciphered Words 0.510 -> 0.557, Scrambled 0.601 -> 0.632) but *hurts* under Ciphered Letters (0.779 -> 0.703).
- **Claude Haiku** collapses to near-zero F1 on Ciphered Words/Scrambled under the token-fallback prompt.
- Even the best case (GPT-4o-mini, 0.633-0.707) stays well below TF-IDF (0.807-0.808).

### Cost & latency (per dataset)

| Model | Cost | Wall time |
|---|---|---|
| GPT-4o-mini | <$1 (~$0.50 Abt-Buy, ~$1.00 Amazon-Google) | 30-80 min |
| Claude Haiku 4.5 | $3-9 | hours |
| GPT-4o | $7-22 | 1-6 h |
| Claude Sonnet 4.6 | $10-13 | 1-6 h |
| All non-LLM methods | $0 | seconds |

The cost-benefit case for LLM matching is narrow: it is justified only on clean, description-rich text where the modest F1 gain (0.928 vs 0.761) outweighs orders-of-magnitude differences in cost and latency.

> **Note on results:** earlier versions of this repo reported lower TF-IDF/Soft TF-IDF F1 because (a) the threshold sweep started at 0.50, which is above the optimal operating point for cosine/edit-distance methods, and (b) the "Soft TF-IDF" method lacked IDF weighting. Both are fixed (`scripts/lib/methods.py`); the threshold sweep now runs the full 0.05-0.95 grid and Soft TF-IDF is the real IDF-weighted, Jaro-Winkler version.

---

## Methods Evaluated

### Character-Based
- **Jaro-Winkler**: Edit distance with prefix weighting
- **Levenshtein**: Minimum edit distance
- **Monge-Elkan**: Token-level Jaro-Winkler averaging

### Token-Based
- **TF-IDF**: Cosine similarity on term frequency vectors
- **Soft TF-IDF**: IDF-weighted TF-IDF with Jaro-Winkler fuzzy token matching (threshold >= 0.9)

### Neural Embeddings
- **SentenceTransformer**: all-MiniLM-L6-v2
- **OpenAI Embeddings**: text-embedding-3-small

### Large Language Model
- **GPT-4o-mini** (primary): TF-IDF blocking (top-50 candidates), temperature = 0
- Additional models tested for robustness: GPT-4o, Claude Sonnet 4.6, Claude Haiku 4.5

---

## Datasets

Both datasets are filtered to records with at least one ground-truth match, following standard benchmark practice.

- **Abt-Buy** (1,081 matched pairs): Title + full product description. Rich, heterogeneous text.
- **Amazon-Google** (1,113 matched pairs): Title only. Short, keyword-dense text.

---

## Evaluation Protocol

- **Threshold optimization**: Each method's decision threshold is swept over tau in {0.05, 0.10, ..., 0.95}; the threshold maximizing F1 is reported.
- **Cross-validation**: 5-fold CV confirms threshold selection does not overfit -- threshold is chosen on 4 training folds and evaluated on the held-out fold (seed = 42). Results are in `results/cv_results.csv` with mean and std columns.
- **LLM blocking**: TF-IDF cosine similarity retrieves top-50 candidates per record before LLM classification. Blocking recall is ~99% on both datasets under all conditions.

---

## Reproducing the Results

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy scikit-learn tqdm
pip install jellyfish sentence-transformers openai anthropic

# R (for figures)
# Requires: ggplot2, cowplot, readr, dplyr, showtext (+ Roboto Condensed font)
```

API keys needed for LLM/embedding methods:

```bash
export OPENAI_API_KEY=...      # GPT-4o-mini, GPT-4o, OpenAI embeddings
export ANTHROPIC_API_KEY=...   # Claude Sonnet, Claude Haiku
```

### Step 1: Prepare transformed datasets

```bash
python scripts/01_prepare_datasets.py --dataset all
```

Creates `data/{dataset}/data_test/` with ciphered_letters, ciphered_words, and scrambled variants.

### Step 1b: 2x2 factorial transforms (descriptive words x identifiers)

```bash
python scripts/02_prepare_factorial.py --dataset all --verify
python tests/test_factorial.py        # offline unit tests
```

This step adds three cells next to Original:

|                      | identifiers intact | identifiers ciphered |
|---|---|---|
| **descriptions intact**   | `original`         | `ids_ciphered`       |
| **descriptions ciphered** | `desc_ciphered`    | `all_ciphered`       |

- **Identifiers** are model numbers and SKUs: pieces that contain both a letter and a digit (`w55`, `pslx350h`), plus short non-English prefixes attached to them (`dsc` in `dsc-w55`). Quantities (`4gb`, `1080p`, `1024x768`) are not identifiers. Neither are pieces that appear in more than 20 records.
- **Descriptive words** are everything else, including brand names.
- Digits and single characters are never changed.
- **Ciphers.**
  - Descriptive words become pronounceable nonce words of the same length that are not English words. Unlike v1 Ciphered Words, this brings in no new meanings.
  - Identifiers become random codes of the same shape: a letter for each letter, a digit for each digit.
- Both maps are global and injective, and they never produce an original token. Exact token identity across the two catalogs is therefore unchanged. `--verify` checks that the TF-IDF score matrices are identical in every cell (to 1e-9).
- F1 can still move in the third decimal, because floating-point summation order breaks exact ties between candidates.
- Each dataset gets a `factorial_manifest.json` with class counts, the full maps, and identifier coverage.
- Identifier coverage at seed 42:
  - On Abt-Buy, 50% of true pairs share an identifier piece, and 43% share one that occurs in no other record.
  - On Amazon-Google, the figures are 4% and 0.8%. The identifier factor mainly matters on Abt-Buy.
- Run the cells with `--transformations factorial`, which runs Original plus the three new cells. They are not part of `all`.

### Step 2: Run experiments

```bash
# All methods on all transformations and datasets
python scripts/04_run_experiments.py --dataset all --methods all --transformations all

# Or run selectively
python scripts/04_run_experiments.py --dataset abt-buy --methods llm,tfidf --transformations original
```

Results are saved as parquet files in `results/{dataset}/{method}/`. The superseded v1 LLM results are in `results/v1_llm/{dataset}/{method}/`.

### Step 3: Cross-validate thresholds

```bash
python scripts/05_cv_threshold.py
```

Produces `results/cv_results.csv` (F1 mean/std, precision, recall for all 128 configurations).

### Step 4: Generate summary table

```bash
python scripts/06_summarize_results.py
```

Produces `results/summary.csv` with TP/FP/FN counts and (for LLM methods) cost and wall time.

### Step 5: Generate figures

```bash
Rscript scripts/08_figures.R
```

Produces PNG figures in `figs/`. Requires the Roboto Condensed font and the R packages tidyverse, cowplot, ggh4x, and ragg.

---

## Harness v2 (LLM rerun, set up 2026-11)

The v1 LLM results (`llm`, `llm_haiku*`, `llm_claude*`, `llm_gpt4o*`) parsed free-text answers by line prefix and stored API errors and unparseable answers as blank "no match" rows. An audit on 2026-11-06 found these blank rows throughout the ablation conditions (for example, every Haiku token-fallback Scrambled row failed at $0 cost). Harness v2 replaces that pipeline for all new LLM runs. The v1 code and results are left untouched for reference.

What changed:

- **One code path for every model** (`scripts/lib/llm_clients.py`, `scripts/lib/llm_runner.py`) with structured JSON output (`match`, `confidence`, `rationale`) instead of line parsing.
- **Row-level `status`**: `ok`, `no_match`, `parse_fail`, `truncated`, `refusal`, `api_error`. Raw responses, stop reasons, token counts, cost, served model and request ids are saved for every row.
- **Fails loudly**: a bad key, missing credit, or unknown model aborts at once. Five consecutive API errors, or more than 1% of calls, also abort the run. A run with unresolved API errors is never written to `results/`. Partial progress goes to `checkpoints/`, and rerunning the same command resumes it.
- **No sampling parameters.** Current Claude models reject `temperature`, so run-to-run variance is measured with `--repeat`.
- **Instance-level diagnostics** on every row: the rank of the true match among the blocked candidates, the candidate ids, and the prompt size.

Models (`llm_clients.MODELS`; prices checked 2026-11-06, re-check before budgeting):

| Method | Model | Run | Notes |
|---|---|---|---|
| `llm_haiku55` | `claude-haiku-5-5` | full grid | reported |
| `llm_sonnet55` | `claude-sonnet-5-5` | fixed 30% subsample | reported |
| `llm_gpt4omini` | `gpt-4o-mini` | full grid | reported; bridge to the TKDE submission |
| `llm_kimik3` | Kimi K3 (`moonshotai/kimi-k3` on OpenRouter, pinned to the Moonshot AI provider) | fixed 30% subsample | reported; open weights, hosted |
| `llm_haiku55_cli`, `llm_sonnet55_cli` | Claude Code CLI on a subscription | any | **pilots only**; written to `pilot_results/` |

Each method also has a token-fallback variant with a `_tf` suffix. Sonnet and Kimi use the same seed, so they see the same 30% of query records, and those records are also part of the Haiku and GPT-4o-mini runs. Claude models run at effort `medium` with adaptive thinking, which cannot be turned off. GPT-4o-mini is not a reasoning model. Kimi K3 runs at its provider's default reasoning setting.

Subscription pilots run `claude -p` from an empty temporary directory, with no tools, no MCP servers, only project-level settings, and a replaced system prompt. `ANTHROPIC_API_KEY` is removed from their environment, so they can never bill the API. The CLI's JSON output fields and whether user-level context still reaches the model have not been checked yet. On the first pilot, compare the logged `input_tokens` with the prompt size, and look at `raw_response`. Use `--workers 2` or so, to stay inside subscription limits.

Setup and checks (none of these call an API):

```bash
cp .env.example .env            # then fill in keys
pip install -r requirements.txt # anthropic==2.6.0, openai==3.46.0
python tests/test_llm_harness.py      # offline tests with fake clients
python scripts/03_estimate_llm_cost.py     # prices one pass from locally built prompts
```

Pilots (subscription; no API cost):

```bash
python scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku55_cli \
    --transformations original,ciphered_words --sample-frac 0.05 --workers 2
```

Reported runs (these call APIs and cost money; about $65-150 per pass of the standard prompt on 2 datasets x 4 conditions):

```bash
python scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini   # full grid
python scripts/04_run_experiments.py --dataset all --methods llm_sonnet55,llm_kimik3     # 30% subsample
python scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations factorial                                                       # 2x2 cells; repeat with
                                                                                      # llm_sonnet55,llm_kimik3 (about +$50 for all 4)
# Options: --effort low|high, --repeat N, --shuffle-candidates, --top-k K, --workers N
```

Robustness to the candidate-list design (reviewer objection to showing 50 candidates per query). The main design is rank-then-select: TF-IDF ranks every candidate for free, and the LLM picks the best of the top 50, listed in rank order. That is how a practical LLM matcher would be built. The shortlist is identical in every condition, because TF-IDF does not change under the ciphers. Two checks, both on Haiku and GPT-4o-mini over Original plus the three v1 ablations, about $10-15 together:

```bash
python scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations original,ciphered_letters,ciphered_words,scrambled --shuffle-candidates             # rank-order hint removed
python scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations original,ciphered_letters,ciphered_words,scrambled --shuffle-candidates --top-k 10  # short list
python scripts/03_estimate_llm_cost.py --models haiku55,gpt4omini --top-k 10                                 # price the k=10 run
```

Results go to `llm_<model>__shuffle0` and `llm_<model>__shuffle0__k10`. Pairwise yes/no prompting (one call per query-candidate pair) is deliberately not run. It is not the rank-then-select setting studied here, and it multiplies token cost. The paper states this as a limitation.

Non-default options are appended to the result directory name (for example `results/abt-buy/llm_haiku55__effort-high/`), so runs never overwrite each other. `05_cv_threshold.py` and `06_summarize_results.py` pick up the new directories automatically.

## Project Structure

```
2026-semantic-ablation-matching/
├── data/
│   ├── abt-buy/
│   │   ├── data_original/           # Source CSVs + ground-truth mapping
│   │   └── data_test/               # Transformed variants + factorial manifest
│   └── amazon-google/
│       ├── data_original/
│       └── data_test/
│
├── scripts/                         # numbered in run order
│   ├── 01_prepare_datasets.py       # v1 transforms: ciphered_letters, ciphered_words, scrambled
│   ├── 02_prepare_factorial.py      # 2x2 cells: desc_ciphered, ids_ciphered, all_ciphered
│   ├── 03_estimate_llm_cost.py      # offline LLM cost estimate (no API calls)
│   ├── 04_run_experiments.py        # all methods; LLMs through harness v2
│   ├── 05_cv_threshold.py           # 5-fold CV thresholds -> results/cv_results.csv
│   ├── 06_summarize_results.py      # summary table -> results/summary.csv
│   ├── 08_figures.R                 # publication figures -> figs/
│   └── lib/                         # shared code imported by the scripts
│       ├── paths.py                 # repo-root-anchored paths
│       ├── datasets.py              # dataset config and loading
│       ├── transforms.py            # v1 transform functions
│       ├── factorial.py             # 2x2 factorial transform
│       ├── methods.py               # non-LLM methods + v1 LLM code (reference only)
│       ├── llm_clients.py           # harness v2 model clients
│       └── llm_runner.py            # harness v2 runner
│
├── tests/                           # offline tests (no API calls)
│
├── results/
│   ├── {dataset}/{method}/          # Per-method parquet files
│   ├── v1_llm/{dataset}/{method}/   # Superseded v1 LLM results (contaminated; see Harness v2)
│   ├── summary.csv                  # Fixed-threshold results with TP/FP/FN and cost
│   └── cv_results.csv               # Cross-validated F1 (authoritative for figures/tables)
│
├── figs/                            # Generated figures
├── checkpoints/                     # gitignored: partial harness-v2 runs
├── pilot_results/                   # gitignored: subscription pilots, never reported
└── README.md
```

---

## Key Results Files

- **`results/cv_results.csv`** -- Threshold-optimized, cross-validated F1/precision/recall. **Use this for figures and tables.**
- **`results/summary.csv`** -- Fixed-threshold results. Has TP/FP/FN counts and LLM `cost_usd`/`wall_time_seconds`, but thresholds are not F1-optimal. Use for cost/runtime analysis only.
