

<!-- README.md is generated from README.qmd. Edit README.qmd, then run: quarto render README.qmd -->

# Matching Without Meaning

**Do LLM entity matchers fall back on lexical evidence when the meaning
is removed?**

LLM entity matchers lean on semantics. We test what happens when meaning
is removed but the lexical evidence that identifies a match is kept.
TF-IDF is the control: it sees the same evidence in every condition, so
its F1 does not move, which shows that evidence alone still solves the
task.

> **Status.** This README is rendered from `README.qmd` and reads every
> number from `results/` and `data/`.
>
> - The non-LLM methods were rerun on 2026-11-08 with the current code,
>   on all seven conditions.
> - `openai_embeddings` has not been rerun yet (it is a paid call), so
>   its rows still come from the earlier code.
> - Harness v2 LLM runs: not yet run.
> - The v1 LLM results from the TKDE submission are superseded. They
>   appear in the figures labeled “(v1)” and in [their own
>   section](#v1-llm-results-superseded).

## Research question

Does an LLM entity matcher use the evidence that actually distinguishes
entities, or does it depend on meaning the task does not require?

We use *controlled semantic ablation*: transforms that remove meaning
while keeping each token’s identity across the two catalogs intact. A
method that relies only on which tokens the two records share is
unaffected. A method that relies on meaning degrades.

### The v1 transforms

1.  **Original**: the baseline.
2.  **Ciphered Letters**: one consistent character substitution applied
    to both catalogs (for example ‘a’ becomes ‘x’).
3.  **Ciphered Words**: one consistent word replacement applied to both
    catalogs (for example ‘Sony’ becomes ‘Goat’).
4.  **Scrambled**: a deterministic letter scramble within each word,
    first and last letters kept, identical across catalogs.

### The 2x2 factorial: descriptive words x identifiers

|                           | identifiers intact | identifiers ciphered |
|---------------------------|--------------------|----------------------|
| **descriptions intact**   | `original`         | `ids_ciphered`       |
| **descriptions ciphered** | `desc_ciphered`    | `all_ciphered`       |

- **Identifiers** are model numbers and SKUs: pieces that contain both a
  letter and a digit (`w55`, `pslx350h`), plus short non-English
  prefixes attached to them (`dsc` in `dsc-w55`). Quantities (`4gb`,
  `1080p`) are not identifiers, and neither are pieces that appear in
  more than 20 records.
- **Descriptive words** are everything else, including brand names.
- Descriptive words become pronounceable nonce words of the same length
  that are not English words. Identifiers become random codes of the
  same shape. Both maps are global and injective, so TF-IDF scores are
  identical in every cell (checked to 1e-9 by
  `02_prepare_factorial.py --verify`).

Identifier coverage (seed 42, from `factorial_manifest.json`): on
Abt-Buy, 49.8% of true pairs share an identifier piece and 42.6% share
one that occurs in no other record. On Amazon-Google the shares are 4.4%
and 0.8%, so the identifier factor mainly matters on Abt-Buy.

## Results

All F1, precision, and recall values use 5-fold cross-validated
thresholds (`results/cv_results.csv`).

### Overview figures

![F1 by transformation: semantic methods, with TF-IDF as the dashed
reference line](figs/f1_comparison.png)

*F1 under each v1 transform for the semantic methods. Solid line and
grey band: the method’s own Original F1 (± 1 SD across folds). Dashed
red line: TF-IDF F1 on Original, the lexical control. LLM rows labeled
“(v1)” are the superseded TKDE runs; they will be replaced by the
harness v2 runs.*

![Precision and recall: semantic methods, with TF-IDF as the red
diamonds](figs/precision_recall_scatter.png)

*Precision and recall at the cross-validated threshold. Red diamonds:
TF-IDF in each condition (they coincide, because TF-IDF is invariant).*

### F1 by method and condition

The semantic methods, with TF-IDF as the lexical control. Across all
seven conditions, TF-IDF F1 ranges from 0.696 to 0.704 on Abt-Buy and
from 0.807 to 0.808 on Amazon-Google.

**Abt-Buy** (name + description):

| Method              | Original | Ciphered Letters | Ciphered Words | Scrambled |
|:--------------------|---------:|-----------------:|---------------:|----------:|
| OpenAI Embeddings   |    0.797 |            0.404 |          0.422 |     0.507 |
| SentenceTransformer |    0.583 |            0.149 |          0.348 |     0.359 |
| TF-IDF              |    0.697 |            0.697 |          0.700 |     0.704 |

**Amazon-Google** (title only):

| Method              | Original | Ciphered Letters | Ciphered Words | Scrambled |
|:--------------------|---------:|-----------------:|---------------:|----------:|
| OpenAI Embeddings   |    0.802 |            0.639 |          0.700 |     0.723 |
| SentenceTransformer |    0.755 |            0.617 |          0.683 |     0.675 |
| TF-IDF              |    0.808 |            0.808 |          0.807 |     0.808 |

### Precision, recall, and F1 on Original

**Abt-Buy:**

| Method              | Precision | Recall |    F1 | F1 SD (5 folds) |
|:--------------------|----------:|-------:|------:|----------------:|
| OpenAI Embeddings   |     0.798 |  0.796 | 0.797 |           0.009 |
| SentenceTransformer |     0.584 |  0.581 | 0.583 |           0.027 |
| TF-IDF              |     0.705 |  0.689 | 0.697 |           0.025 |

**Amazon-Google:**

| Method              | Precision | Recall |    F1 | F1 SD (5 folds) |
|:--------------------|----------:|-------:|------:|----------------:|
| OpenAI Embeddings   |     0.803 |  0.801 | 0.802 |           0.028 |
| SentenceTransformer |     0.759 |  0.751 | 0.755 |           0.040 |
| TF-IDF              |     0.808 |  0.808 | 0.808 |           0.030 |

### 2x2 factorial

F1 in each cell of the descriptive-words x identifiers design.

**Abt-Buy:**

| Method | Original | Descriptions ciphered | Identifiers ciphered | Both ciphered |
|:---|---:|---:|---:|---:|
| OpenAI Embeddings | 0.797 | not run | not run | not run |
| SentenceTransformer | 0.583 | 0.362 | 0.526 | 0.300 |
| TF-IDF | 0.697 | 0.697 | 0.697 | 0.696 |

**Amazon-Google:**

| Method | Original | Descriptions ciphered | Identifiers ciphered | Both ciphered |
|:---|---:|---:|---:|---:|
| OpenAI Embeddings | 0.802 | not run | not run | not run |
| SentenceTransformer | 0.755 | 0.671 | 0.742 | 0.665 |
| TF-IDF | 0.808 | 0.808 | 0.808 | 0.808 |

### Candidate-list robustness

The main design shows the LLM the TF-IDF top 50 candidates in rank
order. Two checks remove the rank-order hint (`__shuffle0`) and shorten
the list to 10 (`__shuffle0__k10`).

*Not yet run.*

### LLM output status

Share of rows by status (`ok`, `no_match`, `parse_fail`, `truncated`,
`refusal`, `api_error`). Parse failures are reported as their own
outcome, never counted as “no match”.

*Not yet run.*

### Shortlist recall

Share of queries whose true match is in the TF-IDF top-k shortlist.

*Not yet run.*

### Instance-level analysis

Including the share of LLM misses where the true pair shares an
identifier piece that no other candidate has.

*Not yet run.*

### Cost and wall time

Harness v2 LLM runs only.

*Not yet run.*

## Appendix: lexical methods

Soft TF-IDF, Levenshtein, Jaro-Winkler, and Monge-Elkan appear only
here. Their role is to show that lexical methods are robust to the
transforms, which holds by design.

![F1 by transformation: lexical methods](figs/f1_comparison_lexical.png)

![Precision and recall: lexical
methods](figs/precision_recall_scatter_lexical.png)

**Abt-Buy:**

| Method | Original | Ciphered Letters | Ciphered Words | Scrambled | Descriptions ciphered | Identifiers ciphered | Both ciphered |
|:---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.697 | 0.697 | 0.700 | 0.704 | 0.697 | 0.697 | 0.696 |
| Soft TF-IDF | 0.692 | 0.692 | 0.669 | 0.687 | 0.693 | 0.687 | 0.691 |
| Monge-Elkan | 0.287 | 0.287 | 0.191 | 0.273 | 0.271 | 0.242 | 0.252 |
| Levenshtein | 0.058 | 0.058 | 0.033 | 0.072 | 0.070 | 0.057 | 0.064 |
| Jaro-Winkler | 0.058 | 0.058 | 0.046 | 0.065 | 0.055 | 0.047 | 0.049 |

**Amazon-Google:**

| Method | Original | Ciphered Letters | Ciphered Words | Scrambled | Descriptions ciphered | Identifiers ciphered | Both ciphered |
|:---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.808 | 0.808 | 0.807 | 0.808 | 0.808 | 0.808 | 0.808 |
| Soft TF-IDF | 0.803 | 0.803 | 0.800 | 0.807 | 0.808 | 0.805 | 0.808 |
| Monge-Elkan | 0.740 | 0.740 | 0.700 | 0.735 | 0.701 | 0.736 | 0.699 |
| Levenshtein | 0.555 | 0.555 | 0.532 | 0.569 | 0.551 | 0.554 | 0.549 |
| Jaro-Winkler | 0.488 | 0.488 | 0.483 | 0.489 | 0.486 | 0.482 | 0.487 |

## v1 LLM results (superseded)

These are the LLM results from the TKDE submission, kept in
`results/v1_llm/`. **Do not cite them.** The v1 harness parsed free-text
answers by line prefix and stored API errors and unparseable answers as
blank “no match” rows. An audit on 2026-11-06 found such rows throughout
the ablation conditions; for example, every Claude Haiku token-fallback
Scrambled row failed. Harness v2 replaces them.

**Abt-Buy:**

| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|:---|---:|---:|---:|---:|
| GPT-4o-mini (v1) | 0.928 | 0.779 | 0.510 | 0.601 |
| GPT-4o-mini, token fallback (v1) | not run | 0.703 | 0.557 | 0.632 |
| Claude Haiku 4.5 (v1) | 0.961 | 0.656 | 0.209 | 0.636 |
| Claude Haiku 4.5, token fallback (v1) | 0.944 | 0.067 | 0.007 | 0.000 |
| GPT-4o (v1) | 0.955 | not run | not run | not run |
| Claude Sonnet 4.6 (v1) | 0.954 | not run | not run | not run |

**Amazon-Google:**

| Method | Original | Ciphered Letters | Ciphered Words | Scrambled |
|:---|---:|---:|---:|---:|
| GPT-4o-mini (v1) | 0.821 | 0.582 | 0.576 | 0.555 |
| GPT-4o-mini, token fallback (v1) | not run | 0.653 | 0.633 | 0.707 |
| Claude Haiku 4.5 (v1) | 0.895 | 0.456 | 0.581 | 0.853 |
| Claude Haiku 4.5, token fallback (v1) | 0.881 | 0.378 | 0.011 | 0.000 |
| GPT-4o (v1) | 0.904 | 0.749 | 0.751 | 0.850 |
| Claude Sonnet 4.6 (v1) | 0.843 | not run | not run | not run |

## Methods evaluated

- **LLMs (harness v2):** Claude Haiku 5.5 and GPT-4o-mini on the full
  grid; Claude Sonnet 5.5 and Kimi K3 on a fixed 30% subsample of query
  records. Each query sees the TF-IDF top-50 candidates and returns
  structured JSON.
- **Embeddings:** SentenceTransformer (all-MiniLM-L6-v2) and OpenAI
  Embeddings (text-embedding-3-small).
- **Lexical control:** TF-IDF, cosine similarity on term-frequency
  vectors.
- **Other lexical methods (appendix):** Soft TF-IDF (IDF-weighted, with
  Jaro-Winkler fuzzy token matching at 0.9 or above), Monge-Elkan
  (token-level Jaro-Winkler averaging), Levenshtein, and Jaro-Winkler.

## Datasets

Both datasets are filtered to records with at least one ground-truth
match, following standard benchmark practice.

- **Abt-Buy**: 1,097 ground-truth pairs, 1,081 query records. Methods
  see the name and the full description.
- **Amazon-Google**: 1,300 ground-truth pairs, 1,113 query records.
  Methods see the title only.

## Evaluation protocol

- **Thresholds.** Each method’s decision threshold is chosen by 5-fold
  cross-validation over τ ∈ {0.05, 0.10, …, 0.95}: the threshold is
  picked on four folds and scored on the fifth (seed 42).
  `results/cv_results.csv` holds the mean and SD of F1 across folds.
- **LLM shortlist.** TF-IDF ranks every candidate, and the LLM chooses
  among the top 50. TF-IDF scores do not change under the transforms, so
  every condition sees the same shortlist.

## Reproducing the results

### Prerequisites

Python is managed with [uv](https://docs.astral.sh/uv/).
`pyproject.toml` pins the direct dependencies, `uv.lock` pins the full
tree, and `.python-version` pins Python 3.10.

``` bash
uv sync                        # creates .venv/ with the locked versions (incl. dev: pytest)
uv run tests/test_llm_harness.py
```

Every command below runs with `uv run`, which uses that environment.
Without uv, `pip install -r requirements.txt` installs the same locked
versions (the file is generated by `uv export`; regenerate it after any
`uv lock`).

R (figures and this README) and [Quarto](https://quarto.org):

``` r
install.packages(c("tidyverse", "cowplot", "ggh4x", "ragg", "knitr", "rmarkdown", "jsonlite"))
```

`08_figures.R` also needs the Roboto Condensed font installed
system-wide.

API keys for the LLM and embedding methods go in `.env` (see
`.env.example`):

``` bash
cp .env.example .env           # then fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY
```

### Step 1: Transformed datasets

The v1 transforms are committed in `data/{dataset}/data_test/`. Do not
regenerate them: the Ciphered Words map depends on `PYTHONHASHSEED`, and
the committed CSVs are the only copy of the map the paper used.

``` bash
uv run scripts/01_prepare_datasets.py --dataset all     # v1 transforms (see above before running)
uv run scripts/02_prepare_factorial.py --dataset all --verify   # 2x2 cells + TF-IDF invariance check
uv run tests/test_factorial.py
```

### Step 2: Run the methods

``` bash
# Non-LLM methods (free), all v1 transforms plus the factorial cells
uv run scripts/04_run_experiments.py --dataset all \
    --methods tfidf,soft_tfidf,levenshtein,jaro_winkler,monge_elkan,sentence_transformer \
    --transformations all,factorial
```

Results are written as parquet files to
`results/{dataset}/{method}/{transformation}.parquet`. LLM runs are
described under [Harness v2](#harness-v2-llm-runs).

### Step 3: Tables

``` bash
uv run scripts/05_cv_threshold.py        # -> results/cv_results.csv (source of every reported F1)
uv run scripts/06_summarize_results.py   # -> results/summary.csv (TP/FP/FN, cost, wall time)
```

### Step 4: Figures

``` bash
Rscript scripts/08_figures.R             # -> figs/*.png
```

### Step 5: This README

``` bash
quarto render README.qmd                 # -> README.md
```

## Harness v2 (LLM runs)

The v1 LLM pipeline parsed free-text answers by line prefix and stored
API errors and unparseable answers as blank “no match” rows. Harness v2
replaces it for all new LLM runs.

- **One code path for every model** (`scripts/lib/llm_clients.py`,
  `scripts/lib/llm_runner.py`) with structured JSON output (`match`,
  `confidence`, `rationale`).
- **Row-level `status`**: `ok`, `no_match`, `parse_fail`, `truncated`,
  `refusal`, `api_error`. Raw responses, stop reasons, token counts,
  cost, served model, and request ids are saved for every row.
- **Fails loudly.** A bad key, missing credit, or unknown model aborts
  at once. Five consecutive API errors, or more than 1% of calls, also
  abort the run. A run with unresolved API errors is never written to
  `results/`. Partial progress goes to `checkpoints/`, and rerunning the
  same command resumes it.
- **No sampling parameters.** Current Claude models reject
  `temperature`, so run-to-run variance is measured with `--repeat`.
- **Instance-level diagnostics** on every row: the rank of the true
  match among the shortlisted candidates, the candidate ids, and the
  prompt size.

Models (`llm_clients.MODELS`; prices checked 2026-11-06, re-check before
budgeting):

| Method | Model | Run | Notes |
|----|----|----|----|
| `llm_haiku55` | `claude-haiku-5-5` | full grid | reported |
| `llm_sonnet55` | `claude-sonnet-5-5` | fixed 30% subsample | reported |
| `llm_gpt4omini` | `gpt-4o-mini` | full grid | reported; bridge to the TKDE submission |
| `llm_kimik3` | Kimi K3 (`moonshotai/kimi-k3` on OpenRouter, pinned to the Moonshot AI provider) | fixed 30% subsample | reported; open weights, hosted |
| `llm_haiku55_cli`, `llm_sonnet55_cli` | Claude Code CLI on a subscription | any | **pilots only**; written to `pilot_results/` |

Each method also has a token-fallback variant with a `_tf` suffix.
Sonnet and Kimi use the same seed, so they see the same 30% of query
records, and those records are also part of the Haiku and GPT-4o-mini
runs. Claude models run at effort `medium` with adaptive thinking, which
cannot be turned off. GPT-4o-mini is not a reasoning model. Kimi K3 runs
at its provider’s default reasoning setting.

Subscription pilots run `claude -p` from an empty temporary directory,
with no tools, no MCP servers, only project-level settings, and a
replaced system prompt. `ANTHROPIC_API_KEY` is removed from their
environment, so they can never bill the API. Use `--workers 2` or so, to
stay inside subscription limits.

Checks that make no API calls:

``` bash
uv run tests/test_llm_harness.py               # offline tests with fake clients
uv run scripts/03_estimate_llm_cost.py         # prices one pass from locally built prompts
```

Pilot (subscription; no API cost):

``` bash
uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku55_cli \
    --transformations original,ciphered_words --sample-frac 0.05 --workers 2
```

Reported runs (these call paid APIs; `03_estimate_llm_cost.py` prices
them first):

``` bash
uv run scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini   # full grid
uv run scripts/04_run_experiments.py --dataset all --methods llm_sonnet55,llm_kimik3     # 30% subsample
uv run scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations factorial                                                       # 2x2 cells
# Options: --effort low|high, --repeat N, --shuffle-candidates, --top-k K, --workers N
```

Candidate-list robustness, on Haiku and GPT-4o-mini over the four v1
conditions:

``` bash
uv run scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations original,ciphered_letters,ciphered_words,scrambled --shuffle-candidates             # rank-order hint removed
uv run scripts/04_run_experiments.py --dataset all --methods llm_haiku55,llm_gpt4omini \
    --transformations original,ciphered_letters,ciphered_words,scrambled --shuffle-candidates --top-k 10  # short list
```

Non-default options are appended to the result directory name (for
example `results/abt-buy/llm_haiku55__shuffle0/`), so runs never
overwrite each other. `05_cv_threshold.py` and `06_summarize_results.py`
pick up new directories automatically. Pairwise yes/no prompting (one
call per query-candidate pair) is deliberately not run: it is not the
rank-then-select setting studied here, and it multiplies token cost.

## Project structure

    2026-semantic-ablation-matching/
    ├── data/
    │   ├── abt-buy/
    │   │   ├── data_original/           # source CSVs + ground-truth mapping
    │   │   └── data_test/               # transformed variants + factorial manifest
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
    │   ├── 08_figures.R                 # figures -> figs/
    │   └── lib/                         # shared code imported by the scripts
    │
    ├── tests/                           # offline tests (no API calls)
    │
    ├── results/
    │   ├── {dataset}/{method}/          # per-method parquet files
    │   ├── v1_llm/{dataset}/{method}/   # superseded v1 LLM results
    │   ├── cv_results.csv               # cross-validated F1/precision/recall (source of every reported F1)
    │   └── summary.csv                  # fixed-threshold TP/FP/FN, cost, wall time
    │
    ├── figs/                            # generated figures (main + *_lexical appendix figures)
    ├── checkpoints/                     # gitignored: partial harness-v2 runs
    ├── pilot_results/                   # gitignored: subscription pilots, never reported
    ├── README.qmd                       # source of this README
    └── README.md                        # rendered; do not edit by hand
