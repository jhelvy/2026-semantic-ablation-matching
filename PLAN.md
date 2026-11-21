# Plan: revision experiments and repo cleanup

Written 2026-11-08 for the revision of "Matching Without Meaning" after the TKDE reject (TKDE-2026-06-2289). The full revision plan, including the manuscript edits and venue strategy, is in the paper repo (`2026-semantic-ablation-paper/pvldb/1/plan.md`). This file covers only the changes to this repo: how it is organized, how the README is built, and what we will run.

**Two plans, kept in sync.** The paper plan is `/Users/jhelvy/gh/research/2026-semantic-ablation-paper/pvldb/1/plan.md` (argument, manuscript edits, venue); it moved there on 2026-11-10, from `~/sync/00_current/llm-fuzzy-joins/1/` by way of `tkde/1/`. When John says "update the plan", update both files in the same sitting, and keep shared decisions consistent: lineup, figures, run order, and status. For script names and commands, this file is the source of truth.

Branch: `harness-v2`. John confirmed keys and credit on 2026-11-09; the paid runs through 2026-11-10 cost about $60 (C.4).

**Commits: John makes every commit.** Claude does not run `git commit` (or push). At each natural commit point, Claude stops, summarizes what changed and how it was checked, proposes a commit message, and waits. John reviews the diff and commits. Wherever this plan says "commit", read it as "stop and flag a commit point".

## The claim the experiments must support

LLM entity matchers lean on semantics. When meaning is removed but the lexical evidence that identifies a match is kept, they do not fall back on that evidence, even though that evidence alone solves the task. TF-IDF is the control that proves the evidence is still there: it does not change under any of the transforms.

**Revised framing (decided 2026-11-10, after the v2 runs).** LLM matchers win through meaning. When meaning is present, or can be recovered from the transformed text, a strong small model stays well above the lexical baseline. When meaning is truly removed but the lexical evidence is kept, the LLMs add nothing over that baseline, and on Abt-Buy they do worse than TF-IDF's own top pick. The claim that LLMs do not fall back on the preserved evidence still holds; what changed is which transforms actually remove meaning. The evidence:

- **Some transforms can be read** (decode test, `09_decode_test.py`, $0.49). All three models recover most of the Scrambled text (65–99% of altered words). Ciphered Letters is a fixed letter substitution that can be broken: Sonnet recovers 90–95% when it answers, and Luna decoded some batches almost fully (up to 93%) but often answered "unrecoverable". Ciphered Words, the control, cannot be decoded (4% or less).
- **On readable transforms, Luna holds up.** Abt-Buy CV F1: Ciphered Letters 0.93 and Scrambled 0.88, against TF-IDF 0.70.
- **On transforms that remove meaning, Luna falls to the TF-IDF level.** Abt-Buy: Ciphered Words 0.70 vs 0.70, `all_ciphered` 0.74 vs 0.70. Amazon-Google: 0.84 vs 0.81 on both.
- **Against TF-IDF's top-1 on the same shortlist, Luna does worse** (`07_instance_analysis.py`). Abt-Buy Ciphered Words: Luna picks the right match on 0.641 of queries, TF-IDF's top candidate is right on 0.702. Luna is right where TF-IDF is wrong on 5% of queries, and wrong where TF-IDF is right on 11%. `all_ciphered`: 0.639 vs 0.692. On Amazon-Google the two tie (0.803 vs 0.805; 0.815 vs 0.807). Paired bootstrap, Luna minus TF-IDF top-1: Abt-Buy Ciphered Words −6.1 points [−8.7, −3.7], `all_ciphered` −5.3 [−7.8, −2.7]; on Amazon-Google both CIs include zero.
- **Luna misses decisive identifiers it caught on Original.** On 43.5% of Abt-Buy queries, the query and its true match share an identifier piece (model number, SKU) that no other shortlisted candidate has; TF-IDF's top pick is right on 93–94% of them in every condition. Luna gets 99.6% of these on Original, 86.8% on Ciphered Words; 16% of its Ciphered Words misses had such a piece. (Amazon-Google titles have such a piece on only 1.2% of queries, so this check is Abt-Buy only.)
- **Prompt length does not explain the drops (R2 D1c).** Within a condition, input tokens barely correlate with a correct pick (|Spearman| at most 0.2, mostly near 0). At the same prompt size (Abt-Buy, 1,304–1,541 tokens), Luna is right on 0.97 of Original queries and 0.65 of Ciphered Words queries.
- **The Anthropic gap on readable transforms is refusal, not ability.** Haiku refuses 85–97% of Ciphered Letters queries and Sonnet 28–36%. Scored only on the queries Sonnet answered, Sonnet matches Luna (Abt-Buy Ciphered Letters 0.908 vs 0.914; TF-IDF 0.688). On Ciphered Words, on the same answered queries, Sonnet scores 0.553, Luna 0.705, TF-IDF 0.685.

What this changes:

- **Conditions come in two groups.** Ciphered Letters and Scrambled are *meaning-recoverable*: a capable model can read them, so robustness there is not matching without meaning. Ciphered Words and the factorial cells are the *meaning-removal* tests, and they carry the claim. Do not write that every ablation removes meaning.
- **The v1 deficits were partly a model effect.** The weaker v1 models could not decode the ciphers; Luna can. Decided in the Step 1 draft (2026-11-10): the paper cites no v1 numbers, since those runs were contaminated. The Limitations paragraph says instead that a stronger model may decode transforms ours cannot, as the readable conditions already show.
- **Practical message.** With meaningful product text, an LLM reranking a TF-IDF shortlist is very strong, and model choice matters (refusals above all). The limitation bites on low-meaning text: SKUs, part codes, abbreviations, opaque IDs. There, the LLM adds cost but no accuracy over TF-IDF, can override correct lexical evidence, and some models decline the task.
- **Readable conditions stay in the main figures (decided 2026-11-10).** John likes the readable vs meaning-removed contrast. The figures label Ciphered Letters and Scrambled "(readable)" and Ciphered Words "(meaning removed)", separated by a dotted line.
- **Identifiers count when they look like identifiers (found while drafting, 2026-11-10).** On the 43.5% of Abt-Buy queries with a unique shared identifier, Luna is right on 99.6% (Original), 98.1% with both factorial factors ciphered (codes become codes of the same shape), and 86.8% under Ciphered Words (codes become dictionary words). The model uses lexical evidence that has the form of a code, and often ignores it when it does not. The draft's Discussion has a short section on this; John to confirm the reading.

This is a proof of concept, not a model survey. Keep the lineup small and spend the effort on the controls. (Revised 2026-11-20: breadth is added cheaply, with more models on a fixed 30% query subsample and four conditions, plus a Ditto-style supervised baseline; see Part D.)

**Target venue (decided 2026-11-20, John): Data & Knowledge Engineering (DKE).** It replaces PVLDB, which becomes the alternative. DKE published the closest prior work, Moslemi et al. (DKE 164, 2026), semantic perturbations on Abt-Buy and WDC for supervised matchers. The framing above is confirmed; recasting the paper as "semantic ablation testing" was considered and dropped as the main contribution because the general idea has prior art (details and positioning in the paper plan). The manuscript will move to the Elsevier template in a new `dke/1/` folder in the paper repo; until then, copy figures to `pvldb/1/source/`.

**Paper status (2026-11-10).** Target venue then: PVLDB (Experiments & Analysis). The manuscript is on the PVLDB template at `pvldb/1/source/main.tex` in the paper repo, and the Step 1 rewrite is drafted there with every number from `results/`. The title is still being workshopped, around LLM entity matchers relying on semantic meaning. The paper uses `figs/` as is: `f1_comparison.png`, `precision_recall_scatter.png`, `factorial_2x2.png` in the main text; `*_lexical.png`, `candidate_list_robustness.png`, `llm_scale_check.png` in the appendix. The token-fallback comparison is a table, not a figure. After any figure change here, copy the PNGs into `pvldb/1/source/`.

### Method lineup (decided 2026-11-08)

- **Semantic methods (the object of study):** the LLMs (C.1: Haiku 5.5, GPT-6 Luna, Sonnet 5.5) and the two embedding models (`sentence_transformer`, `openai_embeddings`). These are the methods whose sensitivity to semantics the paper measures.
- **Main results use the small models (decided 2026-11-09).** Haiku 5.5 and GPT-6 Luna are what people deploy for matching at scale, so the key figures and core results show them, the embedding models, and the TF-IDF reference. Sonnet 5.5 is a scale check: it answers "a stronger model would read the identifiers". Its results go in a separate supplementary figure, next to the lexical ones. Move Sonnet into the main figures only if it clearly recovers on the transformed conditions: on the same 30% queries, its F1 beats both small models' by more than the paired 95% CI. Otherwise it stays in the supplement, and the text cites it once: a larger model does not fix the failure.
- **One lexical control in the main text: TF-IDF.** It is invariant by construction, and it also produces the LLM's shortlist. It appears as a reference in every main figure (a dashed or differently colored line in F1 plots, a colored point in precision-recall plots). It is not presented as a competitor, and the text does not say lexical methods "overtake" the LLM.
- **Other lexical methods go to the appendix:** Soft TF-IDF, Levenshtein, Jaro-Winkler, Monge-Elkan. Their only role is to show that lexical methods are robust to the transforms, which is true by design. Still run them (they cost nothing), so the appendix and the README can show them, and the main text points to the appendix once.

## Order of work

1. **Reorganize the repo (Part A).** Do this first, before any API spend. Paths, checkpoint locations, and result directory names change in this step. Checkpoints written under the old layout would not resume under the new one, and none exist yet.
2. **Build the README from `README.qmd` (Part B).**
3. **Finish the zero-cost build items (Part C.3).**
4. **Run the experiments in the order in Part C.4.**
5. **Re-render the README after each batch of results.**
6. **Add the third dataset and the previous-generation models (Part D, added 2026-11-13).**

---

## Part A: Repo reorganization

### A.1 Number the scripts in run order

Python cannot import a module whose name starts with a digit (`import 01_prepare` is a syntax error). Several scripts currently import from each other:

- `run_experiments.py` exports `TRANSFORMATIONS` and `load_dataset`;
- `prepare_datasets.py` exports `DATASETS`, `get_text_columns`, and `load_csv`;
- `test_factorial.py` imports `prepare_factorial`.

So the split is: shared code moves into an unnumbered library package, and the numbered scripts become thin command-line entry points that only parse arguments and call the library.

Proposed layout:

```
scripts/
├── 01_prepare_datasets.py      # v1 transforms: ciphered_letters, ciphered_words, scrambled
├── 02_prepare_factorial.py     # 2x2 cells: desc_ciphered, ids_ciphered, all_ciphered
├── 03_estimate_llm_cost.py     # offline cost estimate (was llm_cost_estimate.py)
├── 04_run_experiments.py       # all methods; LLMs through harness v2
├── 05_cv_threshold.py          # 5-fold CV thresholds -> results/cv_results.csv
├── 06_summarize_results.py     # summary + LLM status tables (was analyze_results.py)
├── 07_instance_analysis.py     # NEW: per-query diagnostics; absorbs diagnose_blocking.py
├── 08_figures.R                # all figures -> figs/ (was charts.R)
└── lib/
    ├── __init__.py
    ├── paths.py                # REPO_ROOT, DATA, RESULTS, FIGS, CHECKPOINTS, PILOT_RESULTS
    ├── datasets.py             # DATASETS, load_csv, get_text_columns, load_dataset, TRANSFORMATIONS
    ├── transforms.py           # v1 transform functions (from prepare_datasets.py)
    ├── factorial.py            # classify_pieces, build_map, apply_cell, verify (from prepare_factorial.py)
    ├── methods.py              # non-LLM methods + v1 LLM code (kept for reference)
    ├── llm_clients.py
    └── llm_runner.py
tests/
├── test_llm_harness.py
└── test_factorial.py
```

Rules for the move:

- **One source of truth for paths.** `lib/paths.py` anchors every path to the repo root, so scripts work from any working directory. Remove the `sys.path.append(Path(__file__).parent)` lines and the scattered `Path(__file__).resolve().parent.parent` constructions. The numbered scripts add `scripts/` to the path once and use `from lib import ...`.
- **Pure moves first, then fixes.** Use `git mv` so history follows the files. Keep the moves in a separate commit from any code change, so the diff is reviewable. (Done 2026-11-08: `078ab3d` moves, `f687214` fixes; made before the no-commit rule above.)
- **Drop `diagnose_blocking.py`.** Its job (is the true match in the TF-IDF top-k?) is covered by `07_instance_analysis.py`, which computes shortlist recall for each k from TF-IDF alone, at no cost. Its hard-coded filename guessing goes away with it.
- **Remove the matplotlib plots from `06_summarize_results.py`.** Those plots (`{dataset}_f1_scores.png`, `{dataset}_precision_recall.png`) are not used anywhere. All figures come from `08_figures.R`.
- **Keep the CLI flags unchanged** (`--dataset`, `--methods`, `--transformations`, `--shuffle-candidates`, `--top-k`, and so on), so the commands in this plan only change by script name.

### A.2 Figures go to `figs/`

- Create `figs/` at the repo root, and `git mv results/plots/*.png figs/`.
- Change every `ggsave(file.path('results', 'plots', ...))` in `08_figures.R` to `file.path('figs', ...)`.
- Keep the figure file names stable (`f1_comparison.png`, `precision_recall_scatter.png`, `llm_robustness.png`, `llm_prompt_comparison.png`), so the paper's `\includegraphics` paths only change by folder. New figures for the factorial and robustness runs get descriptive names, such as `factorial_2x2.png` and `candidate_list_robustness.png`.
- `08_figures.R` also needs labels for the new methods (`llm_haiku55`, `llm_gpt6luna`, `llm_sonnet55`) and for the factorial cells.
- `llm_sonnet55` stays out of every main figure. The scale-check figure compares the models on the same queries, so the small models' rows are restricted to Sonnet's 30% subsample. Their full-grid numbers are not used in this figure.
- **Split the two overview figures into a main figure and an appendix figure** (see "Method lineup" above). Each one is currently a stacked cowplot with semantic methods on top and lexical methods below; the halves become separate files.

  | File | Goes in | Rows / points | TF-IDF shown as |
  |---|---|---|---|
  | `f1_comparison.png` | main text | semantic methods only (LLMs, embeddings) | a dashed, differently colored vertical reference line at TF-IDF's F1 in each dataset panel, labeled "TF-IDF" |
  | `f1_comparison_lexical.png` | appendix | lexical methods (TF-IDF, Soft TF-IDF, Monge-Elkan, Levenshtein, Jaro-Winkler) | an ordinary row |
  | `precision_recall_scatter.png` | main text | semantic methods only | one distinctly colored point per dataset and condition, labeled "TF-IDF" |
  | `precision_recall_scatter_lexical.png` | appendix | lexical methods | ordinary points |
  | `llm_scale_check.png` | appendix | Sonnet 5.5 vs Haiku 5.5 and GPT-6 Luna, all on Sonnet's 30% queries, every condition (v1 four + factorial cells) | the dashed reference line, also computed on the 30% queries |

  - TF-IDF is invariant to within about 0.01 F1, so one reference line per dataset (its Original F1) is enough in the F1 figure. If a factorial or seed run ever moves it more than that, draw it per condition instead.
  - Keep the main figures' file names, so the paper's `\includegraphics` paths still work. The legend or caption names the TF-IDF reference as the lexical control.
  - The other figures (`llm_robustness.png`, `llm_prompt_comparison.png`, and the new factorial and robustness figures) also get the TF-IDF reference line, so every main figure uses the same convention.

### A.3 Results layout

- `results/{dataset}/{method}/{transformation}.parquet` stays as it is. Non-default flags already add a suffix to the method directory (`llm_haiku55__k10`, `llm_haiku55__rep1`).
- **Move the v1 LLM results out of the main tree** to `results/v1_llm/{dataset}/{method}/`. These are `llm`, `llm_haiku`, `llm_haiku_token_fallback`, `llm_claude`, `llm_gpt4o`, and `llm_gpt4o_mini_token_fallback`. The 2026-11-06 audit found them contaminated by blank rows from parse failures and API errors. Keeping them in the main tree risks mixing them into v2 tables, because `05_cv_threshold.py` and `06_summarize_results.py` scan every method directory. Keep them, because the paper may report the v1-to-v2 comparison for GPT-4o-mini.
- The non-LLM method results (`tfidf`, `soft_tfidf`, `levenshtein`, `jaro_winkler`, `monge_elkan`, `sentence_transformer`, `openai_embeddings`) stay where they are.
- The derived tables live at the top of `results/`, one CSV per topic, so the README can read them directly:
  - `cv_results.csv`: F1, precision, and recall with CV thresholds (already exists);
  - `summary.csv`: TP/FP/FN, cost, and wall time (already exists);
  - `llm_status.csv` (NEW): counts and rates of each row status (`ok`, `no_match`, `parse_fail`, `truncated`, `refusal`, `api_error`) by dataset, model, and condition;
  - `shortlist_recall.csv` (NEW): share of queries whose true match is in the TF-IDF top-k, by dataset, condition, and k;
  - `instance_analysis.csv` (NEW): per-condition summaries from `07_instance_analysis.py`, described in C.3.
- `checkpoints/` and `pilot_results/` stay at the root and stay gitignored.

### A.4 Other cleanup

- Fix the README project tree, which still says `fuzzy-join-llm/`.
- Add `.quarto/` and `README_files/` to `.gitignore`.
- **Manage Python with uv.** `pyproject.toml` pins the direct dependencies, `uv.lock` pins the full tree, `.python-version` pins 3.10, and `uv sync` builds `.venv/`. Run every script with `uv run`. `requirements.txt` is generated from the lock (`uv export --frozen --no-hashes --no-dev --format requirements-txt -o requirements.txt`) for pip users; regenerate it after any `uv lock`.
- Add an R requirements note for `08_figures.R` and `README.qmd` (tidyverse, cowplot, ggh4x, ragg, knitr, jsonlite).

### A.5 Checks that the reorg changed nothing

Run all of these before committing the reorg. None of them call an API.

- `uv run tests/test_llm_harness.py` and `uv run tests/test_factorial.py` pass (15 + 3 tests today).
- `uv run scripts/02_prepare_factorial.py --dataset all --verify` regenerates the factorial CSVs, and `git diff --stat data/` shows no change. The TF-IDF equality check passes.
- Do **not** rerun `01_prepare_datasets.py` as a check. Its Ciphered Words map is built by iterating over a Python `set`, so it changes with `PYTHONHASHSEED`. The committed CSVs are the only copy of the map the paper used (see C.3, item 5).
- `uv run scripts/05_cv_threshold.py` reproduces the committed `results/cv_results.csv` for every non-LLM method. The v1 LLM rows can be re-produced from `results/v1_llm/` once the CV script is pointed there.
  - Result (2026-11-08): it does, row for row, except Soft TF-IDF. Those committed rows came from an older version of the CV script. But the committed non-LLM parquets themselves do not reproduce from the current `methods.py`. So the free non-LLM methods were rerun (`all,factorial`, both datasets), with a per-dataset `title_only` setting in `lib/datasets.py` (Amazon-Google: title only). `05` now reads Soft TF-IDF from its parquets like every other method. `openai_embeddings` waits for keys (C.4 step 2). The rerun is deterministic (identical under a different `PYTHONHASHSEED`). Amazon-Google results are unchanged except Soft TF-IDF. Abt-Buy is lower for every rerun method (TF-IDF 0.760 → 0.697) and is now flat across conditions; the paper plan, Step 0 item 4, has the details.
- `uv run scripts/03_estimate_llm_cost.py --models haiku55,gpt4omini` gives the same estimate as before ($6.48 for shuffled k=50 over the 4 v1 conditions).
- `Rscript scripts/08_figures.R` writes to `figs/`, and nothing is written to `results/plots/`.

---

## Part B: README built from `README.qmd`

### B.1 Setup

- `README.qmd` is the source and `README.md` is the rendered output. Commit both, because GitHub displays `README.md`.
- Front matter:

  ```yaml
  ---
  format: gfm
  execute:
    echo: false
    warning: false
    message: false
  ---
  ```

- Use R chunks (knitr engine), to match `08_figures.R`. Render with `quarto render README.qmd`.
- Add that render as the final step of the pipeline, and list it in the README's own "Reproducing the results" section.

### B.2 Rules

- **No hand-typed result numbers.** Every number in a table or in the prose comes from a file in `results/` or `data/`. Prose uses inline R, for example `` `r f1("abt-buy", "llm_haiku55", "original")` ``. A small set of helper functions (`f1()`, `pct()`, `cost()`) goes in a setup chunk at the top.
- **The README reads only CSV and JSON**, never parquet. The Python scripts write every table the README needs (A.3), so the README depends only on readr and jsonlite.
- **It renders at every stage.** If a results file or a model's rows are missing, the chunk prints "not yet run" in place of the table and does not fail. The README can then be rendered now, before any v2 runs, and again after each batch.
- **Figures are embedded from `figs/`** with plain Markdown image links. The README does not regenerate them.

### B.3 Content and source files

| README section | Source |
|---|---|
| Research question, transforms, 2x2 design | text (no numbers) |
| F1 by method and condition, both datasets: semantic methods + TF-IDF | `results/cv_results.csv` |
| Precision / recall / F1 on Original: semantic methods + TF-IDF | `results/cv_results.csv` |
| Lexical methods, all conditions (separate section, mirrors the paper's appendix) | `results/cv_results.csv` |
| Scale check: Sonnet 5.5 vs the small models on the 30% queries (separate section, mirrors the paper's appendix) | `results/cv_results.csv` (small-model rows recomputed on the 30% queries) |
| 2x2 factorial table (F1 per cell, per model) | `results/cv_results.csv` |
| Candidate-list robustness (ordered k=50 vs ordered k=10) | `results/cv_results.csv` (method suffix `__k10`) |
| LLM output status rates (parse failures, truncation, refusals) | `results/llm_status.csv` |
| Shortlist recall by k | `results/shortlist_recall.csv` |
| Identifier coverage and class counts for the factorial | `data/{dataset}/data_test/factorial_manifest.json` |
| Instance-level stats ("% of LLM misses where a unique shared identifier was present") | `results/instance_analysis.csv` |
| Cost and wall time per model | `results/summary.csv` (v2 rows only) |
| Figures | `figs/*.png` |
| v1 results (TKDE submission) | `results/v1_llm/` via `cv_results.csv`, clearly labeled as superseded and contaminated |

### B.4 Prose

- The current "Key Findings" text is written around v1 numbers that the audit found contaminated. Do not carry it over as current findings.
- Until the v2 runs finish, the README describes the design and shows the non-LLM results. The v1 LLM numbers appear only in a section labeled as superseded.
- Write the findings after the v2 results are in, with every number inline.
- The title claim should match the revised thesis ("fail to fall back on preserved evidence"), not "non-essential semantic priors". Match the paper's new title once it is chosen.

---

## Part C: Experiment changes

### C.1 Model lineup (reported results)

Revised 2026-11-09: the latest small and mid-size models from each provider. GPT-4o-mini (2024) is replaced by GPT-6 Luna, OpenAI's current small model, at the same price as Haiku 5.5. Kimi K3 is dropped: it only added breadth and was the most expensive model. IDs and prices were verified 2026-11-09 (OpenAI) and 2026-11-06 (Anthropic). Re-check them before running.

| Method | Model | Size | $/1M in/out | Run |
|---|---|---|---|---|
| `llm_haiku55` | `claude-haiku-5-5` | small | 0.10 / 0.50 | full grid |
| `llm_gpt6luna` | `gpt-6-luna` | small | 0.10 / 0.50 | full grid |
| `llm_sonnet55` | `claude-sonnet-5-5` | mid | 2 / 10 | fixed 30% subsample |

- **Roles.** Haiku 5.5 and Luna are the reported models: main figures and core results. Sonnet 5.5 is a scale check, reported in the supplement (`llm_scale_check.png`). See "Method lineup" for the rule that would move it into the main results.
- Sonnet's 30% subsample is a fixed set of query records (one seed), so it is a subset of the Haiku and Luna runs, and all three models can be compared on it.
- **Any further runs on larger models use 10% (decided 2026-11-10).** Pass `--sample-frac 0.1` explicitly; do not change `SUBSAMPLE_FRAC` in `lib/llm_clients.py`, which defines the finished 30% runs and their `__sonnet30` comparison rows. Why: the 30% scale check (done 2026-11-10, $25.89) already settles the question. On every transformed condition except `ids_ciphered`, Sonnet is below Luna on the same queries (paired 95% CI excludes zero, gaps of 5–34 points; on `ids_ciphered` the two are level, corrected 2026-11-10 from `results/bootstrap_ci.csv`), and within about 3 points of Haiku wherever Haiku answers. At about 100 queries per condition the paired CI is roughly ±8 points, enough for gaps this size. A 10% run writes to `llm_sonnet55__sample0.1/`, which `05_cv_threshold.py`'s scale-check code does not read yet (it reads `llm_sonnet55/`), so extend `SCALE_REF` when such runs exist.
- All three models reason at effort `medium` (Claude: adaptive thinking, which cannot be turned off; Luna: `reasoning.effort`).
- No v1 model is rerun as a bridge to the TKDE submission. The v1 numbers are contaminated, and the paper goes to a new venue, so the bridge has no use. (Revised 2026-11-13: the previous generation of each small model is now in scope as a comparison of model generations, not as a bridge. See Part D.)
- Not in the lineup: an OpenAI mid-size model (`gpt-6.1-terra` at $1/$5, or `gpt-6.1-sol` at $2/$10). Add one only if a reviewer asks for a second mid-size model.
- Subscription pilots (`llm_haiku55_cli`, `llm_sonnet55_cli`) write to `pilot_results/` and are never reported.
- Test-set sizes stay as they are: 1,081 Abt-Buy and 1,113 Amazon-Google queries. The 95% CI is about ±2 points at F1 ≈ 0.9 on the full set and about ±3.5 points on the 30% subsample. Paired comparisons across conditions are tighter.

### C.2 Designs

1. **Main design: rank-then-select, ordered, k=50.** TF-IDF ranks every candidate for free, and the LLM picks the best of the top 50, listed in rank order. This is how a practical LLM matcher would be built. The shortlist is identical in every condition, because TF-IDF does not change under the ciphers.
2. **2x2 factorial (built 2026-11-06).** Descriptive words intact or ciphered, crossed with identifiers intact or ciphered. Cells: `original`, `desc_ciphered`, `ids_ciphered`, `all_ciphered`. Run with `--transformations factorial`.
   - Descriptive words become non-English nonce words, and identifiers become random codes of the same shape. Both maps are global and injective. The TF-IDF score matrices are identical across cells (to 1e-9).
   - The key contrast is `desc_ciphered` on Abt-Buy. There, 43% of true pairs share an identifier found in no other record. If the LLM used the preserved evidence, F1 would stay close to Original.
   - On Amazon-Google, identifiers cover only 4% of true pairs, so the factorial there mainly checks that the nonce cipher behaves like v1 Ciphered Words.
3. **Candidate-list robustness (R2's objection to 50 candidates).** Haiku and Luna, full grid, Original plus the three v1 ablations:

   | Run | Flags | What it rules out | Cost |
   |---|---|---|---|
   | ordered, k=50 | (default) | none; the realistic pipeline | in the main budget |
   | ordered, k=10 | `--top-k 10` | long-list difficulty | about $6 |

   - The k=10 run keeps the rank order, so it differs from the main run only in list length.
   - **Shuffled k=50 is cut (decided 2026-11-09).** The rank order hands the LLM a free lexical hint (TF-IDF's top pick), which can only work against our finding. The paper says this in one sentence. The instance analysis adds free evidence: how often the LLM misses when the true match is TF-IDF's top candidate. `--shuffle-candidates` stays in the code.
   - Report F1 both overall and only on queries whose true match made the shortlist, because k=10 has lower shortlist recall.
   - Optionally add the factorial `desc_ciphered` cell to this run.
4. **Not run: pairwise yes/no prompting.** It needs one call per candidate pair, so it multiplies token cost. It is also not the rank-then-select setting this paper studies. The paper states this as a limitation and leaves it to future work.

5. **Shared-token highlighting (added 2026-11-09).** A prompt variant (`highlight`, method suffix `_hl`) that marks, in each candidate, the tokens it shares with the query (e.g. `[w55]`). The text and shortlist are unchanged; only the marking is added. If F1 recovers under the ciphers, the model can use lexical evidence once it is surfaced but does not find it on its own, which is the claim ("fails to fall back"). It also answers R1 D1: the task is solvable when the overlap is visible. Compared with the token-fallback prompt (which tells the model to use shared tokens), it separates "does not look" from "cannot find". Haiku and Luna, both datasets, on `ciphered_letters`, `desc_ciphered`, `all_ciphered` (plus Original as a check that marking does not hurt). About $6 optimistic; pilot free on the CLI first.
6. **Dose-response: partial description cipher (added 2026-11-09).** Cipher a fraction p of descriptive word types (p = 0.25, 0.5, 0.75; p = 0 and 1 are `original` and `desc_ciphered`), identifiers intact, same nonce map and seed. Each map stays injective, so TF-IDF stays invariant (`--verify`). At p = 0.25 the text is still mostly readable, so a graded decline answers "garbled text breaks the model" better than a cliff would. Cells `desc_ciphered_p25`, `_p50`, `_p75`. Haiku and Luna on Abt-Buy (where identifiers matter), all non-LLM methods on both datasets. About $6 optimistic.

### C.3 Still to build (no API cost), in priority order

1. **Part A and Part B above.** Both done 2026-11-08: `README.qmd` renders with `quarto render README.qmd`, reads only CSV/JSON, and prints "not yet run" for missing results.
2. **`07_instance_analysis.py`.** It reads the v2 LLM parquets, which already log the true match's rank, the candidate ids, and the prompt size for each row. It writes `results/shortlist_recall.csv` and `results/instance_analysis.csv` with:
   - shortlist recall for each k, from TF-IDF alone;
   - F1 limited to queries whose true match made the shortlist;
   - the share of LLM misses where the true pair shares an identifier piece that no other candidate has (the headline stat for R2 D2);
   - errors against prompt token count (R2 D1c);
   - paired bootstrap CIs for the Original-to-ablation drop, per model and condition;
   - F1 with refused queries excluded, next to the standard F1 (where refused queries count as misses), per model and condition (item 5).
   - **Done 2026-11-10.** `scripts/07_instance_analysis.py` (no API calls, about 30 s) writes five tables to `results/`:
     - `instance_analysis.csv`: per LLM run and condition, status rates, shortlist recall, TF-IDF top-1 accuracy (the shortlist's first candidate, which agrees with the standalone TF-IDF method on 96–98% of queries), LLM accuracy, share of answers that pick candidate 1, the LLM-vs-TF-IDF correct/incorrect 2x2, CV F1 on all queries and on queries whose true match made the shortlist, the unique-shared-identifier stats, mean input tokens, and the Spearman correlation of tokens with a correct pick.
     - `f1_answered.csv`: CV F1 on the queries each model did not refuse, every main method scored on the same queries.
     - `shortlist_recall.csv`: recall at k = 1, 5, 10, 20, 50. Abt-Buy 0.69 / 0.90 / 0.95 / 0.98 / 0.99; Amazon-Google 0.81 / 0.96 / 0.99 / 0.99 / 0.99. F1 on in-shortlist queries is within 0.01 of overall F1.
     - `prompt_size.csv`: accuracy by input-token quintile, with bins pooled over a run's conditions.
     - `bootstrap_ci.csv`: paired bootstrap 95% CIs (2,000 resamples, seed 42) for the Original-to-condition drop, LLM minus TF-IDF top-1, and model-vs-model gaps; metrics are accuracy and F1 at a fixed F1-optimal threshold (reported F1 stays the CV F1 from `05`).
     - Identifier pieces are the factorial `id_map` keys, found in the original text the LLM sees. The CV helpers live in `scripts/lib/cv.py`, shared with `05` (its output is byte-identical).
     - Results are summarized under "Revised framing" at the top. Not built: a figure for these tables (C.3 item 6).
3. **`llm_status.csv` from `06_summarize_results.py`.** Status rates by model and condition, so parse failures are reported as their own outcome and never hidden as "no match". Refusals matter most: the pilot found safety-classifier blocks on about 20% of v1 Ciphered Words queries for Haiku (item 5). Since 2026-11-09 the CLI adapter logs these blocks as `refusal`, not `api_error`.
4. **Transform-seed parameter.** Add `--seed N` to `01_prepare_datasets.py`, writing `*_sN.csv` the way `02_prepare_factorial.py` already does. Run 3–5 seeds for the non-LLM methods (R2 D6).
5. **v1 Ciphered Words reproducibility. Decided 2026-11-09: keep the committed CSVs as canonical.** Do not regenerate them, and do not change the mapping code in `01_prepare_datasets.py`: sorting before mapping would give a different map, not the committed one. So the LLM runs on `ciphered_words` use the committed files, and nothing has to be rerun.
   - The paper describes the transform accurately: it also ciphers model codes, so Fig 1's "QRT-W55" example is wrong. It also says the map is fixed in the released CSVs rather than regenerated from code.
   - The factorial `all_ciphered` cell is the clean, reproducible version of the same manipulation and carries the identifier argument.
   - **Refusals (pilot finding).** Haiku 5.5 CLI pilot, 2026-11-09 (Abt-Buy, 5% sample, 54 queries per condition). On v1 Ciphered Words, 11 of 54 queries were blocked by Claude's safety classifier ("can't help with this … Details: `[bio]`"); 10 were blocked again on retry. The v1 cipher draws replacement words from macOS `/usr/share/dict/words` (Webster's 2nd, 1934), which is full of obscure chemistry and drug terms ("arsacid", "acetanilid", "arecaine"), and most chosen words start with "a". The factorial cells, which use nonce words, had no refusals.

     | Condition | ok / no_match / refusal | Correct top pick | Excluding refusals |
     |---|---|---|---|
     | Original | 51 / 3 / 0 | 0.907 | 0.907 |
     | v1 Ciphered Words | 22 / 22 / 10 | 0.370 | 0.455 |
     | `desc_ciphered` | 47 / 7 / 0 | 0.796 | 0.796 |
     | `all_ciphered` | 28 / 26 / 0 | 0.463 | 0.463 |
   - Decided 2026-11-09: do not change the cipher words. Choosing a vocabulary after seeing refusals would tune the transform to model behavior, and any real-word list still carries R2's injected-meaning problem. Run v1 Ciphered Words as it is, report its refusal rate and its F1 with and without refused queries, and decide after the paid runs whether it stays in the main results or moves to the appendix. `all_ciphered` reproduces the effect without refusals and carries the claim.
   - **Refusals at full scale (paid runs, 2026-11-10).** Through the API, Haiku 5.5 is refused on most ciphered queries, far more than the CLI pilot showed. The refusals come from the API itself (`stop_reason: refusal`, `category=bio`, some `general_harms`), not from the harness. GPT-6 Luna refused no queries in any run.

     | Haiku 5.5 refusal rate | Abt-Buy (name + description) | Amazon-Google (title only) |
     |---|---|---|
     | Original | 0% | 0% |
     | Ciphered Letters | 96% | 85% |
     | Scrambled | 97% | 4% |
     | Ciphered Words | 47% | 0% |
     | Factorial cells | 3.5% or less | 0.4% or less |

     - The prompt does not fix it. The token-fallback prompt raises refusals (Abt-Buy Ciphered Words: 47% to 87%). k=10 lowers some of them (Abt-Buy Ciphered Words 24%, Amazon-Google Ciphered Letters 60%). Repeat 1 matches the first run to within 2 points.
     - **Decided 2026-11-10: report the refusal rate as a finding.** Without semantic content, the model does not fall back on the lexical evidence: it declines the task. Report refusal rates for every model and condition (item 3) beside F1.
     - **Stop paying for Haiku where it refuses.** The `--repeat 1..2` runs skip Haiku on Abt-Buy `ciphered_letters,scrambled` and Amazon-Google `ciphered_letters`. The main grid, factorial, k=10, and token-fallback runs already include these conditions and stay as they are.
     - **Sonnet 5.5 pilot first (C.4 step 4b).** Sonnet was refused on 10 of 54 Abt-Buy Ciphered Words queries in the API pilot. If it refuses Ciphered Letters and Scrambled the way Haiku does, run Sonnet's full 30% grid only on the conditions it answers, and report the pilot refusal rate for the rest.
6. **`08_figures.R`.** Labels for the new methods and cells; split `f1_comparison.png` and `precision_recall_scatter.png` into main (semantic methods + TF-IDF reference) and appendix (`*_lexical.png`) figures (A.2); add a factorial figure and a robustness figure, both with the TF-IDF reference line.
   - Split done 2026-11-08. Main figures show the semantic methods (v1 LLMs labeled "(v1)" until v2 replaces them) with TF-IDF as a dashed red line (F1) or red diamonds (precision-recall). `llm_robustness.png` and `llm_prompt_comparison.png` also carry the TF-IDF line. The overview figures show the four v1 conditions only. **Factorial and robustness figures done 2026-11-10.** `factorial_2x2.png` is an interaction plot (x: identifiers intact or ciphered; one line per description state; Luna, Haiku, and the two embedding models by dataset; TF-IDF dashed). On Abt-Buy, ciphering descriptions with identifiers kept costs Luna little (0.966 to 0.919), ciphering identifiers costs more (0.824), and Haiku falls below TF-IDF with both ciphered (0.652 vs 0.697); on Amazon-Google the identifier factor barely moves any method. `candidate_list_robustness.png` compares ordered k=50 with k=10 for Haiku and Luna on Original and the three v1 ablations: Luna moves 0.03 F1 or less; Haiku is somewhat higher at k=10 where it collapses (Amazon-Google Ciphered Letters 0.17 to 0.37) but stays far below TF-IDF, so list length does not explain the failure. Both figures skip with a message when their results are missing. Not yet in `README.qmd`. Captions now write "+/-" ("±" rendered as ".." under Rscript).
   - Sonnet scale check done 2026-11-09. `05_cv_threshold.py` also scores Haiku, Luna, and TF-IDF on the query ids in Sonnet's parquet for each condition, as `*__sonnet30` rows. `08_figures.R` builds the main figures from the small v2 models (the v1 LLMs stand in until those exist) and draws `llm_scale_check.png` once Sonnet results exist. `README.qmd` has an "Appendix: scale check" section, and Sonnet is out of the main tables.

7. **Highlight prompt (C.2 item 5).** Add a `highlight` entry to `PROMPTS` in `lib/llm_runner.py` and register `llm_<model>_hl` methods. Shared tokens are computed per candidate with the same tokenizer TF-IDF uses; identical across conditions in structure because the maps are injective. Test offline, then a CLI pilot.
8. **Dose-response cells (C.2 item 6).** Add a fraction setting to `lib/factorial.py` / `02_prepare_factorial.py` that ciphers a seeded random subset of descriptive word types, writing `desc_ciphered_p25/_p50/_p75`. Add them to `FACTORIAL_CELLS` (or a `dose` group for `--transformations`), and to `05_cv_threshold.py`'s `TRANS`, the README, and `08_figures.R` (a dose-response figure with the TF-IDF line). `--verify` must pass for each.

9. **Decode test. Done 2026-11-10 ($0.49).** `scripts/09_decode_test.py` (paid) asks each model to recover the original text of 30 sampled records per dataset (Abt-Buy name, Amazon-Google title), 10 per call, without naming the transform. Score: share of the words the transform changed that reappear in the decoded line. Output cap 32k tokens (an 8k cap truncated Luna's cipher-breaking). Writes `results/decode_test.csv` and `results/decode_test_rows.csv` (with raw responses).

   | Recovered words | Luna | Haiku | Sonnet |
   |---|---|---|---|
   | Scrambled (Abt-Buy / Amazon-Google) | 0.77 / 0.99 | 0.68 / 0.90 | 0.65 / 0.92 |
   | Ciphered Letters | 0.08 / 0.34 | 0.26 / 0.03 (67% refused on A-G) | 0.90 / 0.34 (67% refused on A-G; 0.95 when answered) |
   | Ciphered Words (control) | 0.00 / 0.00 | 0.01 / 0.01 | 0.04 / 0.01 |

   - Luna's Ciphered Letters result varies by batch: it either decodes almost fully (0.93 on one Amazon-Google batch, 0.81 on an Abt-Buy batch in the first run) or answers "[unrecoverable from the altered text]". In matching, the prompt carries 51 records of ciphertext, more than this test gives.
   - **The committed Scrambled CSVs do not match `lib/transforms.py`.** The code keeps each word's first and last letters and leaves digits alone; the committed files shuffle every character, digits included ("pioneer" -> "oenirpe", "CDSB10" -> "1db0cs"). As with Ciphered Words (item 5), the committed CSVs are canonical, and the paper describes them, not the code.

Optional for this paper: an `--include-unmatched` flag for full catalogs (R2 D3), and a Batch API mode, which would roughly halve the cost of every pass.

### C.4 Run order (once keys and credit are in place)

Script names below are the post-reorg names.

Budget (offline estimate from `03_estimate_llm_cost.py`, 2026-11-09; the conservative figure is about 2.3× the optimistic one): about $97 optimistic and $223 conservative for steps 2–9 (step 9 needs C.3 items 7–8 built first).

| Step | Runs | Optimistic |
|---|---|---|
| 5 | Haiku + Luna full grid; Sonnet 30%; 4 v1 conditions | $31 (Haiku $4, Luna $4, Sonnet $23) |
| 6 | all three models, 3 factorial cells | $24 (Sonnet $18) |
| 7 | Haiku + Luna, k=10 | $6 |
| 8 | Haiku + Luna, token-fallback prompt; `--repeat 1..2` | $8 + $15 |
| 9 | Haiku + Luna, highlight prompt; dose-response cells | $6 + $6 |

1. `cp .env.example .env` and fill in `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. Then run `uv sync` and the tests.
2. **Rerun `openai_embeddings`** (paid, but only a few cents: embeddings for about 2,200 records per condition) on both datasets with `--transformations all,factorial`, so every semantic method comes from the current code. The free non-LLM methods were rerun on 2026-11-08 (the committed parquets did not reproduce from the current code). Then rerun `05_cv_threshold.py`.
3. **Subscription pilot (free).** `llm_haiku55_cli` on Abt-Buy `original,ciphered_words` with `--sample-frac 0.05 --workers 2`. Check the CLI output parsing and the isolation, and iterate on the prompt here if needed.
   - Done 2026-11-09, also on `desc_ciphered,all_ciphered`. Parsing works (no `parse_fail`), and the true match was in the shortlist for every query. v1 Ciphered Words drew safety-classifier refusals (C.3 item 5). Each CLI call carries about 1,050 tokens beyond the prompt (fit across 98 calls), probably the CLI's own scaffolding plus the JSON schema; confirm with a free token count once the API key is in. No prompt changes were needed.
4. **API pilot (about $1).** `llm_haiku55,llm_gpt6luna,llm_sonnet55` on Abt-Buy `original,ciphered_words` with `--sample-frac 0.05`. Check the status counts, whether the API refuses the same Ciphered Words queries as the CLI (and whether Luna refuses at all), and Luna's reasoning-token use. Then recalibrate `03_estimate_llm_cost.py` with `--cpt-*` and `--out-tokens` from the logged tokens.
   - Done 2026-11-09 ($0.66). No `parse_fail` or `truncated`. The API refused Haiku on 27 of 54 Ciphered Words queries (the CLI: 10), Sonnet on 10, Luna on none. Per query: Haiku about $0.0003, Luna $0.0002 (Original) to $0.0006 (ciphered, about 850 reasoning tokens), Sonnet $0.0054. The recalibrated total for steps 5–8 was about $63, under half the earlier estimate. Pilot outputs moved to `pilot_results/`.
4b. **Sonnet pilot on Ciphered Letters and Scrambled (about $1).** `llm_sonnet55` on both datasets, `--transformations ciphered_letters,scrambled --sample-frac 0.05`. Added 2026-11-10 after Haiku's full-scale refusals (C.3 item 5). Run it before Sonnet's full runs, and drop any condition Sonnet mostly refuses.
   - Done 2026-11-10 ($1.56, 54–56 queries per cell; outputs in `pilot_results/*/llm_sonnet55__sample0.05_cl_sc/`). Sonnet refuses far less than Haiku, so no condition is dropped:

     | Sonnet 5.5 pilot | Refused | Correct top pick | Excluding refusals | Haiku refused (full run) |
     |---|---|---|---|---|
     | Abt-Buy Ciphered Letters | 24% | 0.611 | 0.805 | 96% |
     | Abt-Buy Scrambled | 28% | 0.574 | 0.795 | 97% |
     | Amazon-Google Ciphered Letters | 30% | 0.554 | 0.795 | 85% |
     | Amazon-Google Scrambled | 4% | 0.857 | 0.889 | 4% |
5. **Main grid.** Run `llm_haiku55,llm_gpt6luna` on the full grid first, then `llm_sonnet55` on its 30% subsample. About $31–70.
   - Haiku and Luna done 2026-11-10 (about $6). Sonnet held for step 4b.
6. **Factorial.** Run all three models with `--transformations factorial`. Original resumes from `checkpoints/` with no new calls, so only the three new cells cost money: about $24–55.
7. **Robustness.** `--top-k 10` (rank order kept) on Haiku and Luna: about $6–14.
8. **Prompt and variance checks.** Run the token-fallback `_tf` variants and `--repeat 1..2` on Haiku and Luna only: about $8–18 per extra pass.
   - Steps 6 (Haiku + Luna), 7, and the `_tf` runs done 2026-11-10. In the repeats, Haiku skips the conditions it refuses (C.3 item 5), so `llm_haiku55__rep1/__rep2` have no Abt-Buy `ciphered_letters,scrambled` and no Amazon-Google `ciphered_letters` (Abt-Buy rep 1 has all four, run before the cut).
9. **Mechanism checks (C.2 items 5–6), once built.** Highlight prompt (`llm_haiku55_hl,llm_gpt6luna_hl`) on Original and the three cipher conditions; dose-response cells for Haiku and Luna on Abt-Buy. About $12–28. CLI pilot both first.
10. **Outputs.** After each batch:
   - run `05_cv_threshold.py`, `06_summarize_results.py`, `07_instance_analysis.py`, and `08_figures.R`;
   - then `quarto render README.qmd`.

---

## Part D: Third dataset, more models, and a supervised baseline (added 2026-11-13; redesigned 2026-11-20)

**Redesign (decided 2026-11-20, John).** The full-grid runs of the older models (about $160) are replaced by a smaller suite across more models, which costs less and gives more breadth:

1. **Fixed 30% query subsample for every new model run:** seed 42, the same queries as Sonnet 5.5's existing runs (Abt-Buy 324, Amazon-Google 334; WDC 300; 958 in all). Every model can then be compared on one set of queries, and Sonnet needs no reruns on the first two datasets. A 20% draw would not be a subset of the 30% one (`rng.choice` does not nest), which is why the sample is not smaller. Precision: at about 320 queries per dataset, the paired 95% CI for a model minus TF-IDF's top-1 is about ±4.5 points (±2.5 pooled over the three datasets), from the 16% discordance on the full Abt-Buy runs. That is enough for the effects the new models test (refusal, collapse, above/at/below TF-IDF; 10–80 points). The narrow claim (Luna 6 points below TF-IDF's top-1 on Abt-Buy) stays on the full grids of Haiku 5.5 and GPT-6 Luna.
2. **Four conditions for the new models:** `original`, `ciphered_letters` (readable), `ciphered_words` and `all_ciphered` (meaning removed). 958 × 4 = 3,832 calls per model.
3. **Models.** Haiku 5.5 and GPT-6 Luna keep their full grids and add WDC (D.5c). New on the 30% suite: Haiku 4.5 and GPT-5.6 Luna (previous generation), Kimi K3 (open weights, back in the lineup; pilot first, D.4b), and Sonnet 5.5 on WDC.
4. **A Ditto-style supervised baseline** (D.8; free, no API).

The 2026-11-13 design below is kept for the record. Where it conflicts with the redesign (full grids for the older models, the D.5 budget, "out of scope: Sonnet"), the redesign wins.

**Decided 2026-11-13 (John).** Three additions, all at small-model scale:

1. **Add WDC Products as a third dataset.** The "LLM below TF-IDF's top-1" result rests on Abt-Buy alone; Amazon-Google is a tie. WDC Products is a second identifier-heavy product dataset, and its corner cases (near-identical variants: same card at 32GB and 64GB) are separated only by digits and codes, so even the Original condition tests whether the model reads them.
2. **Add the previous generation of each small model:** Claude Haiku 4.5 and GPT-5.6 Luna, full grid, all three datasets. This answers a practical question: what does one model generation buy on low-meaning text? The decode test showed that current models read the "readable" ciphers; the older models may not, which would explain the v1 collapse without leaning on the contaminated v1 numbers.
3. **Check the WDC setup with the free methods before any paid run** (D.3).

Framing note (the argument lives in the paper plan): newer models are more robust on the *readable* conditions because they recover meaning. That is not evidence that they use lexical evidence once meaning is gone, and Haiku 5.5's refusals show that "newer" is not uniformly "more reliable".

No public parts or SKU catalog benchmark exists (searched 2026-11-13: electronic components, spare parts, industrial catalogs). OpenSanctions Pairs (names, ID numbers, transliterations; pairwise, CC-BY-NC) is the lower-meaning alternative, kept for later because it is a different domain and task shape.

### D.1 WDC Products: source and setup

- **Source.** Peeters & Bizer, "WDC Products" (EDBT 2024), http://webdatacommons.org/largescaleproductcorpus/wdc-products/. File: `https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip` (2.4 MB, sha256 `5385e2491e82c7f6c208d8eac7ec777a4a44180769f3a339c99803ff2b068981`, downloaded 2026-11-13).
- **Variant: 80% corner cases, multi-class, 0% unseen products.** The test set is `wdcproductsmulti80cc20rnd000un_gs.json.gz` (1,000 offers, 500 products); the catalog comes from `wdcproductsmulti80cc20rnd000un_train_large.json.gz` (2,841 offers, the same 500 products). Offers come from many shops and are mostly PC parts, memory cards, lenses, printers, and watches; some titles are in Spanish, German, or French.
- **Task shape, matching our harness.**
  - Queries (left side, `offers`): the 1,000 test offers.
  - Catalog (right side, `catalog`): one offer per product from `train_large`, the one with the lowest `id` (500 records). Deterministic and blind to text.
  - Mapping: each query to the catalog record with its `cluster_id`. Every query has exactly one true match, and each catalog record matches two queries (many-to-one; Amazon-Google already has one-to-many pairs). Check that `04`, `05`, and `07` score many-to-one correctly before reporting.
- **Text: title only** (`title_only=True`, as for Amazon-Google). Descriptions are missing on 28% of test offers, are long and partly HTML or non-English, and would add cost. Titles carry the codes. CSV columns: `id`, `title`, `description`, `manufacturer` (from `brand`), `price`; the transforms cover all text columns, as for the other datasets.
- **Measured 2026-11-13** (scratch check, title only, scikit-learn TF-IDF rather than the repo's; recheck with `04`): TF-IDF top-1 0.624; recall at 10 0.982, at 50 0.995. Share of title tokens that are letter-digit codes: WDC 11.5%, Abt-Buy 12.7%, Amazon-Google 3.3%. Share of true pairs whose titles share such a code: WDC 45%, Abt-Buy 45%, Amazon-Google 11.5%. So WDC is not lower in meaning than Abt-Buy; it is a second Abt-Buy-like dataset with harder near-duplicates.
- **License: open question for John.** The dataset page states no license; the arXiv paper is CC BY-NC-ND 4.0, which may or may not be meant for the data. Default until it is settled: gitignore `data/wdc-products/` except a README, and regenerate every file from the download with a deterministic script (D.2). Results parquets carry ids and model output, not record text, so they are committed as usual. Ask the WDC team (Bizer group) whether transformed CSVs can be redistributed before the repo goes public.

### D.2 WDC transforms

- **New script `scripts/01b_prepare_wdc.py`.** Downloads the zip (or reuses a cached copy), checks the sha256, writes `data/wdc-products/data_original/{offers,catalog}.csv` and `wdc_mapping.csv`, then writes the three v1 transforms to `data_test/` with the functions in `lib/transforms.py`. It never calls `process_dataset` for another dataset.
- **Add `wdc-products` to `DATASETS`** in `lib/datasets.py` (`left_name: offers`, `right_name: catalog`, `title_only: True`). Guard `01_prepare_datasets.py` so it refuses `wdc-products` and no longer includes it in `--dataset all` (WDC is built by `01b`; the "never rerun 01" rule is unchanged for the other two).
- **Deterministic Ciphered Words map.** The v1 map depends on set iteration order (`PYTHONHASHSEED`). Add an opt-in `sort_words=True` path to `extract_unique_words` / `generate_ciphered_words`; the default stays as is, so nothing about the committed CSVs changes. `01b` runs the transforms in the v1 order (letters, words, scrambled) in a fresh process, so the global `random.seed(42)` gives the same letter cipher every run. Test: two runs under different `PYTHONHASHSEED` values give byte-identical CSVs.
- **Scrambled follows `lib/transforms.py`** (first and last letters kept, words of 3 letters or fewer kept, digit runs kept). Found 2026-11-13 while checking this: the committed Amazon-Google Scrambled CSVs match that code exactly, but the committed Abt-Buy Scrambled CSVs shuffle every character, short words and multi-digit numbers included (C.3 item 9 noted the Abt-Buy behavior but not that Amazon-Google differs). WDC uses the code, as Amazon-Google does. The paper has to describe the two existing datasets separately (paper plan).
- **Factorial:** `uv run scripts/02_prepare_factorial.py --dataset wdc-products --verify`. Use the dataset name, not `all`, which reorders the other datasets' manifests. Default `--max-id-df 20`.
- **Hard-coded dataset lists to extend:** `lib/cv.py` (`CFG`), `06_summarize_results.py`, `09_decode_test.py` (`DATASETS`, `MODEL_KEYS`), `08_figures.R` (labels and panel layout), `README.qmd`.
- **Status, 2026-11-15 (commit point 2 built, not yet committed).**
  - `scripts/01b_prepare_wdc.py`, the `DATASETS` entry, `sort_words`, the `01` guard (refuses `wdc-products`, skipped by `--dataset all`), the `.gitignore` entry, and `data/wdc-products/README.md` are in. The build gives 1,000 offers, 500 catalog records, and a 1,000-row mapping; the zip sha256 matches. `data_test/wdc_build.json` records the zip and `/usr/share/dict/words` hashes (the Ciphered Words map also depends on that dictionary).
  - `tests/test_wdc_determinism.py` (3 tests): builds under `PYTHONHASHSEED` 1 and 98765 give byte-identical CSVs (9 files); the default `sort_words=False` path still differs across seeds, which is why the committed CSVs cannot be regenerated.
  - **Factorial bug found and fixed.** WDC titles contain the piece `2_1` (no letters, not pure digits). `classify_pieces` made it a descriptive piece, `_nonce` mapped it to itself, and `build_map` looped forever (12 CPU-minutes before it was killed). `is_kept` in `lib/factorial.py` now also keeps letterless pieces. Abt-Buy and Amazon-Google classes are identical before and after (kept sets 500 and 1,069), so their CSVs, maps, and manifests do not change. New test `test_letterless_piece_is_kept_not_looped` in `tests/test_factorial.py`.
  - `02_prepare_factorial.py --dataset wdc-products --verify` passed: 13,238 pieces (identifier 2,460; descriptive 10,075; kept 703), none lengthened. 49.1% of true pairs share an identifier piece, 8.5% share one unique to the pair. TF-IDF scores are invariant to 1e-9 across the three cells; v1 TF-IDF F1 is 0.6557 on all four cells.

### D.3 Free checks on WDC (before any paid run)

- `uv run scripts/04_run_experiments.py --dataset wdc-products --methods tfidf,soft_tfidf,levenshtein,jaro_winkler,monge_elkan,sentence_transformer --transformations all,factorial`, then `05`.
- Pass conditions: 1,000 queries and 500 catalog records load; the TF-IDF invariance check in `02 --verify` passes; TF-IDF F1 is flat across the seven conditions (within about 0.01); TF-IDF recall at 50 is at least 0.98; a few transformed titles read as expected by eye.
- Then `openai_embeddings` on WDC (paid, a few cents).
- **Status, 2026-11-17: free checks pass (commit point 3 built, not yet committed; `openai_embeddings` not run yet).**
  - `04` ran the six free methods on all seven conditions (42 parquets under `results/wdc-products/`); `05` and `06` regenerated `cv_results.csv` and `summary.csv`. `lib/cv.py` `CFG` and `06` (record-count check) now include WDC. Every Abt-Buy and Amazon-Google row in both tables is identical to before; `llm_status.csv` is unchanged.
  - Many-to-one scoring checked in code: every method scores each query on its own argmax over the catalog, and `cv.true_and_score` tests (query, prediction) pairs against the mapping, so two queries sharing a catalog record are scored correctly. `07` reads only LLM parquets; recheck when WDC LLM runs exist.
  - Pass conditions: 1,000 queries and 500 catalog records load; `02 --verify` passed; TF-IDF CV F1 is 0.653 on six conditions and 0.654 on Ciphered Words (flat); TF-IDF shortlist recall (computed offline with `llm_runner.build_requests`, k=50) is 0.655 / 0.953 / 0.986 / 0.992 / 0.996 at k = 1 / 5 / 10 / 20 / 50, the same on every condition except Ciphered Words (0.652 at k=1). The repo's TF-IDF top-1 is 0.655, not the 0.624 of the D.1 scratch check.
  - CV F1 on WDC, Original / all_ciphered: TF-IDF 0.653 / 0.653, Soft TF-IDF 0.667 / 0.659, Sentence-Transformer 0.521 / 0.484, Monge-Elkan 0.479 / 0.463, Jaro-Winkler 0.392 / 0.382, Levenshtein 0.351 / 0.384. Sentence-Transformer is below TF-IDF even on Original, and loses far less to ablation than on Abt-Buy (0.583 to 0.300).
  - **Found: the v1 Ciphered Words map is not injective.** Where a word length has more unique words than the dictionary supplies (1,500 per length), `create_word_mappings` fills in random consonant-vowel words drawn with replacement, so distinct words can share a cipher word ("hub" and "6tb" both become "tat"). Target words with two or more sources: WDC 286, committed Abt-Buy 121, committed Amazon-Google 4,055. That is why Ciphered Words is the one condition where TF-IDF moves (Abt-Buy 0.697 to 0.700, Amazon-Google 0.808 to 0.807, WDC 0.653 to 0.654). The factorial cells are injective by construction. Per C.3 item 5 nothing is regenerated; the paper should not say v1 Ciphered Words preserves token identity exactly.
  - **`openai_embeddings` on WDC done 2026-11-17** (about 150k–300k tokens, under $0.01). CV F1: Original 0.653 (equal to TF-IDF), ids_ciphered 0.628, ciphered_letters 0.524, scrambled 0.533, desc_ciphered 0.505, all_ciphered 0.498, ciphered_words 0.496. Unlike Sentence-Transformer, it drops well below TF-IDF once meaning is removed, as on Abt-Buy. Abt-Buy and Amazon-Google rows in `cv_results.csv` and `summary.csv` are unchanged.

### D.4 Previous-generation models

Checked 2026-11-13 against the Anthropic and OpenAI deprecation and pricing pages:

| Method | Model | $/1M in/out | Status |
|---|---|---|---|
| `llm_haiku45` | `claude-haiku-4-5-20251001` | 1.00 / 5.00 | Active, but past its "not sooner than Oct 15, 2026" retirement floor; Anthropic can retire it with 60 days' notice. **Run it first.** |
| `llm_gpt56luna` | `gpt-5.6-luna` | 0.20 / 1.20 | Active. Its place as GPT-6 Luna's predecessor is inferred from the name; confirm on the models endpoint during the pilot. |
| optional: `llm_gpt4omini` | `gpt-4o-mini-2024-07-18` | 0.15 / 0.60 | Shuts down Jan 14, 2027. Two generations back, and the TKDE model. **John to decide**; not in the default budget. |

Not used: `gpt-5-mini` and `gpt-5-nano` (shut down Dec 11, 2026).

- **Harness changes (`lib/llm_clients.py`).**
  - Haiku 4.5 rejects `output_config.effort` and has no adaptive thinking; thinking is `thinking: {type: "enabled", budget_tokens: N}` (N at least 1024 and below `max_tokens`). Add a `thinking_budget` field to `ModelConfig`; `_call_anthropic` sends `thinking` when it is set and sends no `effort`.
  - Structured output (`output_config.format`, JSON schema) on Haiku 4.5: confirm in the pilot. If the API rejects it, send the schema in the prompt only and rely on `parse_match_json`; `parse_fail` rates are reported either way.
  - Temperature is not sent (unchanged; the 2026-10-21 API version rejects it).
  - GPT-5.6 Luna uses the existing OpenAI path with `reasoning.effort = medium`, like GPT-6 Luna.
  - Offline tests for both configs with the fake clients in `tests/test_llm_harness.py`.
  - **Status (2026-11-17, commit point 4; no API call made).** `llm_haiku45` (`claude-haiku-4-5-20251001`, $1 / $5, `thinking_budget=1024`, `effort=None`) and `llm_gpt56luna` (`gpt-5.6-luna`, $0.20 / $1.20, effort `medium`) are in `MODELS`, so `llm_<key>` and `llm_<key>_tf` exist. Checked with the claude-api skill (not from memory): Haiku 4.5 takes `thinking: {type: "enabled", budget_tokens: N}` (N at least 1024, below `max_tokens`; max_tokens is 8,000 here), has no `effort` parameter, and supports structured outputs (`output_config.format`) together with extended thinking, so the schema is sent as for the other models. The skill table lists $1 / $5 and a 200K context window; the dated id matches `models.md`. `call_model` raises `FatalAPIError` if `--effort` is passed for a model with a thinking budget (the API would 400 every call). Temperature, `fallbacks` and `effort` are not sent. 4 new offline tests (19 in `test_llm_harness.py`, 26 in all with the factorial and WDC tests): request shape and cost for both models, statuses, effort guard, registry and full-grid default. `03_estimate_llm_cost.py --models haiku45,gpt56luna` runs offline with the pre-pilot defaults ($19.41 Haiku 4.5, $4.39 Luna per 4 conditions on Abt-Buy); recalibrate after the pilot. Still open for the pilot: the `gpt-5.6-luna` id on the models endpoint, and whether Haiku 4.5 returns clean JSON under thinking.
- **API pilot done (2026-11-17, John approved; $0.71 total).** `llm_haiku45,llm_gpt56luna` on Abt-Buy `original,ciphered_words`, `--sample-frac 0.05` (54 queries per condition, 216 calls, blocking recall 0.981, k=50). Outputs were moved to `pilot_results/api_pilot_d4/abt-buy/` (gitignored, never read by 05/06), because 04 had written them under `results/abt-buy/*__sample0.05/`, which 05/06 would scan.
  - **Id and JSON checks passed.** `gpt-5.6-luna` accepted and served as `gpt-5.6-luna`. Haiku 4.5 served as `claude-haiku-4-5-20251001`, structured output works under thinking, stop reasons all `end_turn`/`completed`, no `parse_fail`, `truncated`, `refusal` or `api_error` in any of the 216 rows.
  - **Statuses.** Haiku 4.5: original 48 ok / 6 no_match, ciphered_words 19 ok / 35 no_match. Luna 5.6: original 51 / 3, ciphered_words 48 / 6. (Sample F1 at the best threshold: Haiku 4.5 0.922 / 0.466; Luna 5.6 0.941 / 0.646; 54 queries, so not for interpretation.) Haiku 4.5 never refused, where Haiku 5.5 refused 47% of Ciphered Words queries on the full Abt-Buy run.
  - **Tokens per call (mean; median/p90 in brackets).** Haiku 4.5 output: original 731 (663/1,092), ciphered_words 848 (872/1,031); input 1,747 / 1,893, which is 0.81 / 0.86 of Haiku 5.5's 2,166 / 2,209 on the same conditions. Haiku 4.5 reasons far more than Haiku 5.5 (mean output 137 / 245): `budget_tokens=1024` is the minimum, and the model uses most of it on every call. Luna 5.6 output: original 94 (60/168), ciphered_words 882 (559/2,130), close to GPT-6 Luna's 96 / 763.
  - **Cost per call:** Haiku 4.5 $0.0054 / $0.0061; Luna 5.6 $0.0004 / $0.0014.
  - **Estimates revised from the pilot (all 22,358 calls per model).** D.5a Haiku 4.5: input about 51M Haiku 5.5 tokens (AB+AG logs 35.4M plus WDC 15.7M) times 0.83 is 42M tokens, $42; output 700 to 1,000 tokens per call is 16 to 22M tokens, $78 to $112; total about $120–155, central $135 (was $80). D.5b Luna 5.6: GPT-6 Luna's logged tokens on AB+AG plus the WDC estimate give 33M in and 8M out, about $16.5 (range $16–19; was $13). Total with D.5c and decode tests: about $160, not $100. The output side is the risk, and it cannot be lowered without dropping thinking (budget 1,024 is the API minimum). Alternative for John: run Haiku 4.5 without thinking (about $55), at the cost of reasoning parity with the current models.
- **Reasoning parity (decided in this plan; John can change it).** The current models run at effort `medium`. Their logged output tokens per answered call (thinking included): Haiku 5.5 median 285, p90 645; GPT-6 Luna median 220, p90 1,021. Haiku 4.5 gets extended thinking with `budget_tokens = 1024`, the API minimum and close to Haiku 5.5's p90. The paper reports each model "at the setting closest to medium effort".
- **API pilot (about $1).** `llm_haiku45,llm_gpt56luna` on Abt-Buy `original,ciphered_words`, `--sample-frac 0.05`. Check status counts, refusals, parse failures, and tokens; then recalibrate `03_estimate_llm_cost.py`.
- **Decode test for every small model.** The generation story depends on it: can the older models read Ciphered Letters and Scrambled? Add `haiku45` and `gpt56luna` to `09_decode_test.py`, and WDC to its datasets. Run the new models on all three datasets and the current small models on WDC (about $1–2).
- **Redesign (2026-11-20):** `haiku45` and `gpt56luna` get `default_sample_frac=0.3`, as Sonnet has, so their results land in `results/{ds}/llm_haiku45/` with no `__sample` suffix. `test_previous_generation_method_registry` checks a full-grid default and must change with it. Add `kimik3` to `09_decode_test.py` too.
  - **Done 2026-11-21 (commit point 7)** for the defaults and the test (see D.4b status). Not done yet: `09_decode_test.py` still has only the three current models and two datasets, and its `call()` has no Haiku 4.5 thinking path or Kimi path. That goes in before the decode step, with its own estimate.

### D.4b Kimi K3 (open weights; added 2026-11-20)

- **Route.** OpenRouter, pinned to the Moonshot AI provider with fallbacks off, so every call is served by the same stack. The `openai_compatible` adapter, its provider pinning, and its tests are still in `lib/llm_clients.py`; only the registry entry was removed on 2026-11-09. Restore it from commit `309a1a9^`: `kimik3`, model `moonshotai/kimi-k3`, `$3 / $15` (2026-11-09 price), `max_tokens 16000`, size "open", `default_sample_frac=0.3`, key from `KIMI_API_KEY`, base URL from `KIMI_BASE_URL` (default `https://openrouter.ai/api/v1`), model override `KIMI_MODEL`, structured output via `KIMI_RESPONSE_FORMAT` (default `json_schema`). For Moonshot's own API instead: `KIMI_BASE_URL=https://api.moonshot.ai/v1`, `KIMI_MODEL=kimi-k3`.
- **Before the pilot, verify on OpenRouter** (not from memory): the model id, current price, whether the Moonshot provider supports `json_schema` structured output (else `json_object`), and how reasoning is controlled. Effort stays at the provider default unless a "medium" setting exists; record what was used.
- **Key (John).** Add `KIMI_API_KEY=...` (an OpenRouter key with credit) to `.env`. `.env.example` is missing from the repo, although `CLAUDE.md` mentions it; recreate it with the three key names and no values.
- **Pilot (about $1–2, needs approval):** `llm_kimik3` on Abt-Buy `original,ciphered_words`, `--sample-frac 0.05` (108 calls). Check the served model and provider, the statuses (parse failures, refusals, truncation at 16,000 tokens), and the input and output tokens per call. Move the outputs to `pilot_results/` afterwards, as in D.4.
- **Route changed 2026-11-21 (John): Moonshot's own API, not OpenRouter.** Same $3 / $15 list price, so it saves OpenRouter's 5.5% credit fee; Moonshot also has a batch API at 60% of list price (not used; revisit if the pilot puts the suite above about $50). John's key is from platform.kimi.ai, with $20 of credit.
- **Status (2026-11-21, commit point 7; no API call made).** Checked against platform.kimi.ai/docs (K3 quickstart, chat API reference, response_format, reasoning effort, errors), not from memory:
  - Model `kimi-k3`, base URL `https://api.moonshot.ai/v1`. Moonshot now calls K3 its "previous-generation flagship"; the current one is `kimi-k3.1-pro`. Kept K3, as planned; **John to decide** whether to switch (check that K3.1 Pro is open weights and its price first).
  - `json_schema` with `strict: true` is documented as supported for `kimi-k3`, so the schema is sent as for the other models. The docs warn that content can come back in a Markdown fence; `parse_match_json` already handles that (tested).
  - K3 always thinks and cannot turn it off. Effort is a top-level `reasoning_effort` with levels `low`, `high`, `max` (default `max`). There is no `medium`. **Set to `low`**: `max` is the costliest, and the parity rule in D.4 compares tokens, so the pilot checks Kimi's output tokens at `low` against Haiku 5.5 and GPT-6 Luna at `medium` (median 285 and 220, p90 645 and 1,021). John can switch to `high`.
  - `max_tokens` is deprecated for `max_completion_tokens` (default 131,072, which also counts toward the TPM rate limit); capped at 16,000. Temperature and top_p are fixed on K3 and not sent.
  - Insufficient balance comes back as HTTP 429 `exceeded_current_quota_error`; it is fatal (the run stops). An ordinary 429 `rate_limit_reached_error` is a row-level `api_error` after the SDK's retries.
  - Prices: input $3.00 (cache miss) from Moonshot's context-caching page; output $15.00 from the Moonshot AI row on OpenRouter (Moonshot's own price table did not render). Cache hits ($0.30) are billed at the full input price in `_cost`, so logged costs are an upper bound.
  - Code: `kimik3` back in `MODELS` (model, endpoint, and key name fixed in code, not read from `.env`, so `llm_kimik3` always means this model). New `ModelConfig` fields `effort_param`, `max_tokens_param`, `allowed_efforts`; `call_model` fails before any request if the effort is not `low`/`high`/`max` (so `--effort medium` cannot 400 every row, and `--effort ""` cannot silently fall back to `max`). `haiku45` and `gpt56luna` now default to the 30% subsample. `.env.example` recreated with the three key names. `KIMI_BASE_URL` and `KIMI_MODEL` in `.env` are no longer read.
  - Tests: 21 in `test_llm_harness.py` (28 in all), up from 19. New: Kimi request shape (top-level `reasoning_effort`, `max_completion_tokens`, strict schema, no temperature/thinking/extra_body, fenced JSON), effort guard, fatal vs row-level 429. `test_previous_generation_method_registry` now checks that Haiku 4.5, GPT-5.6 Luna and Kimi K3 get the same 30% query ids as Sonnet.
  - Offline estimate (`03`, its default 600 output tokens per call): $0.013 per call, so the pilot (108 calls) about $1.40 and the 3,832-call suite about $50. Output tokens at `low` are the unknown.
- **API pilot done (2026-11-21, John approved; $1.14 total).** `llm_kimik3` on Abt-Buy `original,ciphered_words`, `--sample-frac 0.05` (54 queries per condition, 108 calls, blocking recall 0.981, k=50, effort `low`, 8 workers). Outputs moved to `pilot_results/api_pilot_d4b/abt-buy/` (gitignored, never read by 05/06), as in D.4.
  - **Checks passed.** Served as `kimi-k3` on every row; all 108 stop reasons `stop`; no `parse_fail`, `truncated`, `refusal` or `api_error`; every row needed one attempt. Strict `json_schema` works (no Markdown fences came back).
  - **Statuses.** Original 51 ok / 3 no_match; Ciphered Words 29 ok / 25 no_match. Kimi never refused, where Haiku 5.5 refused 47% of Ciphered Words queries on the full Abt-Buy run. (Sample F1 at the best threshold: 0.932 / 0.578; 54 queries, not for interpretation.)
  - **Tokens per call (mean; median/p90 in brackets).** Output: original 130 (116/180), Ciphered Words 625 (490/1,137), max 4,076, far below the 16,000 cap. Input 1,540 / 1,741. **Reasoning parity holds at `low`:** on the same conditions the three `medium` models produce Haiku 5.5 137 / 245, GPT-6 Luna 96 / 763, GPT-5.6 Luna 94 / 882 (range 94–137 / 245–882); Haiku 4.5, on its 1,024-token thinking budget, 731 / 848. Keep `low`.
  - **Cost per call:** $0.0066 original, $0.0146 Ciphered Words. Latency mean 7 s / 20 s.
  - **Suite estimate (3,832 calls, `03` with the pilot's output tokens).** Original about $4.60 at 130 output tokens, plus about 1.5× for input, because `03` underestimates Kimi's original-text input (1,019 estimated vs 1,540 logged per call on Abt-Buy; ablated text is close, 1,636 vs 1,741), so about $6.50. The three ablated conditions are $40 at the mean of 625 output tokens, $62 if every call reasoned at the p90 of 1,137. Central about $48, range about $45–65. Ciphered Letters is the unknown: a model that tries to decode it may reason longer.
- **Go or no-go rule.** Price the 3,832-call suite from the pilot. If it is above about $60, choose (John): Kimi on Abt-Buy and WDC only (the identifier-heavy datasets), a lower reasoning setting, or a cheaper open-weights model on the same route (check current options and prices first).

### D.5 Runs and budget

**Redesigned budget (2026-11-20).** Per-call costs from logged runs and the D.4 pilot: Haiku 4.5 about $0.0058, GPT-5.6 Luna about $0.0009, Sonnet 5.5 about $0.0056 (its 30% runs). 3,832 calls per new model (958 queries × 4 conditions). Every paid step still needs John's approval, with its estimate, before it runs.

| Step | Runs | Estimate |
|---|---|---|
| D.4b | `llm_kimik3` pilot, Abt-Buy, 5%, 2 conditions | about $1–2 (**done 2026-11-21: $1.14**) |
| D.5a | `llm_haiku45`, 30% suite, all three datasets | about $22 (**done 2026-11-21: $24.38**) |
| D.5b | `llm_gpt56luna`, 30% suite, all three datasets | about $3.50 (**done 2026-11-21: $2.88**) |
| D.5c | `llm_haiku55`, `llm_gpt6luna` full grid on WDC (unchanged) | about $4.50–4.80 (**done 2026-11-21: $5.48**) |
| D.5d | `llm_sonnet55` on WDC, 30%, 4 conditions | about $7 (**done 2026-11-21: $7.19**) |
| D.5e | `llm_kimik3`, 30% suite, all three datasets (after the pilot's go/no-go) | about $50–80 before the pilot; **about $48 (range $45–65) from the pilot** |
| decode | `haiku45`, `gpt56luna`, `kimik3` on three datasets; current small models on WDC | about $2–3 |

Total about $40 without Kimi, about $90–120 with it; the Kimi range is a guess until the pilot. **Revised 2026-11-21 after the Kimi pilot: about $90 with Kimi (range $85–105).** Batch API would halve the Anthropic and OpenAI steps but needs new harness code; not planned.

**Status 2026-11-21 (John approved D.5a–d; credit added on both providers).**
- **D.5a done: `llm_haiku45`, $24.38** (estimate $23–25). 3,832 calls, 958 queries × 4 conditions. Query ids equal Sonnet's on Abt-Buy and Amazon-Google, and equal GPT-5.6 Luna's on all three datasets (WDC 300). No refusals, parse failures or API errors. **One `truncated` row** (Amazon-Google Ciphered Letters, `b0007prnjo`): it hit the 8,000-token cap with no text; the 99th percentile of output is 2,653 tokens. It is kept as model behavior (counts as a missed match), so the paper's "no final run contains ... truncations" sentence needs a qualifier. Mean output 687–1,068 tokens per call. No-match answers are frequent under ablation (Abt-Buy Ciphered Words 225 of 324; WDC 189 of 300), where Haiku 5.5 refused instead.
- **D.5b done: `llm_gpt56luna`, $2.88** (estimate $3.50–4.50). 3,832 calls; ids equal Sonnet's (Abt-Buy, Amazon-Google) and Haiku 4.5's (all three). Statuses only `ok` and `no_match`. Served id `gpt-5.6-luna` confirmed in the D.4 pilot.
- Sample F1 at the best threshold, printed by `04` (not cross-validated, not for interpretation), Original / Ciphered Letters / Ciphered Words / all_ciphered: Haiku 4.5 Abt-Buy 0.929 / 0.812 / 0.397 / 0.543, Amazon-Google 0.882 / 0.715 / 0.721 / 0.654, WDC 0.885 / 0.651 / 0.448 / 0.532; GPT-5.6 Luna Abt-Buy 0.954 / 0.910 / 0.686 / 0.755, Amazon-Google 0.892 / 0.832 / 0.817 / 0.821, WDC 0.947 / 0.807 / 0.633 / 0.746.
- **D.5c done.** `llm_gpt6luna` WDC, 1,000 queries × 7 conditions: **$2.53** (estimate about $2.00, 27% over; Ciphered Letters $0.54 and all_ciphered $0.45 reasoned longer than the Amazon-Google logs implied). Statuses only `ok` and `no_match`. `llm_haiku55` WDC, same grid: **$2.95** (estimate $2.55–2.80). **Refusals on WDC:** Ciphered Letters 738 of 1,000, Ciphered Words 86, desc_ciphered 24, all_ciphered 21, Scrambled 8, none on Original or ids_ciphered. No parse failures, truncations or API errors.
- **D.5d done: `llm_sonnet55` WDC, 300 × 4: $7.19** (estimate $6.50–7). Ids equal Haiku 4.5's and GPT-5.6 Luna's; the 300 are a subset of both full WDC runs. 8 refusals on Ciphered Letters, none elsewhere.
- **D.5a–d total: $39.93** (Anthropic $34.52, OpenAI $5.41), against about $38–40 estimated.
- Sample F1 at the best threshold (not cross-validated), Original / C. Letters / C. Words / all_ciphered: GPT-6 Luna WDC 0.933 / 0.845 / 0.676 / 0.770 (Scrambled 0.906, desc 0.846, ids 0.888); Haiku 5.5 WDC 0.903 / 0.314 / 0.442 / 0.601 (Scrambled 0.866, desc 0.724, ids 0.830); Sonnet WDC 0.913 / 0.750 / 0.526 / 0.587.
- **05, 06, 07 rerun 2026-11-21** (John approved). Only rows were added to `cv_results.csv`, `summary.csv`, `llm_status.csv`, `instance_analysis.csv`, `f1_answered.csv`, `prompt_size.csv`, `shortlist_recall.csv`; every old estimate is unchanged. `bootstrap_ci.csv`: estimates unchanged, but CI bounds moved by up to 0.006 (median 0.001) because all comparisons draw from one RNG stream, so adding runs shifts the draws; no CI changed sign. The paper's GPT-6 Luna Abt-Buy Ciphered Words CI goes from [−8.7, −3.7] to [−8.6, −3.8]. **To do:** seed each comparison from its key so CIs stop depending on which runs exist.
- **Shortlists differ between 30% and full runs.** `build_requests` (`lib/llm_runner.py`) draws the subsample before fitting TF-IDF, so IDF is fit on 30% of the queries. The 30% runs (Sonnet, Haiku 4.5, GPT-5.6 Luna) share identical shortlists with each other (checked, 100%), and the full runs share theirs, but across the two groups only 8% (Abt-Buy) to 75% (WDC) of queries get the same 50 candidates. Recall@50 is identical (0.988 / 0.994 / 0.990); TF-IDF top-1 differs by at most 1.3 points (Abt-Buy 0.685 vs 0.698). Every within-run comparison (LLM vs its own shortlist's top-1) is exact. Comparisons between a 30% model and a full-run model (Appendix scale check, `__sonnet30` rows) carry this small shortlist difference and the paper must say so. **Code not changed,** so Kimi K3 (D.5e) gets the same shortlists as the other 30% models.
- **Breadth results (accuracy, shared 30% queries, each model vs TF-IDF top-1 on its own shortlist; `scratchpad/breadth_shared30.csv`, to be moved into 07):** see the paper plan for the interpretation. Ciphered Words gap (pts): Abt-Buy GPT-6 Luna −6.2, GPT-5.6 Luna −5.6, Haiku 5.5 −52.2, Haiku 4.5 −44.1, Sonnet −35.5; Amazon-Google −0.3, −2.1, −22.2, −19.8, −18.6; WDC +4.0, −0.3, −36.3, −30.7, −21.3. Original gap: +20 to +32 on Abt-Buy and WDC and +4 to +8 on Amazon-Google, for every model.
- **Next:** D.5e (Kimi, needs a Moonshot top-up), decode tests, D.8; add the answered-subset comparison to `07`.

Commands (once approved), e.g. `uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku45 --transformations original,ciphered_letters,ciphered_words,all_ciphered` (the 30% default applies), per dataset. After each run, check that its query ids equal Sonnet's on Abt-Buy and Amazon-Google.

Order: Kimi key and registry entry, then the D.4b pilot; D.5a first among paid runs (Haiku 4.5 retirement risk), then D.5b, D.5c, D.5d, D.5e if go, then the decode tests. D.8 (Ditto-style) runs in parallel, as it is free.

**Superseded 2026-11-20** (kept for the record): the full-grid plan below.

Estimates use the current models' median tokens per call times list price (Abt-Buy, Amazon-Google, WDC: 3,194 queries × 7 conditions = 22,358 calls per model); recalibrate after the pilot.

| Step | Runs | Estimate |
|---|---|---|
| D.3 | non-LLM methods on WDC; `openai_embeddings` | free; cents |
| D.4 | pilot; decode tests | about $1; $1–2 |
| D.5a | `llm_haiku45` full grid (`all,factorial`), all three datasets | about $135 (revised 2026-11-17 from pilot; was $80) |
| D.5b | `llm_gpt56luna` full grid, all three datasets | about $17 (revised; was $13) |
| D.5c | `llm_haiku55`, `llm_gpt6luna` full grid on WDC | about $3 each |
| optional | `llm_gpt4omini` full grid, all three datasets | about $8 |

Total about $160 without the optional run (revised 2026-11-17 after the pilot; planned $100).

- **D.5c re-estimated 2026-11-17 (offline).** WDC prompts built with `build_requests` (k=50, 1,000 queries per condition), priced with each model's own Amazon-Google logs per condition (chars per input token, mean output tokens per call; Amazon-Google is the other title-only dataset). Haiku 5.5: 15.7M input tokens, about $2.55 for 7 conditions ($2.80 if it answers every ciphered-letters query instead of refusing 85%). GPT-6 Luna: 10.0M input tokens, about $2.00. Both together about $4.50–4.80. Prices from `lib/llm_clients.py` ($0.10 / $0.50 per 1M for both); recheck before the run.

- **Paused 2026-11-17 (John):** D.5a, D.5b and the decode tests are on hold because the revised cost (about $160) is too high. No paid run is approved. Options on the table: Haiku 4.5 on a fixed 30% subsample (about $40, no new code); Batch API support for both providers (half price, new harness code); Haiku 4.5 without thinking (about $55); drop Haiku 4.5 and keep Luna 5.6 (about $17). Prompt caching does not apply: the shared prefix is about 20 tokens and prompts are below Haiku 4.5's 4,096-token minimum. D.5c (Haiku 5.5 and GPT-6 Luna on WDC, about $4.50–4.80) is independent of this decision and also not yet approved.
- **Superseded 2026-11-20 by the redesign** (top of Part D and the redesigned budget at the top of D.5): 30% suite, Kimi pilot first, Ditto-style baseline (John).
- Order: D.3, then the D.4 pilot, then D.5a (retirement risk), D.5b, D.5c, then the decode tests.
- Run every condition once for Haiku 4.5, including the ones Haiku 5.5 refuses; the refusal rate across generations is part of the comparison.
- Out of scope for WDC and the older models: k=10, the token-fallback prompt, repeats. (Sonnet on WDC is back in scope: D.5d.)

### D.6 Outputs

- `05`, `06`, `07` with WDC and the new methods (the unique-shared-identifier statistic in `07` now applies to two datasets). `07` compares models on their shared queries; for the 30% suite that is Sonnet's query set.
- `08_figures.R`: a third dataset panel in the main figures. A new `model_breadth.png` replaces the planned `model_generations.png`: per condition and dataset, every LLM on the shared 30% queries, each model next to its predecessor, TF-IDF reference line, with refusal rates marked. Main text or appendix is decided once results are in. The Ditto-style results get their own panel or table (D.8).
- `README.qmd`: WDC in the data and results sections; a model-breadth section; the Ditto-style baseline.
- Copy changed figures to the paper's manuscript `source/` folder (`pvldb/1/source/` until `dke/1/` exists).

### D.7 Commit points

1. This plan (both plan files).
2. `01b_prepare_wdc.py`, the `DATASETS` entry, the `sort_words` option, the `01` guard, `.gitignore`, determinism test; the factorial cells and `--verify` output.
3. Non-LLM results on WDC and the regenerated tables (D.3).
4. Harness support for `haiku45` and `gpt56luna`, with offline tests.
5. After the pilot, then after each paid batch (D.5a, D.5b, D.5c, decode tests).

Added 2026-11-20 for the redesign:

6. This redesign (both plan files).
7. Harness: `kimik3` back in `MODELS`, `default_sample_frac=0.3` for `haiku45` and `gpt56luna`, `.env.example`, updated offline tests (free).
8. After the Kimi pilot (D.4b), then after each paid batch (D.5a–e, decode tests).
9. The Ditto-style script, its offline tests, and its results (D.8).

### D.8 Ditto-style supervised baseline (added 2026-11-20; free, no API)

Why: R2 D5 asked for Ditto or DeepMatcher, and Ditto was the strongest model in Moslemi et al. (DKE 2026), the paper we build on. It also tests whether the failure to fall back belongs to zero-shot LLM matching or to neural matchers in general: a matcher trained on ablated data can learn the shared-token evidence directly.

- **Model.** A Ditto-style cross-encoder with Hugging Face `transformers`: records serialized as in Ditto (`COL title VAL ... COL description VAL ...`, same text fields as the other methods via `title_only`), RoBERTa-base fine-tuned for binary match classification. No Ditto optimizations (domain-knowledge injection, TF-IDF summarization, augmentation); the paper says so and calls it "Ditto-style". Max sequence length 256. DeepMatcher is not run.
- **Task shape, the same as the LLMs.** 5-fold cross-validation over queries (fixed seed). Training pairs: each training query's true match plus 9 hard negatives drawn from its TF-IDF top-50 shortlist (seeded). 10% of the training queries are held out to choose the epoch. Test: score all 50 shortlist candidates of each held-out query and predict the argmax; its probability is the score that `05` thresholds. Every query gets a prediction from the fold that held it out.
- **Two regimes.**
  - (A) Train on Original, test on all seven conditions: a matcher with priors learned from meaningful text, like the LLMs. 5 fine-tunes per dataset. Method `ditto_style`.
  - (B) Train and test on the same ablated condition, `ciphered_words` and `all_ciphered`: the matcher can learn token identity. 10 fine-tunes per dataset. Method `ditto_style_matched`; its `original` row is regime A's.
  - 45 fine-tunes in all, one training seed each. Time one fold on the Mac (MPS) first; if the full set is too slow, use DistilRoBERTa or a GPU.
- **Prediction to test.** If (B) recovers to TF-IDF's level or above while (A) falls, the LLMs' failure is a property of zero-shot matching on meaning, not of neural matchers.
- **Code.** New script `scripts/04b_ditto_style.py` (as `01b`, a numbered sibling) writing `results/{ds}/ditto_style{,_matched}/{transformation}.parquet` with the usual columns, so `05`–`08` pick it up. Offline tests for fold assignment, negative sampling, and serialization (not training). `sentence-transformers` already pulls in `torch` and `transformers`; confirm no new dependency is needed.
- **Lineup.** Ditto-style is a supervised reference, not an object of study, and not a lexical method. It gets its own table or panel, not a line in the LLM figures.

## Definition of done

- Every script in `scripts/` is numbered, or lives in `scripts/lib/`. Tests are in `tests/`.
- Figures are only in `figs/`, and `results/plots/` is gone.
- `README.md` is rendered from `README.qmd`, and contains no hand-typed result numbers.
- The A.5 checks pass on the reorganized repo.
- The v1 LLM results are separated from v2, and labeled as superseded wherever they appear.
