# Plan: revision experiments and repo cleanup

Written 2026-11-08 for the revision of "Matching Without Meaning" after the TKDE reject (TKDE-2026-06-2289). The full revision plan, including the manuscript edits and venue strategy, is in the paper folder (`llm-fuzzy-joins/1/plan.md`). This file covers only the changes to this repo: how it is organized, how the README is built, and what we will run.

**Two plans, kept in sync.** The paper plan is `/Users/jhelvy/sync/00_current/llm-fuzzy-joins/1/plan.md` (argument, manuscript edits, venue). When John says "update the plan", update both files in the same sitting, and keep shared decisions consistent: lineup, figures, run order, and status. For script names and commands, this file is the source of truth.

Branch: `harness-v2`. No API calls have been made yet. Do not run any paid API call until keys and credit are confirmed.

**Commits: John makes every commit.** Claude does not run `git commit` (or push). At each natural commit point, Claude stops, summarizes what changed and how it was checked, proposes a commit message, and waits. John reviews the diff and commits. Wherever this plan says "commit", read it as "stop and flag a commit point".

## The claim the experiments must support

LLM entity matchers lean on semantics. When meaning is removed but the lexical evidence that identifies a match is kept, they do not fall back on that evidence, even though that evidence alone solves the task. TF-IDF is the control that proves the evidence is still there: it does not change under any of the transforms.

This is a proof of concept, not a model survey. Keep the lineup small and spend the effort on the controls.

### Method lineup (decided 2026-11-08)

- **Semantic methods (the object of study):** the LLMs (C.1) and the two embedding models (`sentence_transformer`, `openai_embeddings`). These are the methods whose sensitivity to semantics the paper measures.
- **One lexical control in the main text: TF-IDF.** It is invariant by construction, and it also produces the LLM's shortlist. It appears as a reference in every main figure (a dashed or differently colored line in F1 plots, a colored point in precision-recall plots). It is not presented as a competitor, and the text does not say lexical methods "overtake" the LLM.
- **Other lexical methods go to the appendix:** Soft TF-IDF, Levenshtein, Jaro-Winkler, Monge-Elkan. Their only role is to show that lexical methods are robust to the transforms, which is true by design. Still run them (they cost nothing), so the appendix and the README can show them, and the main text points to the appendix once.

## Order of work

1. **Reorganize the repo (Part A).** Do this first, before any API spend. Paths, checkpoint locations, and result directory names change in this step. Checkpoints written under the old layout would not resume under the new one, and none exist yet.
2. **Build the README from `README.qmd` (Part B).**
3. **Finish the zero-cost build items (Part C.3).**
4. **Run the experiments in the order in Part C.4.**
5. **Re-render the README after each batch of results.**

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
- `08_figures.R` also needs labels for the new methods (`llm_haiku55`, `llm_sonnet55`, `llm_gpt4omini`, `llm_kimik3`) and for the factorial cells.
- **Split the two overview figures into a main figure and an appendix figure** (see "Method lineup" above). Each one is currently a stacked cowplot with semantic methods on top and lexical methods below; the halves become separate files.

  | File | Goes in | Rows / points | TF-IDF shown as |
  |---|---|---|---|
  | `f1_comparison.png` | main text | semantic methods only (LLMs, embeddings) | a dashed, differently colored vertical reference line at TF-IDF's F1 in each dataset panel, labeled "TF-IDF" |
  | `f1_comparison_lexical.png` | appendix | lexical methods (TF-IDF, Soft TF-IDF, Monge-Elkan, Levenshtein, Jaro-Winkler) | an ordinary row |
  | `precision_recall_scatter.png` | main text | semantic methods only | one distinctly colored point per dataset and condition, labeled "TF-IDF" |
  | `precision_recall_scatter_lexical.png` | appendix | lexical methods | ordinary points |

  - TF-IDF is invariant to within about 0.01 F1, so one reference line per dataset (its Original F1) is enough in the F1 figure. If a factorial or seed run ever moves it more than that, draw it per condition instead.
  - Keep the main figures' file names, so the paper's `\includegraphics` paths still work. The legend or caption names the TF-IDF reference as the lexical control.
  - The other figures (`llm_robustness.png`, `llm_prompt_comparison.png`, and the new factorial and robustness figures) also get the TF-IDF reference line, so every main figure uses the same convention.

### A.3 Results layout

- `results/{dataset}/{method}/{transformation}.parquet` stays as it is. Non-default flags already add a suffix to the method directory (`llm_haiku55__shuffle0`, `llm_haiku55__shuffle0__k10`).
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
| 2x2 factorial table (F1 per cell, per model) | `results/cv_results.csv` |
| Candidate-list robustness (ordered k=50, shuffled k=50, shuffled k=10) | `results/cv_results.csv` (method suffixes `__shuffle0`, `__shuffle0__k10`) |
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

IDs and prices were verified 2026-11-06. Re-check them before running.

| Method | Model | $/1M in/out | Run |
|---|---|---|---|
| `llm_haiku55` | `claude-haiku-5-5` | 0.10 / 0.50 | full grid |
| `llm_sonnet55` | `claude-sonnet-5-5` | 2 / 10 | fixed 30% subsample |
| `llm_gpt4omini` | `gpt-4o-mini` | 0.15 / 0.60 | full grid; bridge to the TKDE version |
| `llm_kimik3` | Kimi K3 (`moonshotai/kimi-k3` on OpenRouter, pinned to Moonshot AI, fallbacks off) | 3 / 15 | fixed 30% subsample |

- Sonnet and Kimi share a seed, so they see the same 30% of query records. Those records are also in the Haiku and GPT-4o-mini runs, so all four models can be compared on the common subset.
- Claude models run at effort `medium` with adaptive thinking. GPT-4o-mini does not reason. Kimi K3 runs at its provider's default.
- Subscription pilots (`llm_haiku55_cli`, `llm_sonnet55_cli`) write to `pilot_results/` and are never reported.
- Test-set sizes stay as they are: 1,081 Abt-Buy and 1,113 Amazon-Google queries. The 95% CI is about ±2 points at F1 ≈ 0.9 on the full set and about ±3.5 points on the 30% subsample. Paired comparisons across conditions are tighter.

### C.2 Designs

1. **Main design: rank-then-select, ordered, k=50.** TF-IDF ranks every candidate for free, and the LLM picks the best of the top 50, listed in rank order. This is how a practical LLM matcher would be built. The shortlist is identical in every condition, because TF-IDF does not change under the ciphers.
2. **2x2 factorial (built 2026-11-06).** Descriptive words intact or ciphered, crossed with identifiers intact or ciphered. Cells: `original`, `desc_ciphered`, `ids_ciphered`, `all_ciphered`. Run with `--transformations factorial`.
   - Descriptive words become non-English nonce words, and identifiers become random codes of the same shape. Both maps are global and injective. The TF-IDF score matrices are identical across cells (to 1e-9).
   - The key contrast is `desc_ciphered` on Abt-Buy. There, 43% of true pairs share an identifier found in no other record. If the LLM used the preserved evidence, F1 would stay close to Original.
   - On Amazon-Google, identifiers cover only 4% of true pairs, so the factorial there mainly checks that the nonce cipher behaves like v1 Ciphered Words.
3. **Candidate-list robustness (R2's objection to 50 candidates).** Haiku and GPT-4o-mini, full grid, Original plus the three v1 ablations:

   | Run | Flags | What it rules out | Cost |
   |---|---|---|---|
   | ordered, k=50 | (default) | none; the realistic pipeline | in the main budget |
   | shuffled, k=50 | `--shuffle-candidates` | the rank-order hint | about $6.50 |
   | shuffled, k=10 | `--shuffle-candidates --top-k 10` | long-list difficulty | about $4.50 |

   - Report F1 both overall and only on queries whose true match made the shortlist, because k=10 has lower shortlist recall.
   - Optionally add the factorial `desc_ciphered` cell to these runs.
4. **Not run: pairwise yes/no prompting.** It needs one call per candidate pair, so it multiplies token cost. It is also not the rank-then-select setting this paper studies. The paper states this as a limitation and leaves it to future work.

### C.3 Still to build (no API cost), in priority order

1. **Part A and Part B above.**
2. **`07_instance_analysis.py`.** It reads the v2 LLM parquets, which already log the true match's rank, the candidate ids, and the prompt size for each row. It writes `results/shortlist_recall.csv` and `results/instance_analysis.csv` with:
   - shortlist recall for each k, from TF-IDF alone;
   - F1 limited to queries whose true match made the shortlist;
   - the share of LLM misses where the true pair shares an identifier piece that no other candidate has (the headline stat for R2 D2);
   - errors against prompt token count (R2 D1c);
   - paired bootstrap CIs for the Original-to-ablation drop, per model and condition.
3. **`llm_status.csv` from `06_summarize_results.py`.** Status rates by model and condition, so parse failures are reported as their own outcome and never hidden as "no match".
4. **Transform-seed parameter.** Add `--seed N` to `01_prepare_datasets.py`, writing `*_sN.csv` the way `02_prepare_factorial.py` already does. Run 3–5 seeds for the non-LLM methods (R2 D6).
5. **v1 Ciphered Words reproducibility.** Sort before mapping, so the map no longer depends on `PYTHONHASHSEED`. Decide between two options:
   - keep the committed CSVs as canonical and describe the transform accurately (it also ciphers model codes, so Fig 1's "QRT-W55" example is wrong);
   - regenerate the CSVs, and then rerun every non-LLM method for this condition.

   The factorial `all_ciphered` cell is the cleaner replacement either way.
6. **`08_figures.R`.** Labels for the new methods and cells; split `f1_comparison.png` and `precision_recall_scatter.png` into main (semantic methods + TF-IDF reference) and appendix (`*_lexical.png`) figures (A.2); add a factorial figure and a robustness figure, both with the TF-IDF reference line.

Optional for this paper: an `--include-unmatched` flag for full catalogs (R2 D3), and a Batch API mode, which would save roughly $15–30 per Claude pass.

### C.4 Run order (once keys and credit are in place)

Script names below are the post-reorg names.

1. `cp .env.example .env` and fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `KIMI_API_KEY`. Then run `uv sync` and the tests.
2. **Rerun `openai_embeddings`** (paid, but only a few cents: embeddings for about 2,200 records per condition) on both datasets with `--transformations all,factorial`, so every semantic method comes from the current code. The free non-LLM methods were rerun on 2026-11-08 (the committed parquets did not reproduce from the current code). Then rerun `05_cv_threshold.py`.
3. **Subscription pilot (free).** `llm_haiku55_cli` on Abt-Buy `original,ciphered_words` with `--sample-frac 0.05 --workers 2`. Check the CLI output parsing and the isolation, and iterate on the prompt here if needed.
4. **API pilot (about $1).** `llm_haiku55,llm_gpt4omini,llm_kimik3` on Abt-Buy `original` with `--sample-frac 0.05`. Check the status counts and that Kimi accepts `json_schema`. Then recalibrate `03_estimate_llm_cost.py` with `--cpt-*` and `--out-tokens` from the logged tokens.
5. **Main grid.** Run `llm_haiku55,llm_gpt4omini` on the full grid first: does the v1 GPT-4o-mini collapse survive the fixed harness? Then run `llm_sonnet55,llm_kimik3` on their 30% subsample. Budget: about $65–150 per pass.
6. **Factorial.** Run all four models with `--transformations factorial`. Original resumes from `checkpoints/` with no new calls, so only the three new cells cost money: about $50.
7. **Robustness.** Run `--shuffle-candidates`, then `--shuffle-candidates --top-k 10`, both on Haiku and GPT-4o-mini: about $11 together.
8. **Prompt and variance checks.** Run the token-fallback `_tf` variants and `--repeat 1..2` on Haiku and GPT-4o-mini only, at about $7–14 per extra pass.
9. **Outputs.** After each batch:
   - run `05_cv_threshold.py`, `06_summarize_results.py`, `07_instance_analysis.py`, and `08_figures.R`;
   - then `quarto render README.qmd`.

## Definition of done

- Every script in `scripts/` is numbered, or lives in `scripts/lib/`. Tests are in `tests/`.
- Figures are only in `figs/`, and `results/plots/` is gone.
- `README.md` is rendered from `README.qmd`, and contains no hand-typed result numbers.
- The A.5 checks pass on the reorganized repo.
- The v1 LLM results are separated from v2, and labeled as superseded wherever they appear.
