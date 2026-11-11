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
- **The v1 deficits were partly a model effect.** The weaker v1 models could not decode the ciphers; Luna can. Say so, rather than presenting the v1 drops as general.
- **Practical message.** With meaningful product text, an LLM reranking a TF-IDF shortlist is very strong, and model choice matters (refusals above all). The limitation bites on low-meaning text: SKUs, part codes, abbreviations, opaque IDs. There, the LLM adds cost but no accuracy over TF-IDF, can override correct lexical evidence, and some models decline the task.
- **Still open:** whether Ciphered Letters and Scrambled stay in the main figures (grouped apart as "meaning recoverable") or move to the appendix.

This is a proof of concept, not a model survey. Keep the lineup small and spend the effort on the controls.

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
- No v1 model is rerun as a bridge to the TKDE submission. The v1 numbers are contaminated, and the paper goes to a new venue, so the bridge has no use.
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

## Definition of done

- Every script in `scripts/` is numbered, or lives in `scripts/lib/`. Tests are in `tests/`.
- Figures are only in `figs/`, and `results/plots/` is gone.
- `README.md` is rendered from `README.qmd`, and contains no hand-typed result numbers.
- The A.5 checks pass on the reorganized repo.
- The v1 LLM results are separated from v2, and labeled as superseded wherever they appear.
