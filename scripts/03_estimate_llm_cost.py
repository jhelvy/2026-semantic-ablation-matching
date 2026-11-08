"""
Estimate harness-v2 LLM run costs WITHOUT calling any API.

Builds every prompt exactly as llm_runner would (same blocking, top-k,
truncation, prompt template), counts characters, converts to tokens with a
chars-per-token ratio, and prices each model in llm_clients.MODELS.

Token estimates are approximate: ciphered/scrambled text splits into more
tokens than English, so a lower --chars-per-token is used for those
conditions. Output tokens (thinking + JSON answer) are an assumption; check
them against the first real run and re-estimate.

Usage:
    uv run scripts/03_estimate_llm_cost.py
    uv run scripts/03_estimate_llm_cost.py --datasets abt-buy --models haiku55,gpt4omini
    uv run scripts/03_estimate_llm_cost.py --out-tokens 1500 --cpt-original 1.8 --cpt-transformed 1.2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.datasets import DATASETS, TRANSFORMATIONS, load_dataset   # noqa: E402
from lib.llm_clients import MODELS                       # noqa: E402
from lib.llm_runner import build_requests                # noqa: E402
from lib.methods import set_title_only                   # noqa: E402

# Defaults are optimistic heuristics. v1 billing for Haiku 4.5 on Abt-Buy implies
# roughly 2-2.5x more input tokens than chars/4, so pass lower values (e.g.
# --cpt-original 1.8 --cpt-transformed 1.2) for a conservative budget, or
# calibrate both flags from the input_tokens column of a small pilot run.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="abt-buy,amazon-google")
    ap.add_argument("--transformations", default=",".join(TRANSFORMATIONS))
    ap.add_argument("--models", default=",".join(k for k, c in MODELS.items() if c.provider != "claude_cli"))
    ap.add_argument("--prompt", default="standard", choices=["standard", "token_fallback"])
    ap.add_argument("--top-k", type=int, default=None,
                    help="Candidates per query (default: the runner's adaptive top-k, 50 here)")
    ap.add_argument("--out-tokens", type=int, default=600,
                    help="Assumed output tokens per call incl. thinking (reasoning models)")
    ap.add_argument("--cpt-original", type=float, default=4.0,
                    help="Chars per token for Original text")
    ap.add_argument("--cpt-transformed", type=float, default=2.5,
                    help="Chars per token for ciphered/scrambled text")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        set_title_only(DATASETS[ds]["title_only"])
        for t in args.transformations.split(","):
            df_l, df_r, df_m = load_dataset(ds, t)
            reqs, top_k = build_requests(df_l, df_r, df_m, args.prompt, top_k=args.top_k)
            chars = sum(r["prompt_chars"] for r in reqs)
            cpt = args.cpt_original if t == "original" else args.cpt_transformed
            in_tok = chars / cpt
            for mk in args.models.split(","):
                cfg = MODELS[mk]
                frac = cfg.default_sample_frac or 1.0
                # gpt-4o-mini is the only non-reasoning model; Kimi K3 reasons by default
                out_per_call = 150 if cfg.key == "gpt4omini" else args.out_tokens
                n = len(reqs) * frac
                cost = (in_tok * frac) / 1e6 * cfg.input_price + n * out_per_call / 1e6 * cfg.output_price
                rows.append(dict(dataset=ds, transformation=t, model=mk, queries=int(n),
                                 top_k=top_k, input_Mtok=round(in_tok * frac / 1e6, 2),
                                 output_Mtok=round(n * out_per_call / 1e6, 2), cost_usd=round(cost, 2)))
            print(f"  built {ds}/{t}: {len(reqs)} prompts, {chars/1e6:.1f}M chars", file=sys.stderr)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    tot = df.groupby("model", sort=False)[["input_Mtok", "output_Mtok", "cost_usd"]].sum()
    print("\nTotal per model (one pass over the selected datasets x transformations):")
    print(tot.round(2).to_string())
    print(f"\nGrand total: ${df['cost_usd'].sum():.2f}  "
          f"(Batch APIs would roughly halve this; not implemented in harness v2 yet)")


if __name__ == "__main__":
    main()
