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
    python scripts/llm_cost_estimate.py
    python scripts/llm_cost_estimate.py --datasets abt-buy --models haiku55,luna
    python scripts/llm_cost_estimate.py --out-tokens 800 --subsample-frac 0.2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from llm_clients import MODELS                       # noqa: E402
from llm_runner import build_requests                # noqa: E402
from run_experiments import TRANSFORMATIONS, load_dataset   # noqa: E402

# Defaults are optimistic heuristics. v1 billing for Haiku 4.5 on Abt-Buy implies
# roughly 2-2.5x more input tokens than chars/4, so pass lower values (e.g.
# --cpt-original 1.8 --cpt-transformed 1.2) for a conservative budget, or
# calibrate both flags from the input_tokens column of a small pilot run.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="abt-buy,amazon-google")
    ap.add_argument("--transformations", default=",".join(TRANSFORMATIONS))
    ap.add_argument("--models", default=",".join(k for k in MODELS if k != "openweight"))
    ap.add_argument("--prompt", default="standard", choices=["standard", "token_fallback"])
    ap.add_argument("--out-tokens", type=int, default=600,
                    help="Assumed output tokens per call incl. thinking (reasoning models)")
    ap.add_argument("--cpt-original", type=float, default=4.0,
                    help="Chars per token for Original text")
    ap.add_argument("--cpt-transformed", type=float, default=2.5,
                    help="Chars per token for ciphered/scrambled text")
    ap.add_argument("--subsample-frac", type=float, default=0.2,
                    help="Fraction used for models with full_run=False (Opus, Astra)")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        for t in args.transformations.split(","):
            df_l, df_r, df_m = load_dataset(ds, t)
            reqs, top_k = build_requests(df_l, df_r, df_m, args.prompt)
            chars = sum(r["prompt_chars"] for r in reqs)
            cpt = args.cpt_original if t == "original" else args.cpt_transformed
            in_tok = chars / cpt
            for mk in args.models.split(","):
                cfg = MODELS[mk]
                frac = 1.0 if cfg.full_run else args.subsample_frac
                out_per_call = args.out_tokens if cfg.effort else 150
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
