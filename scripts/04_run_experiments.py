"""
AUTOMATED ENTITY MATCHING EXPERIMENT RUNNER - CORRECTED
=======================================================
Runs methods with threshold optimization and proper output schema.

Usage:
    uv run scripts/04_run_experiments.py --dataset abt-buy
    uv run scripts/04_run_experiments.py --dataset abt-buy --methods jaro_winkler,tfidf

Harness-v2 LLM methods (see scripts/lib/llm_runner.py; `uv run scripts/03_estimate_llm_cost.py`
prices a run without calling any API):
    uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku55,llm_gpt6luna
    uv run scripts/04_run_experiments.py --dataset all --methods llm_sonnet55   # 30% subsample by default
    uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku55_cli --sample-frac 0.05 --workers 2   # subscription pilot -> pilot_results/
    uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_sonnet55 --effort high
    uv run scripts/04_run_experiments.py --dataset abt-buy --methods llm_haiku55 --repeat 2

Non-default options are appended to the result directory name, e.g.
results/abt-buy/llm_haiku55__effort-high/original.parquet, so runs never overwrite each other.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.datasets import DATASETS, TRANSFORMATION_GROUPS, load_dataset  # noqa: E402
from lib.llm_runner import LLM_METHODS  # noqa: E402
from lib.methods import METHODS as _V1_METHODS, set_title_only  # noqa: E402
from lib.paths import RESULT_ROOTS  # noqa: E402

METHODS = {**_V1_METHODS, **LLM_METHODS}

def run_label(method_name, llm_options):
    """Result-directory name: method plus any non-default harness-v2 options."""
    o = llm_options or {}
    parts = [method_name]
    if o.get("effort") is not None:
        parts.append(f"effort-{o['effort']}")
    if o.get("sample_frac"):
        parts.append(f"sample{o['sample_frac']}")
    if o.get("shuffle_candidates"):
        parts.append(f"shuffle{o.get('shuffle_seed', 0)}")
    if o.get("top_k"):
        parts.append(f"k{o['top_k']}")
    if o.get("repeat"):
        parts.append(f"rep{o['repeat']}")
    return "__".join(parts)


def run_single_experiment(dataset_name, transformation, method_name, method_func, llm_options=None):
    """Run single method on single transformation"""
    
    print(f"\n{'='*80}")
    print(f"Running: {method_name} on {dataset_name} ({transformation})")
    print(f"{'='*80}")
    
    try:
        # Load data
        set_title_only(DATASETS[dataset_name]['title_only'])
        print(f"Loading data... (title_only={DATASETS[dataset_name]['title_only']})")
        df_left, df_right, df_mapping = load_dataset(dataset_name, transformation)
        print(f"  Left: {len(df_left)} records")
        print(f"  Right: {len(df_right)} records")
        print(f"  Ground truth: {len(df_mapping)} matches")
        
        # Run method (includes threshold optimization)
        import inspect
        accepts_llm_options = 'llm_options' in inspect.signature(method_func).parameters
        label = run_label(method_name, llm_options) if accepts_llm_options else method_name
        output_dir = RESULT_ROOTS[getattr(method_func, "result_root", "results")] / dataset_name / label
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{transformation}.parquet"

        print(f"\nRunning {method_name}...")
        if accepts_llm_options:
            opts = {k: v for k, v in (llm_options or {}).items() if k != "repeat"}
            results_df = method_func(df_left, df_right, df_mapping, output_path=str(output_file),
                                     llm_options=opts)
        elif 'output_path' in inspect.signature(method_func).parameters:
            results_df = method_func(df_left, df_right, df_mapping, output_path=str(output_file))
        else:
            results_df = method_func(df_left, df_right, df_mapping)
        
        # Add metadata
        results_df['method'] = label
        results_df['transformation'] = transformation
        results_df['dataset'] = dataset_name
        results_df['timestamp'] = datetime.now().isoformat()
        
        # Save final results to parquet
        results_df.to_parquet(output_file, index=False)
        print(f"\nSaved: {output_file}")
        
        # Print summary
        correct = results_df['is_correct'].sum()
        total = len(results_df)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"  Total predictions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Run entity matching experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()) + ['all'],
                       help='Dataset to run')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated methods or "all"')
    parser.add_argument('--transformations', type=str, default='all',
                       help='Comma-separated transformations and/or groups: "all" (v1 four), '
                            '"factorial" (original + desc_ciphered, ids_ciphered, all_ciphered)')
    # Harness-v2 LLM options (ignored by non-LLM and v1 methods)
    parser.add_argument('--effort', type=str, default=None,
                        help='Reasoning effort override (low/medium/high/xhigh/max; OpenAI Luna/Terra also accept "none")')
    parser.add_argument('--sample-frac', type=float, default=None,
                        help='Run on a fixed random subsample of query records (seed 42)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel API requests')
    parser.add_argument('--shuffle-candidates', action='store_true',
                        help='Shuffle blocked candidate order (ordering-effect control)')
    parser.add_argument('--shuffle-seed', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=None, help='Override blocking top-k')
    parser.add_argument('--repeat', type=int, default=None,
                        help='Repeat index for run-to-run variance (separate result dir)')
    
    args = parser.parse_args()
    llm_options = {
        "effort": args.effort,
        "sample_frac": args.sample_frac,
        "workers": args.workers,
        "shuffle_candidates": args.shuffle_candidates,
        "shuffle_seed": args.shuffle_seed,
        "top_k": args.top_k,
        "repeat": args.repeat,
    }
    
    # Determine what to run
    if args.dataset == 'all':
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = [args.dataset]
    
    if args.methods == 'all':
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    
    transformations_to_run = []
    for t in (t.strip() for t in args.transformations.split(',')):
        for name in TRANSFORMATION_GROUPS.get(t, [t]):
            if name not in transformations_to_run:
                transformations_to_run.append(name)
    
    print("="*80)
    print("ENTITY MATCHING EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nDatasets: {datasets_to_run}")
    print(f"Methods: {methods_to_run}")
    print(f"Transformations: {transformations_to_run}")
    print(f"\nTotal experiments: {len(datasets_to_run) * len(methods_to_run) * len(transformations_to_run)}")
    
    # Run experiments
    results_summary = []
    
    for dataset in datasets_to_run:
        for method_name in methods_to_run:
            if method_name not in METHODS:
                print(f"⚠️  Unknown method: {method_name}, skipping")
                continue
            
            method_func = METHODS[method_name]
            
            for transformation in transformations_to_run:
                success = run_single_experiment(dataset, transformation, method_name, method_func,
                                                llm_options=llm_options)
                results_summary.append({
                    'dataset': dataset,
                    'method': method_name,
                    'transformation': transformation,
                    'success': success
                })
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    df_summary = pd.DataFrame(results_summary)
    print(f"\nTotal experiments: {len(df_summary)}")
    print(f"Successful: {df_summary['success'].sum()}")
    print(f"Failed: {(~df_summary['success']).sum()}")
    
    if (~df_summary['success']).any():
        print("\nFailed experiments:")
        failed = df_summary[~df_summary['success']]
        for _, row in failed.iterrows():
            print(f"  - {row['dataset']} / {row['method']} / {row['transformation']}")

if __name__ == "__main__":
    main()
