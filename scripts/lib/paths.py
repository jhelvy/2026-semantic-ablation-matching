"""
Single source of truth for repo paths. Every path is anchored to the repo root,
so scripts behave the same from any working directory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
V1_LLM_RESULTS = RESULTS / "v1_llm"          # superseded v1 LLM runs (contaminated; kept for comparison)
FIGS = REPO_ROOT / "figs"
CHECKPOINTS = REPO_ROOT / "checkpoints"      # gitignored: partial harness-v2 runs
PILOT_RESULTS = REPO_ROOT / "pilot_results"  # gitignored: subscription (CLI) pilots, never reported
ENV_FILE = REPO_ROOT / ".env"

# result_root names used by method functions -> directories
RESULT_ROOTS = {"results": RESULTS, "pilot_results": PILOT_RESULTS}
