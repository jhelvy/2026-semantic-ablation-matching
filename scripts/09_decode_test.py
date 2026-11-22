"""
DECODE TEST (paid; about $1)
============================

Can a model read the transformed text? If it can recover the original words, the
transform did not remove meaning for that model, and good matching on it is not
evidence of matching without meaning.

Sample 30 left-side records per dataset (seed 42) and take the field the matchers
see first (Abt-Buy: name; Amazon-Google: title). Send them in batches of 10 with a
prompt that does NOT name the transformation, the same way the matching prompt
does not. Ciphered Words is the negative control: its whole-token map cannot be
decoded, so recovery there should be near zero.

Output cap: MAX_TOKENS (32k, above the matching runs' 8k). This is a capability probe:
breaking a substitution cipher can take long reasoning, and an 8k cap truncated it.

Score: of the original word tokens the transform changed (alphabetic, compared as
multisets against the transformed text), the share that appear in the model's
decoded line.

Outputs: results/decode_test_rows.csv (per record), results/decode_test.csv (summary).
A run replaces only the (dataset, transformation, model) cells it covers; rows for other
cells are kept, so new models or datasets can be added without re-calling old ones.

Settings match each model's matching runs: `effort` for models that have it (Kimi K3:
`reasoning_effort` low), Haiku 4.5's 1,024-token thinking budget (it has no effort).

Usage:
    uv run scripts/09_decode_test.py --dry-run                    # list jobs, no calls
    uv run scripts/09_decode_test.py --models haiku45,gpt56luna,kimik3
    uv run scripts/09_decode_test.py --datasets wdc-products --transforms all_ciphered
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.cv import load  # noqa: E402
from lib.llm_clients import MODELS, _client, _cost  # noqa: E402
from lib.paths import DATA, RESULTS  # noqa: E402

N_RECORDS, BATCH, SEED = 30, 10, 42
MAX_TOKENS, WORKERS = 32000, 9
# all_ciphered (factorial cell, both factors) added 2026-11-21; it is in the 30% breadth suite.
TRANSFORMS = ["ciphered_letters", "scrambled", "ciphered_words", "all_ciphered"]
MODEL_KEYS = ["gpt6luna", "haiku55", "sonnet55", "haiku45", "gpt56luna", "kimik3"]
SETS = {
    "abt-buy": dict(orig="abt-buy/data_original/abt.csv", test="abt-buy/data_test/abt_{t}.csv",
                    mp="abt-buy/data_original/abt_buy_perfect_mapping.csv", col="name"),
    "amazon-google": dict(orig="amazon-google/data_original/Amazon.csv",
                          test="amazon-google/data_test/amazon_{t}.csv",
                          mp="amazon-google/data_original/amazon_googleproducts_perfectmapping.csv",
                          col="title"),
    "wdc-products": dict(orig="wdc-products/data_original/offers.csv",
                         test="wdc-products/data_test/offers_{t}.csv",
                         mp="wdc-products/data_original/wdc_mapping.csv", col="title"),
}

PROMPT = """Each numbered line below is a product listing whose text has been altered and is hard to read.
Recover the original English text of each line as well as you can.

{lines}

Respond with JSON only: {{"decoded": [one string per line, in the same order]}}."""


def words(s):
    return re.findall(r"[a-z]+", str(s).lower())


def recovery(orig, transformed, decoded):
    """(changed tokens, recovered tokens): multiset of original words the transform changed."""
    changed = Counter(words(orig)) - Counter(words(transformed))
    got = Counter(words(decoded))
    return sum(changed.values()), sum(min(c, got[w]) for w, c in changed.items())


def call(cfg, prompt):
    """Free-text call. Returns (status, text, cost, stop_reason)."""
    if cfg.provider == "anthropic":
        import anthropic
        kw = ({"thinking": {"type": "enabled", "budget_tokens": cfg.thinking_budget}}
              if cfg.thinking_budget else {"output_config": {"effort": cfg.effort}})   # Haiku 4.5: no effort
        try:
            r = _client("anthropic").messages.create(
                model=cfg.model_id, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}], **kw)
        except anthropic.APIError as e:
            return "api_error", str(e)[:500], 0.0, ""
        text = "".join(b.text for b in r.content if b.type == "text")
        cost = _cost(cfg, r.usage.input_tokens, r.usage.output_tokens)
        status = {"refusal": "refusal", "max_tokens": "truncated"}.get(r.stop_reason, "ok")
        return status, text, cost, str(r.stop_reason)
    import openai
    if cfg.provider == "openai_compatible":   # Kimi K3 on Moonshot: Chat Completions
        try:
            r = _client(f"openai_compatible:|{cfg.base_url}|{cfg.api_key_env}").chat.completions.create(
                model=cfg.model_id, messages=[{"role": "user", "content": prompt}],
                **{cfg.max_tokens_param: MAX_TOKENS, "reasoning_effort": cfg.effort})
        except openai.APIError as e:
            return "api_error", str(e)[:500], 0.0, ""
        ch = r.choices[0]
        u = r.usage
        cost = _cost(cfg, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        status = ("refusal" if getattr(ch.message, "refusal", None) or ch.finish_reason == "content_filter"
                  else "truncated" if ch.finish_reason == "length" else "ok")
        return status, ch.message.content or "", cost, str(ch.finish_reason)
    try:
        r = _client("openai").responses.create(
            model=cfg.model_id, input=[{"role": "user", "content": prompt}],
            max_output_tokens=MAX_TOKENS, reasoning={"effort": cfg.effort})
    except openai.APIError as e:
        flagged = "flagged as potentially violating" in str(e).lower()
        return ("refusal" if flagged else "api_error"), str(e)[:500], 0.0, ""
    cost = _cost(cfg, r.usage.input_tokens or 0, r.usage.output_tokens or 0)
    refused = any(getattr(p, "type", "") == "refusal" for it in r.output or [] for p in getattr(it, "content", None) or [])
    stop = str(r.status) + (f":{r.incomplete_details.reason}" if r.incomplete_details else "")
    status = "refusal" if refused else "truncated" if r.status == "incomplete" else "ok"
    return status, r.output_text or "", cost, stop


def parse(text, n):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        dec = json.loads(m.group(0))["decoded"] if m else None
    except (json.JSONDecodeError, KeyError, TypeError):
        dec = None
    if not isinstance(dec, list) or len(dec) != n:
        return None
    return [str(x) for x in dec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_KEYS))
    ap.add_argument("--datasets", default=",".join(SETS))
    ap.add_argument("--transforms", default=",".join(TRANSFORMS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    models, sets, transforms = a.models.split(","), a.datasets.split(","), a.transforms.split(",")
    assert set(models) <= set(MODEL_KEYS) and set(sets) <= set(SETS) and set(transforms) <= set(TRANSFORMS)

    jobs = []   # (ds, t, model key, chunk)
    for ds, c in ((k, SETS[k]) for k in sets):
        mp = load(DATA / c["mp"])
        orig = load(DATA / c["orig"]); orig["id"] = orig["id"].astype(str)
        orig = orig[orig["id"].isin(set(mp.iloc[:, 0].astype(str)))]
        sample = orig.sample(N_RECORDS, random_state=SEED)[["id", c["col"]]].rename(columns={c["col"]: "orig"})
        for t in transforms:
            tdf = load(DATA / c["test"].format(t=t)); tdf["id"] = tdf["id"].astype(str)
            s = sample.merge(tdf[["id", c["col"]]].rename(columns={c["col"]: "transformed"}), on="id")
            for mk in models:
                for b in range(0, len(s), BATCH):
                    jobs.append((ds, t, mk, b // BATCH + 1, s.iloc[b:b + BATCH]))

    def run(job):
        ds, t, mk, batch, chunk = job
        lines = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(chunk["transformed"]))
        status, text, cost, stop = call(MODELS[mk], PROMPT.format(lines=lines))
        dec = parse(text, len(chunk)) if status == "ok" else None
        if status == "ok" and dec is None:
            status = "parse_fail"
        print(f"{ds:14s}{t:18s}{mk:10s} batch {batch}: {status:10s} {stop:24s} ${cost:.4f}", flush=True)
        out = []
        for i, r in enumerate(chunk.itertuples()):
            d = dec[i] if dec else ""
            n_changed, n_rec = recovery(r.orig, r.transformed, d)
            out.append(dict(dataset=ds, transformation=t, model=mk, batch=batch, id=r.id, status=status,
                            stop_reason=stop, orig=r.orig, transformed=r.transformed, decoded=d,
                            n_changed=n_changed, n_recovered=n_rec, cost_usd=cost / len(chunk),
                            raw_response=text if i == 0 else ""))
        return out

    cells = sorted({(j[0], j[1], j[2]) for j in jobs})
    print(f"{len(jobs)} calls over {len(cells)} (dataset, transformation, model) cells")
    if a.dry_run:
        return

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = [r for out in ex.map(run, jobs) for r in out]

    rows = pd.DataFrame(rows)
    old = RESULTS / "decode_test_rows.csv"
    if old.exists():   # keep every cell this run did not cover
        prev = pd.read_csv(old, dtype={"id": str})
        key = prev[["dataset", "transformation", "model"]].apply(tuple, axis=1)
        rows = pd.concat([prev[~key.isin(cells)], rows], ignore_index=True)
    rows.to_csv(RESULTS / "decode_test_rows.csv", index=False)
    summ = (rows.groupby(["dataset", "transformation", "model"])
            .agg(n=("id", "size"), refused=("status", lambda x: (x == "refusal").mean()),
                 n_changed=("n_changed", "sum"), n_recovered=("n_recovered", "sum"),
                 cost_usd=("cost_usd", "sum")).reset_index())
    answered = rows[rows["status"] == "ok"].groupby(["dataset", "transformation", "model"])[
        ["n_changed", "n_recovered"]].sum()
    summ = summ.merge((answered["n_recovered"] / answered["n_changed"]).rename("recovery_answered"),
                      on=["dataset", "transformation", "model"], how="left")
    summ["recovery_all"] = summ["n_recovered"] / summ["n_changed"]
    summ.to_csv(RESULTS / "decode_test.csv", index=False)
    print(summ.round(3).to_string(index=False))
    ran = rows[rows[["dataset", "transformation", "model"]].apply(tuple, axis=1).isin(cells)]
    print(f"\nthis run ${ran['cost_usd'].sum():.2f}; all cells ${rows['cost_usd'].sum():.2f}\nsaved -> {RESULTS / 'decode_test.csv'}")


if __name__ == "__main__":
    main()
