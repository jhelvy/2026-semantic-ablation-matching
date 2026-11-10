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

Usage:
    uv run scripts/09_decode_test.py
"""

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
TRANSFORMS = ["ciphered_letters", "scrambled", "ciphered_words"]
MODEL_KEYS = ["gpt6luna", "haiku55", "sonnet55"]
SETS = {
    "abt-buy": dict(orig="abt-buy/data_original/abt.csv", test="abt-buy/data_test/abt_{t}.csv",
                    mp="abt-buy/data_original/abt_buy_perfect_mapping.csv", col="name"),
    "amazon-google": dict(orig="amazon-google/data_original/Amazon.csv",
                          test="amazon-google/data_test/amazon_{t}.csv",
                          mp="amazon-google/data_original/amazon_googleproducts_perfectmapping.csv",
                          col="title"),
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
        try:
            r = _client("anthropic").messages.create(
                model=cfg.model_id, max_tokens=MAX_TOKENS,
                output_config={"effort": cfg.effort}, messages=[{"role": "user", "content": prompt}])
        except anthropic.APIError as e:
            return "api_error", str(e)[:500], 0.0, ""
        text = "".join(b.text for b in r.content if b.type == "text")
        cost = _cost(cfg, r.usage.input_tokens, r.usage.output_tokens)
        status = {"refusal": "refusal", "max_tokens": "truncated"}.get(r.stop_reason, "ok")
        return status, text, cost, str(r.stop_reason)
    import openai
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
    jobs = []   # (ds, t, model key, chunk)
    for ds, c in SETS.items():
        mp = load(DATA / c["mp"])
        orig = load(DATA / c["orig"]); orig["id"] = orig["id"].astype(str)
        orig = orig[orig["id"].isin(set(mp.iloc[:, 0].astype(str)))]
        sample = orig.sample(N_RECORDS, random_state=SEED)[["id", c["col"]]].rename(columns={c["col"]: "orig"})
        for t in TRANSFORMS:
            tdf = load(DATA / c["test"].format(t=t)); tdf["id"] = tdf["id"].astype(str)
            s = sample.merge(tdf[["id", c["col"]]].rename(columns={c["col"]: "transformed"}), on="id")
            for mk in MODEL_KEYS:
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

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = [r for out in ex.map(run, jobs) for r in out]

    rows = pd.DataFrame(rows)
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
    print(f"\ntotal ${rows['cost_usd'].sum():.2f}\nsaved -> {RESULTS / 'decode_test.csv'}")


if __name__ == "__main__":
    main()
