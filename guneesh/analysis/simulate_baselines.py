"""Local simulation of the PAIEC streaming protocol, used to score simple baselines.

Protocol, following Starting Kit/README.md:
  * Eligible pairs: (subject, benchmark) with >= 80 distinct items, binary 0/1 responses only.
  * Each pair: persistent random 50/50 split of its distinct items into acquisition and
    evaluation pools (odd count -> extra item to acquisition).
  * Default (no labeling.py) acquisition: candidates stream in random order and are
    accepted with probability labels_remaining / items_remaining, i.e. a uniform random
    subset of min(31, |acquisition|) items; budgets use nested prefixes of that subset.
  * A revealed label is one recorded response (a random trial when an item has repeats).
  * At budget b every prediction sees the labels revealed so far for ALL sampled pairs.
  * Brier is averaged over every response row of the evaluation pool; budgets
    0,1,3,7,15,31 are combined with weights 0.1,0.2,0.2,0.2,0.2,0.1; pairs weigh equally.

Usage:  python analysis/simulate_baselines.py [--seeds 3] [--max-pairs-per-bench 0]
"""

from __future__ import annotations

import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "measurement-db")


def load_binary_benchmarks(min_items=80):
    tables = {}
    for path in sorted(glob.glob(os.path.join(DATA, "*", "formatted_tables", "responses.parquet"))):
        bench = path.replace("\\", "/").split("/")[-3]
        df = pd.read_parquet(path, columns=["subject_key", "item_key", "response"])
        df = df.dropna(subset=["response"])
        if df.empty or not df["response"].isin([0, 1]).all():
            continue  # keep strictly binary benchmarks only
        df["response"] = df["response"].astype(np.int8)
        df["subject_key"] = df["subject_key"].astype(str)
        df["item_key"] = df["item_key"].astype(str)
        n_items = df.groupby("subject_key")["item_key"].nunique()
        keep = n_items[n_items >= min_items].index
        df = df[df["subject_key"].isin(keep)]
        if not df.empty:
            tables[bench] = df
    return tables


def build_pairs(tables, seed, max_pairs_per_bench=0):
    """Split each eligible pair and draw its random acquisition prefix."""
    pairs = []
    for bench, df in tables.items():
        subjects = sorted(df["subject_key"].unique())
        rng_b = np.random.default_rng(zlib.crc32(f"{bench}|{seed}".encode()))
        if max_pairs_per_bench and len(subjects) > max_pairs_per_bench:
            subjects = list(rng_b.choice(subjects, max_pairs_per_bench, replace=False))
        for subj, g in df[df["subject_key"].isin(subjects)].groupby("subject_key"):
            rng = np.random.default_rng(zlib.crc32(f"{bench}|{subj}|{seed}".encode()))
            items = np.array(sorted(g["item_key"].unique()), dtype=object)
            rng.shuffle(items)
            n_acq = (len(items) + 1) // 2
            acq, ev = items[:n_acq], set(items[n_acq:])
            # uniform random subset of size min(31, |acq|), kept in stream order
            chosen = acq[: min(31, len(acq))]
            by_item = g.groupby("item_key")["response"].apply(lambda s: s.to_numpy())
            labels = [int(rng.choice(by_item[it])) for it in chosen]
            eval_rows = g[g["item_key"].isin(ev)]["response"].to_numpy()
            pairs.append({"bench": bench, "subject": subj, "labels": np.array(labels),
                          "eval_y": eval_rows})
    return pairs


# ---------- baseline predictors (pair-level: identical prediction for all items) ----------

def make_predictors(pairs, budget):
    """Return {name: list of predicted probabilities per pair} at a given budget.

    Each predictor may use every pair's first `budget` labels (shared evidence)."""
    own = [p["labels"][:budget] for p in pairs]
    bench_succ, bench_n = {}, {}
    for p, l in zip(pairs, own):
        bench_succ[p["bench"]] = bench_succ.get(p["bench"], 0) + l.sum()
        bench_n[p["bench"]] = bench_n.get(p["bench"], 0) + len(l)
    out = {k: [] for k in ("constant_0.5", "empirical_mean (official)", "laplace_mean",
                           "pooled_bench_mean", "eb_shrink_to_bench")}
    for p, l in zip(pairs, own):
        n, k = len(l), l.sum()
        out["constant_0.5"].append(0.5)
        out["empirical_mean (official)"].append(k / n if n else 0.5)
        out["laplace_mean"].append((k + 1) / (n + 2))
        bs, bn = bench_succ[p["bench"]], bench_n[p["bench"]]
        pooled = (bs + 1) / (bn + 2)
        out["pooled_bench_mean"].append(pooled)
        s = 4.0  # prior strength in pseudo-labels
        out["eb_shrink_to_bench"].append((k + s * pooled) / (n + s))
    return out


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def run(seed, tables, max_pairs_per_bench):
    pairs = build_pairs(tables, seed, max_pairs_per_bench)
    per_budget = {}
    for b in BUDGETS:
        preds = make_predictors(pairs, b)
        for name, ps in preds.items():
            per_budget.setdefault(name, []).append(
                np.array([brier(pr, p["eval_y"]) for pr, p in zip(ps, pairs)]))
    w = np.array(WEIGHTS)[:, None]
    rows = []
    bench_of = np.array([p["bench"] for p in pairs])
    for name, mats in per_budget.items():
        m = np.vstack(mats)                     # budgets x pairs
        alc_pair = (w * m).sum(0)
        bench_mean = pd.Series(alc_pair).groupby(bench_of).mean().mean()
        rows.append({"method": name, "ALC (pairs equal)": alc_pair.mean(),
                     "ALC (benchmarks equal)": bench_mean,
                     **{f"B{b}": m[i].mean() for i, b in enumerate(BUDGETS)}})
    return pd.DataFrame(rows), len(pairs), len(set(bench_of))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-pairs-per-bench", type=int, default=0)
    args = ap.parse_args()
    tables = load_binary_benchmarks()
    print(f"binary benchmarks with >=1 eligible pair: {len(tables)}")
    res = []
    for s in range(args.seeds):
        df, npairs, nb = run(s, tables, args.max_pairs_per_bench)
        df["seed"] = s
        res.append(df)
    allr = pd.concat(res)
    summ = allr.groupby("method").agg(["mean", "std"]).drop(columns="seed", level=0)
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    print(f"pairs per seed: {npairs}, benchmarks: {nb}, seeds: {args.seeds}")
    print(summ.xs("mean", axis=1, level=1).sort_values("ALC (pairs equal)"))
    print("\nstd across seeds of ALC (pairs equal):")
    print(summ[("ALC (pairs equal)", "std")])


if __name__ == "__main__":
    main()
