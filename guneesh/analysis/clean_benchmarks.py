"""Select benchmarks that are clean for LEGO-CM-style training, and count their items.

A benchmark is CLEAN if
  1. benchmark-level direction == higher_is_better, and
  2. its scale has known bounds that map to [0, 1]:
       binary                    -> keep 0/1
       interval with min and max -> (y - min) / (max - min)
       discrete with values      -> (y - min(values)) / (max(values) - min(values))
  3. it is not a pairwise win-rate (outcome depends on the opponent, not on the item), and
  4. its recorded responses actually lie inside the declared scale (checked on responses.parquet).
Benchmarks with per-item ("mixed") scales, unbounded or unknown bounds, or other directions are excluded.
Writes analysis/clean_benchmarks.csv and analysis/excluded_benchmarks.csv.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "measurement-db")


def scale_bounds(rt, s):
    if rt == "binary":
        return 0.0, 1.0
    if s.get("values"):
        v = [float(x) for x in s["values"]]
        return min(v), max(v)
    if s.get("min") is not None and s.get("max") is not None:
        return float(s["min"]), float(s["max"])
    return None


def main():
    B = pd.read_parquet(os.path.join(HERE, "all_benchmarks.parquet")).reset_index(drop=True)
    keep, drop = [], []
    for r in B.itertuples():
        s = json.loads(r.response_scale) if isinstance(r.response_scale, str) else {}
        d = s.get("direction")
        if d != "higher_is_better":
            drop.append((r.benchmark_id, r.response_type, f"direction={d}"))
            continue
        if r.response_type == "win_rate":
            drop.append((r.benchmark_id, r.response_type, "pairwise win rate"))
            continue
        bnd = scale_bounds(r.response_type, s)
        if bnd is None or not bnd[1] > bnd[0]:
            drop.append((r.benchmark_id, r.response_type, "no usable min/max"))
            continue
        path = os.path.join(DATA, r.benchmark_id, "formatted_tables", "responses.parquet")
        if not os.path.exists(path):
            drop.append((r.benchmark_id, r.response_type, "responses not downloaded"))
            continue
        x = pd.read_parquet(path, columns=["subject_id", "item_id", "response"]).dropna(subset=["response"])
        y = x.response.astype(float)
        lo, hi = bnd
        outside = float(((y < lo - 1e-9) | (y > hi + 1e-9)).mean())
        if r.response_type == "binary":
            outside = float((~y.isin([0.0, 1.0])).mean())
        if len(x) == 0:
            drop.append((r.benchmark_id, r.response_type, "all responses are null"))
            continue
        if outside > 0:
            drop.append((r.benchmark_id, r.response_type, f"{outside:.1%} of responses outside declared scale"))
            continue
        per_subj = x.groupby("subject_id").item_id.nunique()
        z = (y - lo) / (hi - lo)
        keep.append({"benchmark": r.benchmark_id, "response_type": r.response_type, "min": lo, "max": hi,
                     "items_with_responses": x.item_id.nunique(), "subjects": x.subject_id.nunique(),
                     "responses": len(x), "pairs_ge80_items": int((per_subj >= 80).sum()),
                     "share_exact_0_or_1": float(((z == 0) | (z == 1)).mean()), "mean_score01": float(z.mean())})
    K = pd.DataFrame(keep).sort_values(["response_type", "benchmark"])
    D = pd.DataFrame(drop, columns=["benchmark", "response_type", "reason"])
    K.to_csv(os.path.join(HERE, "clean_benchmarks.csv"), index=False)
    D.to_csv(os.path.join(HERE, "excluded_benchmarks.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(K.to_string(index=False))
    print("\nCLEAN by type:\n", K.groupby("response_type").agg(benchmarks=("benchmark", "size"),
          items=("items_with_responses", "sum"), responses=("responses", "sum")).to_string())
    print(f"\nTOTAL clean: {len(K)} benchmarks, {K.items_with_responses.sum():,} items, {K.responses.sum():,} responses;"
          f" with >=1 test-eligible pair (>=80 items): {(K.pairs_ge80_items > 0).sum()}")
    print("\nEXCLUDED by reason:\n", D.reason.str.replace(r"^[\d.]+% of", "x% of", regex=True).value_counts().to_string())
    print(D[~D.reason.str.startswith("direction=None")].to_string(index=False))


if __name__ == "__main__":
    main()
