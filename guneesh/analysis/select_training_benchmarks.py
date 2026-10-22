"""Training benchmark list: clean benchmarks (clean_benchmarks.py) with >= MIN_SUBJECTS subjects.

Subjects = distinct subject_id with at least one non-null response in the benchmark.
Writes analysis/train_benchmarks.csv.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_SUBJECTS = 5

if __name__ == "__main__":
    K = pd.read_csv(os.path.join(HERE, "clean_benchmarks.csv"))
    T = K[K.subjects >= MIN_SUBJECTS].sort_values(["response_type", "benchmark"])
    T.to_csv(os.path.join(HERE, "train_benchmarks.csv"), index=False)
    print(f"{len(T)} of {len(K)} clean benchmarks have >= {MIN_SUBJECTS} subjects")
    print(T.groupby("response_type").agg(benchmarks=("benchmark", "size"), items=("items_with_responses", "sum"),
                                         responses=("responses", "sum")).to_string())
    print(f"TOTAL: {T.items_with_responses.sum():,} items, {T.responses.sum():,} responses, "
          f"{(T.pairs_ge80_items > 0).sum()} benchmarks with a >=80-item pair")
    print("\nRemoved (<5 subjects):")
    print(", ".join(f"{r.benchmark} ({r.subjects})" for r in K[K.subjects < MIN_SUBJECTS].itertuples()))
