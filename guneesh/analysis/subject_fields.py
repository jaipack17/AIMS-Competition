"""Describe subject attributes across the binary training benchmarks.

Used to decide how to split ability (theta) into parts, e.g. model + harness.
"""
import sys
import pandas as pd

ROOT = "data/measurement-db"
FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "reasoning_effort", "harness_version", "subject_features_extra"]

tb = pd.read_csv("analysis/train_benchmarks.csv")
tb = tb[tb["response_type"] == "binary"] if "response_type" in tb else tb
rows = []
for b in tb["benchmark"]:
    s = pd.read_parquet(f"{ROOT}/{b}/formatted_tables/subjects.parquet")
    s["benchmark"] = b
    rows.append(s)
S = pd.concat(rows, ignore_index=True).fillna("")
S.to_parquet("analysis/subjects_binary_train.parquet")
print("benchmarks", S.benchmark.nunique(), "subject rows", len(S))
print("unique subject_id", S.subject_id.nunique())
full = S[FIELDS].astype(str).agg("|".join, axis=1)
print("unique full attribute tuples", full.nunique())
# does one subject_id always carry the same attributes?
k = S.assign(full=full).groupby("subject_id")["full"].nunique()
print("subject_ids with >1 attribute tuple", (k > 1).sum())
for f in FIELDS:
    v = S[f].astype(str)
    print(f"{f:24s} blank={(v == '').mean():.2f} unique={v.nunique()}  top={v.value_counts().head(6).to_dict()}")
