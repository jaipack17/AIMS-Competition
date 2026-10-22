"""For each binary training benchmark, how many of its eligible pairs
(subject with >= 80 distinct items) have a subject that is also seen in
some OTHER training benchmark, under different subject keys.

This mimics the test: a new benchmark arrives; can we reuse an ability
learned elsewhere for its subjects?
"""
import pandas as pd
import pyarrow.parquet as pq

ROOT = "data/measurement-db"
S = pd.read_parquet("analysis/subjects_binary_train.parquet")
ALL = ["normalized_name", "provider", "release_date", "access_date", "harness",
       "reasoning_effort", "harness_version", "subject_features_extra"]
KEYS = {
    "full_tuple(all 8 fields)": ALL,
    "name+harness+effort": ["normalized_name", "harness", "reasoning_effort"],
    "name+effort": ["normalized_name", "reasoning_effort"],
    "name only": ["normalized_name"],
}
# eligible pairs
elig = []
for b in S.benchmark.unique():
    r = pq.read_table(f"{ROOT}/{b}/formatted_tables/responses.parquet",
                      columns=["subject_id", "item_id"]).to_pandas()
    n = r.groupby("subject_id").item_id.nunique()
    for sid in n[n >= 80].index:
        elig.append((b, sid))
E = pd.DataFrame(elig, columns=["benchmark", "subject_id"]).merge(S, on=["benchmark", "subject_id"])
print("eligible pairs", len(E), "in", E.benchmark.nunique(), "benchmarks;",
      "named:", (E.normalized_name != "").mean().round(3))
out = {}
for name, cols in KEYS.items():
    key = S[cols].astype(str).agg("|".join, axis=1)
    blank = S.normalized_name == ""
    # set of benchmarks each key appears in (blank names never match)
    kb = pd.DataFrame({"key": key, "b": S.benchmark})[~blank].groupby("key").b.agg(set)
    ek = E[cols].astype(str).agg("|".join, axis=1)
    seen = [(k in kb.index) and len(kb[k] - {b}) > 0 and nm != ""
            for k, b, nm in zip(ek, E.benchmark, E.normalized_name)]
    E["seen"] = seen
    per_b = E.groupby("benchmark").seen.mean()
    out[name] = (E.seen.mean(), per_b.mean(), (per_b == 0).sum())
for k, (a, bm, z) in out.items():
    print(f"{k:28s} pairs seen elsewhere={a:.3f}  mean over benchmarks={bm:.3f}  benchmarks with none seen={z}")
