"""Leave-benchmark-out coverage of subject fields, from the per-benchmark subjects tables."""
import glob, os
import pandas as pd

rows = []
for f in glob.glob("data/mdb/*/formatted_tables/subjects.parquet"):
    s = pd.read_parquet(f)
    s["bench"] = f.split("/")[2]
    rows.append(s)
S = pd.concat(rows, ignore_index=True)
B = pd.concat([pd.read_parquet(f) for f in glob.glob("data/mdb/*/formatted_tables/benchmarks.parquet")], ignore_index=True)
for c in ["normalized_name","provider","release_date","access_date","harness","reasoning_effort","harness_version","subject_features_extra"]:
    S[c] = S[c].fillna("").astype(str)
print("benchmarks", S.bench.nunique(), "| subject rows", len(S), "| distinct subject_id", S.subject_id.nunique())
print("\nField population / distinct values (rows):")
for c in ["normalized_name","provider","release_date","access_date","harness","reasoning_effort","harness_version","subject_features_extra"]:
    print(f"  {c:24s} non-empty {(S[c]!='').mean():6.1%}  distinct {S.loc[S[c]!='',c].nunique()}")
print("\nsubjects per benchmark: median", S.groupby('bench').size().median(), " ==1:", (S.groupby('bench').size()==1).sum(), " >=10:", (S.groupby('bench').size()>=10).sum())

# leave-one-benchmark-out: a subject-row is "seen" if its value appears in any other benchmark
fields = {
    "name": ["normalized_name"], "provider": ["provider"], "harness": ["harness"],
    "effort": ["reasoning_effort"], "harness_version": ["harness_version"],
    "name+harness+effort+version (full subject)": ["normalized_name","harness","reasoning_effort","harness_version"],
    "name+harness+effort": ["normalized_name","harness","reasoning_effort"],
}
S["key_full"] = S[["normalized_name","harness","reasoning_effort","harness_version"]].agg("|".join, axis=1)
res = {}
for label, cols in fields.items():
    k = S[cols].agg("|".join, axis=1)
    nb = k.groupby(k).transform(lambda x: S.loc[x.index, "bench"].nunique())
    seen = nb > 1
    res[label] = seen
    # count only rows with non-empty values for single fields
    mask = (S[cols[0]] != "") if len(cols) == 1 else slice(None)
    print(f"{label:44s} seen-elsewhere {seen[mask].mean():6.1%} of rows")

S["name_seen"] = res["name"]; S["full_seen"] = res["full subject)".join(["name+harness+effort+version (", ""])] if False else res["name+harness+effort+version (full subject)"]
per = S.groupby("bench").agg(n=("subject_id","size"), name_seen=("name_seen","mean"), full_seen=("full_seen","mean"))
print("\nPer-benchmark share of subjects whose NAME appears in another benchmark: quantiles")
print(per.name_seen.describe(percentiles=[.1,.25,.5,.75]).round(3).to_string())
print("\nPer-benchmark share whose full (name,harness,effort,version) appears elsewhere:")
print(per.full_seen.describe(percentiles=[.1,.25,.5,.75]).round(3).to_string())
print("benchmarks with name_seen<0.5:", (per.name_seen<.5).sum(), "of", len(per))

# time trend: coverage by release year
S["year"] = pd.to_numeric(S.release_date.str[:4], errors="coerce")
print("\nName-seen-elsewhere by subject release year:")
print(S.dropna(subset=["year"]).groupby("year").agg(rows=("name_seen","size"), name_seen=("name_seen","mean")).tail(8).round(3).to_string())
# benchmark release vs coverage
B["byear"] = pd.to_numeric(B.release_date.str[:4], errors="coerce")
m = per.join(B.set_index("benchmark_id").byear)
print("\nBenchmark release year vs share of seen names:")
print(m.groupby("byear").agg(nbench=("n","size"), name_seen=("name_seen","mean")).tail(8).round(3).to_string())
print("\nTop harness values:", S.loc[S.harness!="","harness"].value_counts().head(8).to_dict())
print("effort values:", S.loc[S.reasoning_effort!="","reasoning_effort"].value_counts().head(8).to_dict())
S.to_parquet("data/subjects_all.parquet")
