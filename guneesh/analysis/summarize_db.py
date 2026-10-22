"""Summarize measurement-db from the small tables: response scales and subject/model counts."""
import glob, json, os
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "measurement-db")
b = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{D}/*/formatted_tables/benchmarks.parquet")])
b["kind"] = b.response_scale.map(lambda s: json.loads(s).get("kind") if isinstance(s, str) else None)
for col in ("response_type", "kind", "release", "granularity"):
    print(f"== {col}"); print(b[col].value_counts(dropna=False).to_string())
print("benchmarks with repeated trials (max_trial>1):", int((b.max_trial > 1).sum()))
bi = b[b.response_type == "binary"]
print(f"binary benchmarks: {len(bi)}; with n_items>=80: {(bi.n_items >= 80).sum()}")

ss = []
for f in glob.glob(f"{D}/*/formatted_tables/subjects.parquet"):
    d = pd.read_parquet(f); d["bench"] = os.path.basename(os.path.dirname(os.path.dirname(f))); ss.append(d)
s = pd.concat(ss, ignore_index=True)
print(f"\n== subject rows {len(s)}, unique subject_id {s.subject_id.nunique()}")
print(f"rows with normalized_name: {s.normalized_name.notna().mean():.3f}")
print(f"unique normalized_name {s.normalized_name.nunique()}, unique display_name {s.display_name.nunique()}, providers {s.provider.nunique()}")
s.to_parquet(os.path.join(os.path.dirname(__file__), "all_subjects.parquet"))
b.to_parquet(os.path.join(os.path.dirname(__file__), "all_benchmarks.parquet"))
