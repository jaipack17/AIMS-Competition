"""Summarize per-benchmark table sizes in measurement-db (no downloads)."""
import json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

R = "aims-foundations/measurement-db"
api = HfApi()
benches = [f.path for f in api.list_repo_tree(R, repo_type="dataset") if f.path and "/" not in f.path and not f.path.startswith(".") and hasattr(f, "tree_id")]

def one(b):
    try:
        return b, {f.path.split("/")[-1]: f.size for f in api.list_repo_tree(R, repo_type="dataset", path_in_repo=f"{b}/formatted_tables") if getattr(f, "size", None) is not None}
    except Exception as e:
        return b, {"_err": str(e)[:80]}

with ThreadPoolExecutor(16) as ex:
    out = dict(ex.map(one, benches))
json.dump(out, open("data/table_sizes.json", "w"))
tot = {}
for b, d in out.items():
    for k, v in d.items():
        if k != "_err": tot[k] = tot.get(k, 0) + v
print(len(benches), "benchmarks;", sum("_err" in d for d in out.values()), "errors;", sum("responses.parquet" not in d for d in out.values()), "without responses")
for k, v in sorted(tot.items(), key=lambda x: -x[1]): print(f"{k:22s}{v/1e9:9.3f} GB")
