"""Download the small per-benchmark tables (benchmarks, subjects) from measurement-db."""
import json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

R = "aims-foundations/measurement-db"
sizes = json.load(open("data/table_sizes.json"))

def get(args):
    b, name = args
    try:
        hf_hub_download(R, f"{b}/formatted_tables/{name}", repo_type="dataset", local_dir="data/mdb")
        return None
    except Exception as e:
        return (b, name, str(e)[:100])

jobs = [(b, n) for b in sizes for n in ("benchmarks.parquet", "subjects.parquet")]
with ThreadPoolExecutor(16) as ex:
    errs = [e for e in ex.map(get, jobs) if e]
print(len(jobs) - len(errs), "ok;", len(errs), "errors", errs[:3])
