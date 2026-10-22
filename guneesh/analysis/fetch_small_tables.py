"""Download only the small per-benchmark tables (benchmarks.parquet, subjects.parquet) from measurement-db."""
import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

REPO = "aims-foundations/measurement-db"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "measurement-db")
benches = open(os.path.join(os.path.dirname(__file__), "hf_benchmarks.txt")).read().split()

def get(args):
    b, name = args
    try:
        hf_hub_download(REPO, f"{b}/formatted_tables/{name}", repo_type="dataset", local_dir=OUT)
        return None
    except Exception as e:
        return f"{b}/{name}: {type(e).__name__}"

jobs = [(b, n) for b in benches for n in ("benchmarks.parquet", "subjects.parquet")]
with ThreadPoolExecutor(16) as ex:
    errs = [e for e in ex.map(get, jobs) if e]
print(f"requested {len(jobs)}, failed {len(errs)}")
print("\n".join(errs[:30]))
