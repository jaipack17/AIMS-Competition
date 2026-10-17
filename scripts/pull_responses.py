"""Download responses.parquet for every benchmark in measurement-db."""
import json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

R = "aims-foundations/measurement-db"
sizes = json.load(open("data/table_sizes.json"))

def get(b):
    try:
        hf_hub_download(R, f"{b}/formatted_tables/responses.parquet", repo_type="dataset", local_dir="data/mdb")
        return None
    except Exception as e:
        return (b, str(e)[:100])

with ThreadPoolExecutor(8) as ex:
    errs = [e for e in ex.map(get, sizes) if e]
print("done;", len(errs), "errors", errs[:3])
