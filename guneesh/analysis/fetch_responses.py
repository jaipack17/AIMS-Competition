"""Download every benchmark's responses.parquet from measurement-db into data/measurement-db/."""
import os
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

REPO = "aims-foundations/measurement-db"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "measurement-db")


def get(b):
    for attempt in range(4):
        try:
            hf_hub_download(REPO, f"{b}/formatted_tables/responses.parquet", repo_type="dataset", local_dir=OUT)
            return None
        except Exception as e:
            err = f"{b}: {type(e).__name__}: {str(e)[:100]}"
    return err


if __name__ == "__main__":
    benches = [l.strip() for l in open(os.path.join(HERE, "hf_benchmarks.txt")) if l.strip()]
    with ThreadPoolExecutor(8) as ex:
        errs = [e for e in ex.map(get, benches) if e]
    print(len(benches) - len(errs), "ok;", len(errs), "failed")
    for e in errs:
        print("FAIL", e)
