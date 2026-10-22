"""Download items.parquet for the binary training benchmarks (train_benchmarks.csv) into data/measurement-db/."""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "aims-foundations/measurement-db"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "measurement-db")


def get(b):
    err = None
    for _ in range(4):
        try:
            hf_hub_download(REPO, f"{b}/formatted_tables/items.parquet", repo_type="dataset", local_dir=OUT)
            return None
        except Exception as e:
            err = f"{b}: {type(e).__name__}: {str(e)[:100]}"
    return err


if __name__ == "__main__":
    T = pd.read_csv(os.path.join(HERE, "train_benchmarks.csv"))
    benches = T.loc[T.response_type == "binary", "benchmark"].tolist()
    with ThreadPoolExecutor(8) as ex:
        errs = [e for e in ex.map(get, benches) if e]
    print(len(benches) - len(errs), "ok;", len(errs), "failed", *errs, sep="\n")
