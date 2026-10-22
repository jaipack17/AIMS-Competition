"""Per-benchmark item-text size from parquet footers only (no full download).

For each benchmark's items.parquet, reads the footer metadata over HF and sums the
uncompressed byte size of the `content` and `item_features` columns.
Output: analysis/item_text_sizes.csv
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/aims-foundations/measurement-db"
HERE = os.path.dirname(os.path.abspath(__file__))
fs = HfFileSystem()


def one(b):
    try:
        with fs.open(f"{REPO}/{b}/formatted_tables/items.parquet", "rb", block_size=1 << 16) as f:
            md = pq.ParquetFile(f).metadata
            names = md.schema.names
            tot = {"content": 0, "item_features": 0}
            for rg in range(md.num_row_groups):
                g = md.row_group(rg)
                for c in range(g.num_columns):
                    col = g.column(c)
                    key = col.path_in_schema.split(".")[0]
                    if key in tot:
                        tot[key] += col.total_uncompressed_size
            return {"bench": b, "n_items": md.num_rows, "content_bytes": tot["content"],
                    "features_bytes": tot["item_features"], "err": ""}
    except Exception as e:
        return {"bench": b, "n_items": -1, "content_bytes": -1, "features_bytes": -1, "err": str(e)[:100]}


if __name__ == "__main__":
    benches = [l.strip() for l in open(os.path.join(HERE, "hf_benchmarks.txt")) if l.strip()]
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, benches))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "item_text_sizes.csv"), index=False)
    print(df.err.ne("").sum(), "errors;", len(df), "benchmarks")
    print(df[df.err.eq("")][["n_items", "content_bytes"]].sum())
