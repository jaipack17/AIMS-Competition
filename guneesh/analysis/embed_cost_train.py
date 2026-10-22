"""GPU-hour estimates for the 152 training benchmarks (train_benchmarks.csv), for several per-benchmark caps.

Same method as embed_cost_table.py: unique-text bytes from parquet footers / 4.3 bytes per token,
+40 template tokens, each item capped at 8192 tokens; FLOPs = 2*P*L + 2*layers*d*L^2 per item.
"""
import os

import numpy as np
import pandas as pd

from embed_cost_table import BYTES_PER_TOKEN, GPUS, MAXLEN, MODELS, TEMPLATE_TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    S = pd.read_csv(os.path.join(HERE, "item_text_sizes.csv")).set_index("bench")
    T = pd.read_csv(os.path.join(HERE, "train_benchmarks.csv"))
    d = S.loc[T.benchmark]
    for cap in (None, 2000, 1000, 500):
        n = d.n_items.values if cap is None else np.minimum(d.n_items.values, cap)
        L = np.minimum(d.content_bytes.values / np.maximum(d.n_items.values, 1) / BYTES_PER_TOKEN + TEMPLATE_TOKENS, MAXLEN)
        print(f"\n{'ALL items' if cap is None else f'CAP {cap}'}: {int(n.sum()):,} items, {(n * L).sum() / 1e6:,.0f}M tokens")
        rows = []
        for name, P, nl, dm, auc in MODELS:
            fl = (n * (2 * P * 1e9 * L + 2 * nl * dm * L ** 2)).sum()
            row = {"model": name, "AUC": auc, "GB": round(2 * P + 4)}
            for g, (tf, gb) in GPUS.items():
                row[g] = f"{fl / (tf * 1e12) / 3600:.0f} h" if 2 * P + 4 <= gb else "no fit"
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
    top = d.n_items.sort_values(ascending=False)
    print("\nlargest benchmarks:", (top.head(5)).to_dict(), f"| top-5 share of all items {top.head(5).sum() / top.sum():.0%}")
