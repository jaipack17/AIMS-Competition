"""Paired comparison of each MLP submission with its ridge twin (analysis/mlp_variants.py): local ALC (mean over the
150 draws of each draw's mean over its pairs; lower is better), the paired difference MLP - ridge with its standard
error over draws, the draws and folds where the MLP is better, and the ALC at each label budget.
Run from the project root:  python analysis/mlp_variants_report.py
"""
import numpy as np
import pandas as pd

from mlp_variants import VARIANTS

rows = []
for v in VARIANTS:
    r = pd.read_csv(f"analysis/simulate_alc_regress_{v}.csv")
    m = pd.read_csv(f"analysis/simulate_alc_{v}_mlp.csv")
    assert (r[["fold", "draw", "pair"]].values == m[["fold", "draw", "pair"]].values).all(), v  # same draws and pairs
    dr, dm = r.groupby(["fold", "draw"]).alc.mean(), m.groupby(["fold", "draw"]).alc.mean()
    d = dm - dr
    folds = (m.groupby("fold").alc.mean() - r.groupby("fold").alc.mean() < 0).sum()
    budgets = " ".join(f"{m[f'b{b}'].mean() - r[f'b{b}'].mean():+.4f}" for b in (0, 1, 3, 7, 15, 31))
    rows.append({"variant": v, "ridge": dr.mean(), "mlp": dm.mean(), "mlp - ridge": d.mean(),
                 "se": d.std(ddof=1) / np.sqrt(len(d)), "draws better": f"{(d < 0).sum()}/{len(d)}",
                 "folds better": f"{folds}/5", "Brier diff at budgets 0 1 3 7 15 31": budgets})
print(pd.DataFrame(rows).round(4).to_string(index=False))
