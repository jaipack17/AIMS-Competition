"""Paired comparison of simulate_alc.py results on the same draws: local ALC (mean over draws of each draw's mean over
its pairs; lower is better), the difference new - baseline with its standard error over draws, draws and folds where
the new one is better, and the Brier difference at each label budget.

  python analysis/paired_alc.py label_update_stretch2_date_probe regress_label_update_stretch2_date \\
      label_update_stretch2_date_mlp

The first name is the new submission; the others are baselines (analysis/simulate_alc_<name>.csv). Every csv must hold
the same configuration, folds, draws and pairs (same seed and data), which is checked.
"""
import sys

import numpy as np
import pandas as pd


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(f"analysis/simulate_alc_{name}.csv")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    new_name, bases = sys.argv[1], sys.argv[2:]
    new = load(new_name)
    rows = []
    for b in bases:
        base = load(b)
        for c in sorted(set(new.config) & set(base.config)):
            m, r = new[new.config == c].reset_index(drop=True), base[base.config == c].reset_index(drop=True)
            key = ["fold", "draw", "pair"]
            if len(m) != len(r) or not (m[key].values == r[key].values).all():
                raise SystemExit(f"{new_name} and {b} ({c}) are not on the same draws/pairs")
            dm, dr = m.groupby(["fold", "draw"]).alc.mean(), r.groupby(["fold", "draw"]).alc.mean()
            d = dm - dr
            folds = int((m.groupby("fold").alc.mean() - r.groupby("fold").alc.mean() < 0).sum())
            rows.append({"baseline": b, "config": c, "baseline_alc": dr.mean(), "new_alc": dm.mean(),
                         "new - base": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                         "draws better": f"{int((d < 0).sum())}/{len(d)}", "folds better": f"{folds}/5",
                         "Brier diff at budgets 0 1 3 7 15 31":
                             " ".join(f"{m[f'b{k}'].mean() - r[f'b{k}'].mean():+.4f}" for k in (0, 1, 3, 7, 15, 31))})
    if not rows:
        raise SystemExit("no configuration in common")
    print(f"new: {new_name}  (lower ALC is better; new - base < 0 means the new one is better)")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
