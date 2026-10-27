"""Option 2 (the learned stretch b_tilde = c + k * (b_hat - c), see stretch_factor.py) for the MLP's predictions.

The ridge version retrains each fold's ridge at the final alpha; the MLP has no such hyper-parameter, so its own
held-out fold predictions (data/mlp_b/<run>/fold*/pred.parquet, never trained on those benchmarks) are used as they are.
c for fit f = mean b of fit f over its training items (= the "center" bundled in that fit's mlp.npz; the ridge's
intercept plays this role).  k = within-benchmark calibration slope pooled over the held-out items of the 5 folds,
with each fold's k checked on that fold from the other 4.  sigma_delta_stretched as in stretch_factor.py.

Output: data/mlp_b/<run>/stretch.json, and data/mlp_b/<run>/fold*/center.json (read by analysis/simulate_alc.py).
Run from the project root:  python pipeline/stretch_factor_mlp.py [--dir data/mlp_b]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ridge_train import IRT_DIR, RUNS

RUN = RUNS[0]


def fit_center(fit: str, pred: pd.DataFrame) -> float:
    b = pd.read_parquet(os.path.join(IRT_DIR, RUN, fit, "items.parquet"), columns=["item", "b"]).set_index("item").b
    y = b.reindex(pred.loc[~pred.heldout, "item"]).to_numpy()
    assert not np.isnan(y).any()
    return float(y.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/mlp_b")
    args = ap.parse_args()
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"])
    parts, centres = [], {}
    for f in range(5):
        pred = pd.read_parquet(os.path.join(args.dir, RUN, f"fold{f}", "pred.parquet"))
        centres[f] = fit_center(f"fold{f}", pred)
        with open(os.path.join(args.dir, RUN, f"fold{f}", "center.json"), "w") as fh:
            json.dump({"center": centres[f], "what": "mean training b of this fit"}, fh, indent=1)
        h = pred[pred.heldout].merge(b_final, on="item")
        parts.append(pd.DataFrame({"fold": f, "benchmark": h.benchmark, "b": h.b, "b_hat": h.b_pred}))
    P = pd.concat(parts, ignore_index=True)
    P["bc"] = P.b - P.groupby(["fold", "benchmark"]).b.transform("mean")
    P["pc"] = P.b_hat - P.groupby(["fold", "benchmark"]).b_hat.transform("mean")
    slope = lambda d: float((d.bc * d.pc).sum() / (d.pc ** 2).sum())
    k = slope(P)
    held_out_checks = []
    for f in range(5):
        k_f = slope(P[P.fold != f])
        d = P[P.fold == f]
        mse1, mse_k = float(((d.bc - d.pc) ** 2).mean()), float(((d.bc - k_f * d.pc) ** 2).mean())
        held_out_checks.append({"fold": f, "k_from_other_folds": k_f, "within_mse_k1": mse1, "within_mse_k": mse_k})
        print(f"fold{f}: k from other folds {k_f:.2f}; within-benchmark MSE on this fold {mse1:.4f} (k=1) -> "
              f"{mse_k:.4f} (stretched)")
    g = P.groupby(["fold", "benchmark"])[["b", "b_hat"]].mean()
    gc = g - g.mean()
    between = float((gc.b * gc.b_hat).sum() / (gc.b_hat ** 2).sum())
    stretched = [d.assign(bt=centres[f] + k * (d.b_hat - centres[f])).groupby("benchmark")[["b", "bt"]].mean()
                 for f, d in P.groupby("fold")]
    S = pd.concat(stretched)
    sigma_delta = float(np.sqrt(((S.b - S.bt) ** 2).mean()))
    final_mlp = os.path.join(args.dir, RUN, "final", "mlp.npz")
    if os.path.exists(final_mlp):
        with np.load(final_mlp) as z:
            center = float(z["center"])
    else:  # before the final fit is trained: the same number, from the final IRT fit's b of all embedded items
        pred0 = pd.read_parquet(os.path.join(args.dir, RUN, "fold0", "pred.parquet"), columns=["item"])
        center = float(b_final.set_index("item").b.reindex(pred0["item"]).mean())
    out = {"run": RUN, "source": args.dir, "center": center, "k": k, "between_benchmark_slope": between,
           "sigma_delta_stretched": sigma_delta, "fold_centers": centres, "n_items": int(len(P)),
           "held_out_checks": held_out_checks}
    with open(os.path.join(args.dir, RUN, "stretch.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"k = {k:.3f} (pooled over {len(P):,} held-out items); between-benchmark slope {between:.3f}; "
          f"center c = {center:.4f}; sigma_delta for stretched b_hat {sigma_delta:.3f}")


if __name__ == "__main__":
    main()
