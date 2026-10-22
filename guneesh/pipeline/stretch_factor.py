"""Option 2: a learned stretch of the ridge's difficulty predictions,  b_tilde = c + k * (b_hat - c).

c = ridge intercept = mean training difficulty (the scaled embeddings have mean 0 on the training rows).
k = within-benchmark calibration slope of a ridge with the FINAL model's alpha, measured on held-out benchmarks:
    for each fold f, train the ridge (same rows, targets and scaler as ridge_train.py) at that alpha, predict the
    fold's held-out benchmarks, centre both the final-fit b and b_hat within each benchmark, and regress
    b_centred on b_hat_centred through the origin. k pools all held-out items of the 5 folds.
Reported as well: k from each set of 4 folds and the within-benchmark MSE it gives on the 5th (vs k = 1), so the
gain is measured on data not used to choose k, and the between-benchmark slope (benchmark means), for reference.

Also written: sigma_delta_stretched, the prior scale of the label-update benchmark shift for stretched predictions
(RMS over held-out benchmarks of mean b - mean b_tilde), which build_submission.py uses with --stretch.

Output: data/ridge/<run>/stretch.json.  Run from the project root:  python pipeline/stretch_factor.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from fast_ridge import Scaler, fold_sums, solve
from ridge_train import IRT_DIR, OUT_DIR, RUNS, item_b, load_embeddings, setup_fit

RUN = RUNS[0]  # versionB_4parts_logitnormal, the run the submissions use


def main() -> None:
    X, items, bench = load_embeddings()
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    alpha = float(np.load(os.path.join(OUT_DIR, RUN, "final", "model.npz"))["alpha"])
    b_final = item_b(RUN, "final", items)
    parts, fold_centres = [], {}
    for f in range(5):
        heldout, rows, Y, _, _ = setup_fit(f"fold{f}", bench, items, fold_of)
        scaler = Scaler.fit(X, rows)
        sums = fold_sums(X, Y[:, :1], rows, np.zeros(len(rows), dtype=int), scaler)[0]
        coef, intercept = solve(sums, np.array([alpha]))
        h = np.flatnonzero(heldout)
        b_hat = scaler.transform(X[h]) @ coef[0, 0] + intercept[0, 0]
        fold_centres[f] = float(intercept[0, 0])  # = that fold's mean training difficulty
        parts.append(pd.DataFrame({"fold": f, "benchmark": bench[h], "b": b_final[h], "b_hat": b_hat}))
        print(f"fold{f}: ridge at alpha={alpha:g} predicted {len(h):,} held-out items", flush=True)
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
    intercept_final = float(np.load(os.path.join(OUT_DIR, RUN, "final", "model.npz"))["intercept"])
    # prior scale of the label-update benchmark shift delta for STRETCHED predictions (as in build_submission.py:
    # RMS over held-out benchmarks of mean b - mean b_tilde); each fold's own centre is its training mean
    stretched = []
    for f in range(5):
        d = P[P.fold == f]
        c_f = float(fold_centres[f])
        stretched.append(d.assign(bt=c_f + k * (d.b_hat - c_f)).groupby("benchmark")[["b", "bt"]].mean())
    S = pd.concat(stretched)
    sigma_delta = float(np.sqrt(((S.b - S.bt) ** 2).mean()))
    out = {"run": RUN, "alpha": alpha, "center": intercept_final, "k": k, "between_benchmark_slope": between,
           "sigma_delta_stretched": sigma_delta,
           "n_items": int(len(P)), "held_out_checks": held_out_checks}
    with open(os.path.join(OUT_DIR, RUN, "stretch.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"k = {k:.3f} (pooled over {len(P):,} held-out items); between-benchmark slope {between:.3f}; "
          f"center c = {intercept_final:.4f}; sigma_delta for stretched b_hat {sigma_delta:.3f}")


if __name__ == "__main__":
    main()
