"""Stage 2 for MIRT fits: ridge from the Qwen3-VL-8B item embedding to every item parameter of a MIRT fit.

Same method as pipeline/ridge_train.py (AP two-stage, StandardScaler -> RidgeCV with GroupKFold(5) by benchmark,
AP alpha grid extended at the edges, exact fast_ridge solver), applied to K + 1 targets per MIRT fit:
  nested  fit:  b, v1..vK        m2pl fit:  b, a1..aK
Each target gets its own penalty (ridge_cv selects per column). All requested MIRT runs of one fold share the same
training rows (embedded items of the fold's training benchmarks), so the expensive sums over the 4096-dim embeddings
are built once per fold and every run's targets are solved together.

Rotation: the latent space of each fit is arbitrary up to a rotation; a linear ridge per column is equivariant
(rotating the targets rotates the predictions), so predictions of u . v_hat do not depend on it.

Outputs, data/ridge_mirt/<run>/<fit>/ (temporary files, DONE last):
  model.npz     mean, scale, coef (T x 4096), intercept (T), alpha (T), targets (T names), cv_score, cv_mse
  pred.parquet  item, benchmark, heldout, b_pred, v1_pred.. (or a1_pred..) for every embedded item
  summary.json  per target: alpha, edge flag, held-out check vs the same fit's... (see below), minutes
The held-out check compares predictions on fold f's benchmarks with the FINAL 1D fit's b (for b only; the MIRT
loadings of held-out items exist only in a final MIRT fit) -- a sanity check, the real test is analysis/simulate_alc.

Run:  python pipeline/mirt_ridge.py --runs mirt_nested_K1,mirt_nested_K2 [--fits fold0,...] [--score r2]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from fast_ridge import Scaler, fold_sums, ridge_cv
from ridge_train import ALPHAS, IRT_DIR, MAX_ALPHA, MIN_ALPHA, heldout_check, load_embeddings, predict_all, save_fit, \
    setup_fit

OUT_DIR = "data/ridge_mirt"


def targets_of(run: str, fit: str, items: np.ndarray) -> tuple[list[str], np.ndarray]:
    it = pd.read_parquet(os.path.join(IRT_DIR, run, fit, "items.parquet"))
    cols = ["b"] + sorted([c for c in it.columns if c[0] in "va" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    return cols, it.set_index("item")[cols].reindex(items).to_numpy(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated MIRT runs in data/irt, e.g. mirt_nested_K2")
    ap.add_argument("--fits", default="fold0,fold1,fold2,fold3,fold4")
    ap.add_argument("--score", default="r2", choices=["r2", "mse", "r2_within"])
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    runs = args.runs.split(",")

    t0 = time.time()
    X, items, bench = load_embeddings()
    print(f"embeddings {X.shape} ({time.time() - t0:.0f} s)", flush=True)
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    b_final_1d = pd.read_parquet(os.path.join(IRT_DIR, "versionB_4parts_logitnormal", "final", "items.parquet"),
                                 columns=["item", "b"]).set_index("item").b.reindex(items).to_numpy()

    for fit in args.fits.split(","):
        todo = [r for r in runs if not os.path.exists(os.path.join(args.out_dir, r, fit, "DONE"))]
        if not todo:
            print(f"[{fit}] all runs done, skipped", flush=True)
            continue
        t1 = time.time()
        heldout, rows, _, inner, _ = setup_fit(fit, bench, items, fold_of)   # rows / inner folds as ridge_train
        names, cols = [], []
        for r in todo:
            tn, Y = targets_of(r, fit, items)
            assert not np.isnan(Y[rows]).any(), f"{r} {fit}: training items without parameters"
            names += [(r, t) for t in tn]
            cols.append(Y)
        Y = np.column_stack(cols)
        scaler = Scaler.fit(X, rows)
        sums = fold_sums(X, Y, rows, inner, scaler, groups=bench[rows] if args.score == "r2_within" else None)
        alphas = list(ALPHAS)
        res = ridge_cv(sums, alphas, score=args.score)
        while (res.best == len(alphas) - 1).any() and alphas[-1] < MAX_ALPHA:
            alphas.append(alphas[-1] * 10)
            res = ridge_cv(sums, alphas, score=args.score)
        while (res.best == 0).any() and alphas[0] > MIN_ALPHA:
            alphas.insert(0, alphas[0] / 10)
            res = ridge_cv(sums, alphas, score=args.score)
        print(f"[{fit}] sums + CV for {len(names)} targets of {len(todo)} runs in {(time.time() - t1) / 60:.1f} min",
              flush=True)

        for r in todo:
            js = [j for j, (rr, _) in enumerate(names) if rr == r]
            tn = [names[j][1] for j in js]
            pred = pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout})
            per = {}
            for j, t in zip(js, tn):
                p = predict_all(X, scaler.mean, scaler.scale, res.coef[j], res.intercept[j])
                pred[f"{t}_pred"] = p
                truth = Y[:, j]
                tr = rows                                  # in-sample fit quality on the training items
                per[t] = {"alpha": float(res.alphas[res.best[j]]),
                          "alpha_on_grid_edge": bool(res.best[j] in (0, len(alphas) - 1)),
                          "cv_score": float(res.cv_score[res.best[j], j]),
                          "train_pearson": float(np.corrcoef(p[tr], truth[tr])[0, 1])}
                if t == "b" and fit != "final":
                    per[t]["heldout_vs_final_1d_b"] = heldout_check(p[heldout], b_final_1d[heldout], bench[heldout])
            model = dict(mean=scaler.mean, scale=scaler.scale, coef=res.coef[js], intercept=res.intercept[js],
                         alpha=res.alphas[res.best[js]], targets=np.array(tn), alphas=res.alphas,
                         cv_score=res.cv_score[:, js], cv_mse=res.cv_mse[:, js])
            summary = {"run": r, "fit": fit, "score": args.score, "targets": per, "n_train_items": int(len(rows)),
                       "minutes": round((time.time() - t1) / 60, 2)}
            save_fit(os.path.join(args.out_dir, r, fit), model, pred, summary)
            print(f"[{r} {fit}] " + "  ".join(f"{t}: alpha {v['alpha']:g} cv {v['cv_score']:.3f}" for t, v in per.items())
                  + (f"  | b held-out within-bench r {per['b']['heldout_vs_final_1d_b']['pearson_within_benchmark_mean']:.3f}"
                     if "heldout_vs_final_1d_b" in per["b"] else ""), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
