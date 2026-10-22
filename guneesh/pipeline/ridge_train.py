"""Train the embedding -> IRT difficulty ridge (agent-psychometrics two-stage method, Sec. 3.2.1, Eq. 2).

For every IRT run (beta source) and every fit (fold0..fold4, final):
  training items  embedded items of the binary benchmarks that IRT fit was trained on
                  (fit foldf leaves out fold f's benchmarks; "final" uses all 115)
  target          that fit's item difficulty b (frozen, stage 1)
  model           StandardScaler -> RidgeCV(alphas, cv=GroupKFold(5) by benchmark), every item weighted equally,
                  alpha chosen by mean inner-fold R^2 (sklearn RidgeCV default, as in AP's code),
                  solved exactly by fast_ridge (proof against sklearn: analysis/ridge_vs_sklearn.py)
  alpha grid      AP New Benchmarks grid 0.01..1e5 plus 1e6, 1e7; if the best alpha is the largest one, the grid
                  is extended x10 at a time (up to 1e12) and the fit is redone; the final grid is saved.
Both runs share the same rows, so their sums are built in one pass with two target columns.

Outputs, data/ridge/<run>/<fit>/ (written via temporary files; DONE is written last):
  model.npz      mean, scale, coef, intercept, alpha, alphas, cv_score, cv_mse  (all that model.py needs)
  pred.parquet   item, benchmark, heldout, b_pred  for every embedded item
  summary.json   chosen alpha, CV curve, sizes, held-out sanity check, minutes
  DONE           fits with DONE for both runs are skipped on a restart

The held-out sanity check compares b_pred on fold f's benchmarks with the "final" fit's b for those items.
It only shows the ridge learned something; the real test is the competition-style scoring harness.

Run from the project root:  python pipeline/ridge_train.py [--fits fold0,...,final] [--score r2_within --out_dir data/ridge_within]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fast_ridge import Scaler, fold_sums, ridge_cv

EMB_DIR = "data/emb/qwen3vl8b"
IRT_DIR = "data/irt"
OUT_DIR = "data/ridge"
RUNS = ["versionB_4parts_logitnormal", "versionB_4parts_soft"]
FITS = ["fold0", "fold1", "fold2", "fold3", "fold4", "final"]  # folder names of the IRT fits
# agent-psychometrics experiment_new_benchmarks/config.py ridge_alphas, plus 1e6 and 1e7 (we train on ~200k
# items instead of ~500, so the best penalty can be larger)
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]
MIN_ALPHA = 1e-4  # ... and x0.1 at a time down to this if it sits on the bottom edge
MAX_ALPHA = 1e12  # if the best alpha sits on the top edge, the grid grows x10 at a time up to this


def load_embeddings(emb_dir: str = EMB_DIR, benchmarks=None, cap: int | None = None):
    """All embedding files -> X (n, 4096) float32, item keys "<benchmark>::<item_id>", benchmark per row."""
    X, items, bench = [], [], []
    for path in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))):
        b = os.path.basename(path)[:-4]
        if benchmarks is not None and b not in benchmarks:
            continue
        with np.load(path, allow_pickle=True) as z:
            x, ids = z["X"], z["task_ids"].astype(str)
        if cap is not None:
            x, ids = x[:cap], ids[:cap]
        X.append(x.astype(np.float32, copy=False))
        items += [f"{b}::{i}" for i in ids]
        bench += [b] * len(ids)
    return np.concatenate(X), np.array(items), np.array(bench)


def item_b(run: str, fit: str, items: np.ndarray) -> np.ndarray:
    """b of the given items in one IRT fit; NaN for items that fit did not see."""
    it = pd.read_parquet(os.path.join(IRT_DIR, run, fit, "items.parquet"), columns=["item", "b"])
    return pd.Series(it.b.values, index=it.item.values).reindex(items).to_numpy()


def heldout_check(pred: np.ndarray, truth: np.ndarray, bench: np.ndarray) -> dict:
    ok = ~np.isnan(truth)
    p, t, g = pred[ok], truth[ok], bench[ok]
    per = [np.corrcoef(p[g == b], t[g == b])[0, 1] for b in np.unique(g) if (g == b).sum() >= 3
           and np.std(t[g == b]) > 0 and np.std(p[g == b]) > 0]
    return {"n_items": int(ok.sum()), "pearson_all": float(np.corrcoef(p, t)[0, 1]),
            "pearson_within_benchmark_mean": float(np.mean(per)), "rmse": float(np.sqrt(np.mean((p - t) ** 2)))}


def setup_fit(fit: str, bench: np.ndarray, items: np.ndarray, fold_of: dict, inner_folds: int = 5):
    """Rows of one fit: held-out mask, training rows, targets Y (one column per run), inner fold id per row."""
    heldout = np.array([fold_of[b] == fit for b in bench])
    rows = np.flatnonzero(~heldout)
    Y = np.column_stack([item_b(run, fit, items) for run in RUNS])
    missing = int(np.isnan(Y[rows]).any(1).sum())
    assert missing == 0, f"fit {fit}: {missing} training items have no b"
    splits = list(GroupKFold(n_splits=inner_folds).split(rows, groups=bench[rows]))
    inner = np.empty(len(rows), dtype=int)
    for k, (_, te) in enumerate(splits):
        inner[te] = k
    return heldout, rows, Y, inner, splits


def is_done(out_dir: str, fit: str) -> bool:
    return all(os.path.exists(os.path.join(out_dir, run, fit, "DONE")) for run in RUNS)


def save_fit(out: str, model: dict, pred: pd.DataFrame, summary: dict) -> None:
    """Write model.npz, pred.parquet, summary.json through temporary files, then the DONE marker.
    A crash part-way leaves no DONE, so the fit is simply redone on the next start."""
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "model.npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **model)
    os.replace(tmp, os.path.join(out, "model.npz"))
    tmp = os.path.join(out, "pred.parquet.tmp")
    pred.to_parquet(tmp, index=False)
    os.replace(tmp, os.path.join(out, "pred.parquet"))
    tmp = os.path.join(out, "summary.json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, os.path.join(out, "summary.json"))
    open(os.path.join(out, "DONE"), "w").close()


def predict_all(X: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.concatenate([((X[i:i + 8192].astype(np.float64) - mean) / scale) @ coef
                           for i in range(0, len(X), 8192)]) + intercept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="r2", choices=["r2", "mse", "r2_within"],
                    help="inner-CV selection score; r2_within = R^2 within benchmarks (benchmark means removed)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--fits", default=",".join(FITS))
    ap.add_argument("--inner_folds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    X, items, bench = load_embeddings()
    print(f"embeddings {X.shape} from {len(np.unique(bench))} benchmarks ({time.time() - t0:.0f} s)", flush=True)
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    assert set(np.unique(bench)) == set(fold_of), "embedded benchmarks differ from the 115 in folds.csv"
    b_final = {run: item_b(run, "final", items) for run in RUNS}

    for fit in args.fits.split(","):
        if is_done(args.out_dir, fit):
            print(f"[{fit}] already done, skipped", flush=True)
            continue
        t1 = time.time()
        heldout, rows, Y, inner, _ = setup_fit(fit, bench, items, fold_of, args.inner_folds)
        scaler = Scaler.fit(X, rows)
        sums = fold_sums(X, Y, rows, inner, scaler, groups=bench[rows] if args.score == "r2_within" else None)
        alphas = list(ALPHAS)
        res = ridge_cv(sums, alphas, score=args.score)
        while (res.best == len(alphas) - 1).any() and alphas[-1] < MAX_ALPHA:  # best on the top edge: extend
            alphas.append(alphas[-1] * 10)
            print(f"[{fit}] best alpha on the top edge -> grid extended to {alphas[-1]:g}", flush=True)
            res = ridge_cv(sums, alphas, score=args.score)
        while (res.best == 0).any() and alphas[0] > MIN_ALPHA:  # best on the bottom edge: extend downwards
            alphas.insert(0, alphas[0] / 10)
            print(f"[{fit}] best alpha on the bottom edge -> grid extended to {alphas[0]:g}", flush=True)
            res = ridge_cv(sums, alphas, score=args.score)

        for j, run in enumerate(RUNS):
            alpha = float(res.alphas[res.best[j]])
            b_pred = predict_all(X, scaler.mean, scaler.scale, res.coef[j], res.intercept[j])
            summary = {
                "run": run, "fit": fit, "solver": "fast_ridge float64", "score": args.score, "alpha": alpha,
                "alpha_on_grid_edge": bool(res.best[j] in (0, len(alphas) - 1)),
                "cv": {f"{a:g}": {"score": float(s), "mse": float(m)}
                       for a, s, m in zip(alphas, res.cv_score[:, j], res.cv_mse[:, j])},
                "n_train_items": int(len(rows)), "n_train_benchmarks": int(len(np.unique(bench[rows]))),
                "minutes": round((time.time() - t1) / 60, 2),
            }
            if fit != "final":
                summary["heldout_vs_final_b"] = heldout_check(b_pred[heldout], b_final[run][heldout], bench[heldout])
            model = dict(mean=scaler.mean, scale=scaler.scale, coef=res.coef[j], intercept=res.intercept[j],
                         alpha=alpha, alphas=res.alphas, cv_score=res.cv_score[:, j], cv_mse=res.cv_mse[:, j])
            pred = pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout, "b_pred": b_pred})
            save_fit(os.path.join(args.out_dir, run, fit), model, pred, summary)
            check = summary.get("heldout_vs_final_b", {})
            print(f"[{run} {fit}] alpha={alpha:g}{' (GRID EDGE)' if summary['alpha_on_grid_edge'] else ''} "
                  f"cv {args.score}={res.cv_score[res.best[j], j]:.4f} train={len(rows):,} "
                  + (f"held-out r={check['pearson_all']:.3f} within-bench r={check['pearson_within_benchmark_mean']:.3f} "
                     f"rmse={check['rmse']:.3f}" if check else ""), flush=True)
        print(f"[{fit}] saved in {(time.time() - t1) / 60:.1f} min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
