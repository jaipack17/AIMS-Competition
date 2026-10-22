"""Full-scale check: the same ridge fits as ridge_train.py, done with scikit-learn exactly like agent-psychometrics.

AP path (experiment_new_tasks/feature_predictor.py FeatureBasedPredictor): StandardScaler -> RidgeCV(alphas, cv=...),
float32 embeddings in, so sklearn computes in float32. Everything else is identical to ridge_train.py, so any
difference is float32 rounding only: the same training rows, the same inner GroupKFold(5) splits by benchmark, the
same R^2 selection (RidgeCV default), and for each (run, fit) the final alpha grid the fast run used (including
any automatic extension). That is why this script needs the fast run's DONE fit first.

Outputs, data/ridge_sklearn/<run>/<fit>/: model.npz, pred.parquet, summary.json (+ comparison with fast_ridge), DONE.
Finished fits are skipped on a restart.

Run from the project root, after ridge_train.py:  python pipeline/ridge_sklearn.py [--fits fold0,fold1,fold2,fold3,fold4,final]
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ridge_train import (FITS, IRT_DIR, OUT_DIR as FAST_DIR, RUNS, heldout_check, is_done, item_b, load_embeddings,
                         save_fit, setup_fit)

OUT_DIR = "data/ridge_sklearn"
# float32 Cholesky at the smallest alphas is ill-conditioned; sklearn warns once per fit (thousands of lines).
# Those alphas are never chosen; the warning is silenced only to keep the log readable.
warnings.filterwarnings("ignore", category=LinAlgWarning)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=",".join(FITS))
    args = ap.parse_args()

    t0 = time.time()
    X, items, bench = load_embeddings()
    print(f"embeddings {X.shape} ({time.time() - t0:.0f} s)", flush=True)
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    b_final = {run: item_b(run, "final", items) for run in RUNS}

    for fit in args.fits.split(","):
        if is_done(OUT_DIR, fit):
            print(f"[{fit}] already done, skipped", flush=True)
            continue
        if not is_done(FAST_DIR, fit):
            raise SystemExit(f"[{fit}] fast ridge fit not finished yet; run ridge_train.py first")
        t1 = time.time()
        heldout, rows, Y, _, splits = setup_fit(fit, bench, items, fold_of)
        ss = StandardScaler().fit(X[rows])
        Z = ss.transform(X[rows])                                       # float32, like AP
        for j, run in enumerate(RUNS):
            out = os.path.join(OUT_DIR, run, fit)
            if os.path.exists(os.path.join(out, "DONE")):
                continue
            t2 = time.time()
            fast = np.load(os.path.join(FAST_DIR, run, fit, "model.npz"))
            alphas = [float(a) for a in fast["alphas"]]
            m = RidgeCV(alphas=alphas, cv=splits).fit(Z, Y[rows, j].astype(np.float32))
            b_pred = np.concatenate([m.predict(ss.transform(X[i:i + 8192])) for i in range(0, len(X), 8192)])
            fast_pred = pd.read_parquet(os.path.join(FAST_DIR, run, fit, "pred.parquet"), columns=["b_pred"]).b_pred
            fast_pred = fast_pred.to_numpy()
            compare = {
                "same_alpha": bool(float(m.alpha_) == float(fast["alpha"])),
                "alpha_sklearn": float(m.alpha_), "alpha_fast": float(fast["alpha"]),
                "coef_max_rel_diff": float(np.max(np.abs(m.coef_ - fast["coef"])) / np.max(np.abs(fast["coef"]))),
                "intercept_abs_diff": float(abs(m.intercept_ - fast["intercept"])),
                "pred_max_abs_diff": float(np.max(np.abs(b_pred - fast_pred))),
                "pred_pearson": float(np.corrcoef(b_pred, fast_pred)[0, 1]),
            }
            summary = {"run": run, "fit": fit, "solver": "sklearn RidgeCV float32", "score": "r2",
                       "alpha": float(m.alpha_), "alphas": alphas, "best_score": float(m.best_score_),
                       "n_train_items": int(len(rows)), "vs_fast_ridge": compare,
                       "minutes": round((time.time() - t2) / 60, 2)}
            if fit != "final":
                summary["heldout_vs_final_b"] = heldout_check(b_pred[heldout], b_final[run][heldout], bench[heldout])
            model = dict(mean=ss.mean_, scale=ss.scale_, coef=m.coef_, intercept=m.intercept_, alpha=m.alpha_,
                         alphas=np.array(alphas))
            pred = pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout, "b_pred": b_pred})
            save_fit(out, model, pred, summary)
            print(f"[{run} {fit}] sklearn alpha={m.alpha_:g} (fast {float(fast['alpha']):g}) "
                  f"coef rel diff {compare['coef_max_rel_diff']:.1e} pred max diff {compare['pred_max_abs_diff']:.1e} "
                  f"r={compare['pred_pearson']:.8f}  {summary['minutes']} min", flush=True)
        del Z
        print(f"[{fit}] saved in {(time.time() - t1) / 60:.1f} min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
