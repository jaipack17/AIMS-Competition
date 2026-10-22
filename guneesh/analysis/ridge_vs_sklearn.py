"""Proof that pipeline/fast_ridge.py gives the same model as scikit-learn StandardScaler -> RidgeCV.

Setup (real data, small enough for sklearn): 30 random benchmarks (seed 0), at most 500 embedded items each;
24 benchmarks train the ridge, 6 are held out; target = logitnormal "final" IRT b. Inner CV = GroupKFold(5) by
benchmark, alphas = the grid ridge_train.py uses. For both selection scores (R^2 = RidgeCV default, MSE) it compares
  - chosen alpha, the CV score of every alpha (sklearn GridSearchCV cv_results_ vs fast_ridge)
  - coefficients, intercept, predictions on the held-out benchmarks
sklearn runs in float64 (exact reference) and in float32 (what AP's code does with float32 embeddings; differences
there are float32 rounding, not a different method). Times are printed for every path.

Run from the project root:  python analysis/ridge_vs_sklearn.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from fast_ridge import Scaler, fold_sums, ridge_cv  # noqa: E402
from ridge_train import ALPHAS, load_embeddings, item_b  # noqa: E402


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def main() -> None:
    names = sorted(f[:-4] for f in os.listdir("data/emb/qwen3vl8b") if f.endswith(".npz"))
    rng = np.random.default_rng(0)
    pick = list(rng.choice(names, 30, replace=False))
    X, items, bench = load_embeddings(benchmarks=set(pick), cap=500)
    y = item_b("versionB_4parts_logitnormal", "final", items)
    test = np.isin(bench, pick[24:])
    tr = np.flatnonzero(~test)
    print(f"train {len(tr):,} items / 24 benchmarks, held-out {test.sum():,} items / 6 benchmarks, dim {X.shape[1]}")
    splits = list(GroupKFold(n_splits=5).split(tr, groups=bench[tr]))
    inner = np.empty(len(tr), dtype=int)
    for k, (_, te) in enumerate(splits):
        inner[te] = k

    ok = True
    for score, sk_scoring in [("r2", None), ("mse", "neg_mean_squared_error")]:
        print(f"\n=== selection score: {score} ===")
        t = time.time()
        sc = Scaler.fit(X, tr)
        res = ridge_cv(fold_sums(X, y[:, None], tr, inner, sc), ALPHAS, score=score)
        pred_fast = sc.transform(X[test]) @ res.coef[0] + res.intercept[0]
        t_fast = time.time() - t

        for dtype in (np.float64, np.float32):
            t = time.time()
            ss = StandardScaler().fit(X[tr].astype(dtype))
            Ztr = ss.transform(X[tr].astype(dtype))
            m = RidgeCV(alphas=ALPHAS, cv=splits, scoring=sk_scoring).fit(Ztr, y[tr].astype(dtype))
            pred_sk = m.predict(ss.transform(X[test].astype(dtype)))
            t_sk = time.time() - t
            print(f"[sklearn {np.dtype(dtype).name}] alpha={m.alpha_:g}  time {t_sk:.1f} s   |  fast_ridge alpha="
                  f"{res.alphas[res.best[0]]:g}  time {t_fast:.1f} s")
            same_alpha = m.alpha_ == res.alphas[res.best[0]]
            d_coef, d_pred = rel(res.coef[0], m.coef_), rel(pred_fast, pred_sk)
            d_int = abs(res.intercept[0] - m.intercept_)
            print(f"   same alpha: {same_alpha}   max rel diff coef {d_coef:.1e}   intercept abs diff {d_int:.1e}   "
                  f"held-out pred rel diff {d_pred:.1e}")
            if dtype is np.float64:
                gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=splits, scoring=sk_scoring).fit(Ztr, y[tr])
                sk_cv = gs.cv_results_["mean_test_score"]
                d_cv = np.max(np.abs(sk_cv - res.cv_score[:, 0]))
                print("   CV score per alpha (sklearn | fast):")
                for a, s1, s2 in zip(ALPHAS, sk_cv, res.cv_score[:, 0]):
                    print(f"     {a:>8g}  {s1: .10f} | {s2: .10f}")
                print(f"   max abs diff of CV scores {d_cv:.1e}")
                ok &= same_alpha and d_coef < 1e-6 and d_pred < 1e-6 and d_cv < 1e-8
    print("\nRESULT:", "fast_ridge MATCHES sklearn RidgeCV (float64)" if ok else "MISMATCH - do not use fast_ridge")


if __name__ == "__main__":
    main()
