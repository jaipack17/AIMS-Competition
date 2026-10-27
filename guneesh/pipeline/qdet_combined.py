"""Embedding + Benedetto's features together (follow-up to pipeline/qdet_upper_bound.py): same benchmarks, same 50/50
split and seeds, same embedding baseline, so the results compare directly with that script's models.

Heuristic features H = ling_paper + read (21; on the question text "q" or the full text "full"), each made
sign(x) log(1 + |x|) (counts have long tails) and standardised with the training half (constant columns dropped).
The embedding Z is standardised with the training half, as in emb_kridge.

  joint[v]       kernel ridge on both blocks: K = Kz + w * Kh, each block's kernel scaled to mean diagonal 1 on the
                 training half, so w is the weight of the 21 features relative to the 4,096 embedding numbers.
                 w in W_GRID and the penalty in LAMS (lam * mean eigenvalue, as emb_kridge) by inner K-fold CV on
                 the training half.  w = 0 is exactly emb_kridge, so CV can always fall back to the embedding.
  correction[v]  emb_kridge with its CV-chosen penalty; out-of-fold predictions on the training half (the same
                 inner folds); a random forest (text2props grid, RandomizedSearchCV) learns the residual
                 y - out-of-fold prediction from H; test prediction = emb_kridge + forest(H).
Also recorded: emb_kridge recomputed here (must equal qdet_upper_bound's value: a check of the shared setup).
Output: data/qdet/combined/<benchmark>.json (resumable), summarised by analysis/qdet_report.py.
Run:  python pipeline/qdet_combined.py [--jobs 2] [--only a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from avgdir_train import RUN, bench_files
from lowrank_dir import informative
from mlp_b_train import keep_awake
from qdet_features import LING_PAPER, READ
from qdet_upper_bound import LAMS, T2P_GRID, metrics, prepare, search
from ridge_train import IRT_DIR

OUT = "data/qdet/combined"
W_GRID = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # 10, 30 added after a pilot benchmark chose the then-largest 3


def standardise(A: np.ndarray, tr: np.ndarray) -> np.ndarray:
    m, s = A[tr].mean(0), A[tr].std(0)
    keep = s > 1e-12
    return (A[:, keep] - m[keep]) / s[keep]


def unit_kernel(A: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Linear kernel of all rows, scaled so its diagonal has mean 1 on the training half (all zeros if A is empty)."""
    K = A @ A.T
    d = np.mean(np.diag(K)[tr]) if A.shape[1] else 0.0
    return K / d if d > 0 else np.zeros_like(K)


def fit_pred(K: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Kernel ridge trained on rows a, predicting rows b; penalty lam * mean eigenvalue (= emb_kridge's formula)."""
    Ka = K[np.ix_(a, a)]
    ya = y[a] - y[a].mean()
    return K[np.ix_(b, a)] @ np.linalg.solve(Ka + lam * np.trace(Ka) / len(a) * np.eye(len(a)), ya) + y[a].mean()


def run_benchmark(name: str, F: pd.DataFrame, emb_path: str, b_final: pd.Series, args) -> dict:
    data = prepare(name, F, emb_path, b_final)
    if data is None:
        return {"benchmark": name, "skipped": "too few items or constant b"}
    F, y, Z, tr, te, seed = data
    res = {"benchmark": name, "n_train": len(tr), "n_test": len(te), "models": {}}
    if not informative(Z):
        res["skipped_models"] = "embedding not informative (as emb_kridge in qdet_upper_bound)"
        return res
    inner = list(KFold(args.cv, shuffle=True, random_state=seed).split(tr))
    Kz = unit_kernel(standardise(Z, tr), tr)

    def cv_mse(K, lam):
        return np.mean([np.mean((fit_pred(K, y, tr[a], tr[b], lam) - y[tr[b]]) ** 2) for a, b in inner])

    t0 = time.time()
    lam_z = min(LAMS, key=lambda lam: cv_mse(Kz, lam))
    base = fit_pred(Kz, y, tr, te, lam_z)
    res["models"]["emb_kridge_check"] = {**metrics(base, y[te]), "lam": lam_z, "sec": round(time.time() - t0, 1)}
    oof = np.empty(len(tr))
    for a, b in inner:
        oof[b] = fit_pred(Kz, y, tr[a], tr[b], lam_z)

    for v in ("q", "full"):
        raw = F[[f"{v}__{c}" for c in LING_PAPER + READ]].to_numpy(np.float64)
        H = standardise(np.sign(raw) * np.log1p(np.abs(raw)), tr)
        t0 = time.time()
        Kh = unit_kernel(H, tr)
        score = {(w, lam): cv_mse(Kz + w * Kh, lam) for w in W_GRID for lam in LAMS}
        w, lam = min(score, key=score.get)
        res["models"][f"joint[{v}]"] = {**metrics(fit_pred(Kz + w * Kh, y, tr, te, lam), y[te]), "w": w, "lam": lam,
                                        "n_features": int(H.shape[1]), "sec": round(time.time() - t0, 1)}
        t0 = time.time()
        if H.shape[1] == 0:  # the features are constant in this benchmark: nothing to correct with
            res["models"][f"correction[{v}]"] = {**metrics(base, y[te]), "no_features": True}
            continue
        s = search(RandomForestRegressor(random_state=seed), T2P_GRID, raw[tr], y[tr] - oof, args, seed)
        res["models"][f"correction[{v}]"] = {**metrics(base + s.predict(raw[te]), y[te]), "cv_r2": float(s.best_score_),
                                             "sec": round(time.time() - t0, 1)}
    return res


def main() -> None:
    keep_awake()
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated benchmarks")
    ap.add_argument("--features", default="data/qdet")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    feats = pd.read_parquet(os.path.join(args.features, "features.parquet"))
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"]).set_index("item").b
    files = bench_files()
    names = args.only.split(",") if args.only else sorted(set(feats.benchmark) & set(files))
    os.makedirs(args.out, exist_ok=True)
    for i, name in enumerate(names):
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            continue
        res = run_benchmark(name, feats[feats.benchmark == name], files[name], b_final, args)
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)
        m = res.get("models", {})
        r = lambda k: m.get(k, {}).get("r", np.nan)
        print(f"[{i + 1}/{len(names)}] {name}: {res.get('skipped', '')}emb r {r('emb_kridge_check'):.3f}, joint q/full "
              f"{r('joint[q]'):.3f}/{r('joint[full]'):.3f}, correction q/full {r('correction[q]'):.3f}/"
              f"{r('correction[full]'):.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
