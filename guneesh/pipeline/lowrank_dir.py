"""Shared low-rank difficulty subspace across benchmarks: b_j = c_m + s_m * a_m' P z_j.

Idea (item-side analogue of observational scaling laws, Ruan, Maddison & Hashimoto 2024): every benchmark m ranks its
items along its own direction w_m in embedding space; we ask whether all these directions lie in one small shared
subspace P (k x d), so that a new benchmark is described by only k numbers a_m (its mix of the shared axes), which a
few labelled items can calibrate. c_m and s_m (level and spread) do not change the ranking inside a benchmark, so this
script measures ranking only (within-benchmark Pearson r); they matter later for calibrated probabilities.

For each fold (training benchmarks only; the fold's 23 held-out benchmarks are only evaluated):
  1. z = embedding standardised with the training rows; per training benchmark the unit ridge direction w_m of the
     fold's 1D IRT b on z (kernel ridge, alpha = 1e3: the value step B chose on its selection benchmarks in all 5 folds;
     same code, avgdir_train.directions).
  2. PCA of the stacked directions W (benchmarks x d) with ObsScaling's perform_pca (copied below). Basis for k:
     orthonormalised [mean direction, PC1, ..., PC(k-1)], so k = 1 is the plain average direction (step B uniform).
  3. A held-out benchmark is calibrated from n items with known b (n = 10, 30, 100, 300, half): ridge of b on the k
     features P z (ours, every k), against kernel ridge on the full embedding with the same n items (no sharing,
     "scratch") and the k = 1 average direction without labels ("zero").  r is measured on the benchmark's other items.
  Penalties (and the headline k) are chosen on 20% of the training benchmarks (seed 0, as in avgdir_train / mlp_b_train)
  with P built from the other training benchmarks; then P is rebuilt from all training benchmarks for the held-out run.

Truth: fold IRT b for training (selection) benchmarks, final IRT b for held-out benchmarks (as in all earlier checks).
Outputs: data/lowrank/<RUN>/<fit>/: basis.npz, selection.csv, heldout.csv, summary.json, DONE (per fold; reruns skip
finished folds).   Run from the project root:  python pipeline/lowrank_dir.py [--fits fold0,...]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler

from avgdir_train import ALPHAS, CAP, MIN_ITEMS, RUN, bench_files, directions, load_bench, scaler
from mlp_b_train import keep_awake
from ridge_train import IRT_DIR

ALPHA = 1e3
NS = (10, 30, 100, 300, "half")
DRAWS = 10
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
K_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90)  # capped at the number of directions
OUT = "data/lowrank"


# ---- copied verbatim from ObsScaling (github.com/ryoungj/ObsScaling, utils/data.py; Apache License 2.0) ----
def perform_pca(train_df, test_df=None, n_components=2, standardize=False, sparse=False, alpha=1, **kwargs):
    """Apply PCA to the data and extract the principal components

    Args:
        train_df: A pandas DataFrame with the training data.
        test_df: A pandas DataFrame with the test data.
        n_components: Number of principal components to keep.
        standardize: Whether to standardize the data before applying PCA.
        sparse: Whether to use SparsePCA instead of PCA.
        alpha: Sparsity controlling parameter for SparsePCA.
        kwargs: Additional keyword arguments for PCA or SparsePCA.
    """
    if standardize:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df) if test_df is not None else None
    else:
        train_scaled = train_df
        test_scaled = test_df

    if sparse:
        pca = SparsePCA(n_components=n_components, alpha=alpha, **kwargs)
    else:
        pca = PCA(n_components=n_components, **kwargs)

    # Fit PCA on the training data
    pca.fit(train_scaled)

    # Transform the training data
    train_pc = pca.transform(train_scaled)
    train_principal_df = pd.DataFrame(data=train_pc,
                                      columns=[f'PC-{i}' for i in range(1, n_components+1)],
                                      index=train_df.index)

    # Transform the test data
    if test_df is not None:
        test_pc = pca.transform(test_scaled)
        test_principal_df = pd.DataFrame(data=test_pc,
                                        columns=[f'PC-{i}' for i in range(1, n_components+1)],
                                        index=test_df.index)
    else:
        test_principal_df = None

    return pca, train_principal_df, test_principal_df
# ---- end of copied code ----


def basis(dirs: dict) -> tuple[np.ndarray, PCA]:
    """Orthonormal nested basis Q (K x d): row 0 = mean direction, then the principal components of the directions."""
    W = pd.DataFrame(np.stack(list(dirs.values())), index=list(dirs))
    pca, _, _ = perform_pca(W, n_components=len(W) - 1, standardize=False)  # same settings as ObsScaling's default
    Q, _ = np.linalg.qr(np.vstack([pca.mean_, pca.components_]).T)  # Gram-Schmidt in this order: nested in k
    Q = Q.T * np.sign(np.diag(Q.T @ np.vstack([pca.mean_, pca.components_]).T))[:, None]
    return Q, pca


def informative(Z: np.ndarray) -> bool:
    """False when every item of a benchmark has the same embedding (e.g. visual_memory): text carries no signal there."""
    return bool(Z.std(0).max() > 1e-8)


def ridge_fit(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with intercept, penalty lam * mean feature variance * n (scale-free): returns the slope vector."""
    Fc, yc = F - F.mean(0), y - y.mean()
    G = Fc.T @ Fc
    return np.linalg.solve(G + lam * np.trace(G) / F.shape[1] * np.eye(F.shape[1]), Fc.T @ yc)


def r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


def evaluate(Z: np.ndarray, y: np.ndarray, Q: np.ndarray, seed: int) -> list[dict]:
    """One benchmark: r on the non-calibration items for every (method, k, n, lam), averaged over draws."""
    F = Z @ Q.T                                            # features for every k at once (nested basis)
    rows, rng = [], np.random.default_rng(seed)
    rows.append({"method": "zero", "k": 1, "n": "0", "lam": np.nan, "r": r(F[:, 0], y)})
    for n in NS:
        n_cal = len(y) // 2 if n == "half" else n
        if n_cal < 3 or n_cal > len(y) // 2:
            continue
        acc = {}
        for _ in range(DRAWS if n != "half" else 3):
            perm = rng.permutation(len(y))
            S, E = perm[:n_cal], perm[n_cal:]
            m = Z[S].mean(0)
            ZS, ZE = Z[S] - m, Z[E] - m
            K, KE = ZS @ ZS.T, ZE @ ZS.T
            ys = y[S] - y[S].mean()
            for lam in LAMS:
                coef = np.linalg.solve(K + lam * np.trace(K) / n_cal * np.eye(n_cal), ys)
                acc.setdefault(("scratch", 0, lam), []).append(r(KE @ coef, y[E]))
                for k in sorted({k for k in K_GRID if k <= Q.shape[0]} | {Q.shape[0]}):
                    acc.setdefault(("ours", k, lam), []).append(r(F[E, :k] @ ridge_fit(F[S, :k], y[S], lam), y[E]))
        rows += [{"method": mt, "k": k, "n": str(n), "lam": lam, "r": float(np.nanmean(v))} for (mt, k, lam), v in acc.items()]
    return rows


def run_benchmarks(names, files, mean, sd, truth, Q, tag, log) -> pd.DataFrame:
    out = []
    for i, b in enumerate(names):
        X, items = load_bench(files[b])
        y = truth.reindex(items).to_numpy()
        ok = ~np.isnan(y)
        Z = (X[ok] - mean) / sd
        if ok.sum() < MIN_ITEMS or np.std(y[ok]) == 0 or not informative(Z):
            log(f"  [{tag}] {b} skipped (too few items, constant b or identical embeddings)")
            continue
        rows = evaluate(Z, y[ok], Q, seed=i)
        out += [{"benchmark": b, "n_items": int(ok.sum()), **row} for row in rows]
        log(f"  [{tag}] {i + 1}/{len(names)} {b} ({ok.sum()} items)")
    return pd.DataFrame(out)


def main() -> None:
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="fold0,fold1,fold2,fold3,fold4")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"{m}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    files = bench_files()
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"]).set_index("item").b
    j_alpha = ALPHAS.index(ALPHA)

    for fit in args.fits.split(","):
        out = os.path.join(OUT, RUN, fit)
        if os.path.exists(os.path.join(out, "DONE")):
            log(f"[{fit}] already done, skipped")
            continue
        os.makedirs(out, exist_ok=True)
        train_b = sorted(b for b in files if fold_of[b] != fit)
        held_b = sorted(b for b in files if fold_of[b] == fit)
        sel_b = set(np.random.default_rng(0).choice(train_b, size=round(0.2 * len(train_b)), replace=False))
        b_fit = pd.read_parquet(os.path.join(IRT_DIR, RUN, fit, "items.parquet"), columns=["item", "b"]).set_index("item").b
        mean, sd = scaler(files, train_b)

        dirs = {}
        for b in train_b:
            X, items = load_bench(files[b])
            Z, y = (X - mean) / sd, b_fit.reindex(items).to_numpy()
            ok = ~np.isnan(y)
            if ok.sum() >= MIN_ITEMS and np.std(y[ok]) > 0 and informative(Z[ok]):
                sub = np.random.default_rng(0).permutation(int(ok.sum()))[:CAP]
                dirs[b] = directions(Z[ok][sub], y[ok][sub])[j_alpha]
        log(f"[{fit}] {len(dirs)} training directions")

        # choices (penalty per method / k / n, headline k per n) on the selection benchmarks, P without them
        sel_path = os.path.join(out, "selection.csv")
        if os.path.exists(sel_path):
            sel = pd.read_csv(sel_path, dtype={"n": str})
        else:
            Q_base, _ = basis({b: w for b, w in dirs.items() if b not in sel_b})
            sel = run_benchmarks(sorted(sel_b), files, mean, sd, b_fit, Q_base, f"{fit} selection", log)
            sel.to_csv(sel_path, index=False)
        g = sel.groupby(["method", "k", "n", "lam"], dropna=False).r.mean().reset_index()
        g = g[g.method != "zero"]
        best_lam = g.loc[g.groupby(["method", "k", "n"]).r.idxmax()]
        ours = best_lam[best_lam.method == "ours"]
        best_k = ours.loc[ours.groupby("n").r.idxmax()].set_index("n").k.to_dict()
        log(f"[{fit}] selection: best k per n {best_k}")

        Q, pca = basis(dirs)
        names = list(dirs)
        np.savez(os.path.join(out, "basis.npz"), Q=Q, pca_mean=pca.mean_, components=pca.components_,
                 explained_variance_ratio=pca.explained_variance_ratio_, coords=np.stack([dirs[b] for b in names]) @ Q.T,
                 benchmarks=np.array(names), scaler_mean=mean, scaler_sd=sd, alpha=ALPHA)
        held = run_benchmarks(held_b, files, mean, sd, b_final, Q, f"{fit} held-out", log)
        keyed = held.merge(best_lam[["method", "k", "n", "lam"]], on=["method", "k", "n", "lam"], how="inner")
        keyed = pd.concat([keyed, held[held.method == "zero"]])
        keyed["fold"] = fit
        keyed.to_csv(os.path.join(out, "heldout.csv"), index=False)

        head = {}
        for n, k in best_k.items():
            o = keyed[(keyed.method == "ours") & (keyed.n == n) & (keyed.k == k)].r
            s = keyed[(keyed.method == "scratch") & (keyed.n == n)].r
            head[n] = {"k": int(k), "ours_r": float(o.mean()), "scratch_r": float(s.mean()), "benchmarks": int(o.notna().sum())}
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"run": RUN, "fit": fit, "alpha": ALPHA, "n_directions": len(dirs), "draws": DRAWS,
                       "zero_r": float(keyed[keyed.method == "zero"].r.mean()), "headline": head,
                       "explained_variance_ratio_first10": pca.explained_variance_ratio_[:10].tolist(),
                       "minutes": (time.time() - t0) / 60}, f, indent=1)
        open(os.path.join(out, "DONE"), "w").close()
        log(f"[{fit}] held-out: zero {keyed[keyed.method == 'zero'].r.mean():.3f}; "
            + "; ".join(f"n={n}: ours(k={v['k']}) {v['ours_r']:.3f} vs scratch {v['scratch_r']:.3f}" for n, v in head.items()))
    log("ALL DONE")


if __name__ == "__main__":
    main()
