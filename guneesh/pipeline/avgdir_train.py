"""Difficulty from per-benchmark directions: fit a difficulty direction INSIDE each training benchmark, then combine.

Why (teammate's analysis, 24 Oct 2026): every benchmark encodes difficulty along its own direction of the embedding
space and these directions are nearly orthogonal, so one pooled ridge fits benchmark-specific structure; the plain
average of the per-benchmark directions transferred better to unseen benchmarks (0.198 vs 0.156 Spearman).

For each fold (training benchmarks only; the fold's held-out benchmarks are only predicted):
  z = embedding standardised with the fold's training rows (StandardScaler).
  Per training benchmark m (>= MIN_ITEMS embedded items, at most CAP of them): centre z and the fold's 1D IRT b inside the benchmark and
  fit a ridge direction w_m(alpha) = Zc' V diag(1 / (e + alpha)) V' bc  (kernel form, eigen-decomposition of Zc Zc':
  every alpha from one decomposition); unit-normalise it.
  Two ways of combining the directions:
    uniform  w = mean over benchmarks of the unit directions                      (teammate's "plain average")
    simw     w(x) = sum_m softmax_m(cos(z, mu_m) / tau) * unit direction of m      (the item's own embedding says
             which training benchmarks it resembles: mu_m = mean of benchmark m's standardised embeddings)
  score(x) = z . w (or z . w(x)); b_pred = c0 + c1 * score, (c0, c1) by least squares on the training items.
Choices made on 20% of the training benchmarks never used for directions (seed 0, the same benchmarks as
mlp_b_train): alpha (uniform), alpha and tau (simw), by the mean within-benchmark Pearson r of score vs b.
The final directions are then refitted on all training benchmarks with the chosen values.

Memory: embeddings are read one benchmark file at a time (no 4 GB array).
Outputs: data/avgdir_uniform/<RUN>/<fit>/ and data/avgdir_simw/<RUN>/<fit>/: pred.parquet (item, benchmark,
heldout, b_pred), summary.json, DONE.     Run from the project root:  python pipeline/avgdir_train.py [--fits ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from mlp_b_train import keep_awake, within_r
from ridge_train import EMB_DIR, IRT_DIR, heldout_check

RUN = "versionB_4parts_logitnormal"
MIN_ITEMS = 30
CAP = 2000          # items per benchmark used to fit its direction (random subset, seed 0): eigh of n x n stays cheap
ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)
TAUS = (0.02, 0.05, 0.1, 0.2, 0.5)


def bench_files() -> dict:
    return {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(EMB_DIR, "*.npz")))}


def load_bench(path: str):
    with np.load(path, allow_pickle=True) as z:
        b = os.path.basename(path)[:-4]
        return z["X"].astype(np.float64), np.array([f"{b}::{i}" for i in z["task_ids"].astype(str)])


def scaler(files: dict, benches: list):
    n, s, ss = 0, 0.0, 0.0
    for b in benches:
        X, _ = load_bench(files[b])
        n, s, ss = n + len(X), s + X.sum(0), ss + (X ** 2).sum(0)
    mean = s / n
    sd = np.sqrt(np.maximum(ss / n - mean ** 2, 0))
    return mean, np.where(sd > 0, sd, 1.0)


def directions(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit ridge directions for every alpha in ALPHAS from centred Z (n x d) and centred y: (len(ALPHAS), d)."""
    Zc, yc = Z - Z.mean(0), y - y.mean()
    e, V = np.linalg.eigh(Zc @ Zc.T)
    Vy = V.T @ yc
    W = np.stack([Zc.T @ (V @ (Vy / (e + a))) for a in ALPHAS])
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def simw_scores(Z: np.ndarray, Mu: np.ndarray, Wd: np.ndarray, tau: float) -> np.ndarray:
    """score = z . sum_m softmax(cos(z, mu_m) / tau) w_m, for the rows of Z (chunked)."""
    out, Mn = [], Mu / np.linalg.norm(Mu, axis=1, keepdims=True)
    for i in range(0, len(Z), 4096):
        z = Z[i:i + 4096]
        c = (z / np.linalg.norm(z, axis=1, keepdims=True)) @ Mn.T / tau
        a = np.exp(c - c.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        out.append(np.einsum("nm,nm->n", a, z @ Wd.T))
    return np.concatenate(out)


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

    for fit in args.fits.split(","):
        outs = {v: os.path.join(f"data/avgdir_{v}", RUN, fit) for v in ("uniform", "simw")}
        if all(os.path.exists(os.path.join(o, "DONE")) for o in outs.values()):
            log(f"[{fit}] already done, skipped")
            continue
        train_b = sorted(b for b in files if fold_of[b] != fit)
        sel_b = set(np.random.default_rng(0).choice(train_b, size=round(0.2 * len(train_b)), replace=False))
        b_fit = pd.read_parquet(os.path.join(IRT_DIR, RUN, fit, "items.parquet"), columns=["item", "b"]).set_index("item").b
        mean, sd = scaler(files, train_b)
        log(f"[{fit}] scaler from {len(train_b)} training benchmarks")

        # per training benchmark: unit directions for every alpha, and the benchmark's mean standardised embedding
        dirs, mus, keep = {}, {}, []
        for b in train_b:
            X, items = load_bench(files[b])
            Z = (X - mean) / sd
            y = b_fit.reindex(items).to_numpy()
            mus[b] = Z.mean(0)
            if len(Z) >= MIN_ITEMS and np.std(y) > 0:
                sub = np.random.default_rng(0).permutation(len(Z))[:CAP]
                dirs[b] = directions(Z[sub], y[sub])
                keep.append(b)
        log(f"[{fit}] directions for {len(keep)} benchmarks")

        # choose alpha / tau on the selection benchmarks with directions of the other training benchmarks
        base = [b for b in keep if b not in sel_b]
        sel_data = {}
        for b in sorted(sel_b):
            X, items = load_bench(files[b])
            sel_data[b] = ((X - mean) / sd, b_fit.reindex(items).to_numpy())
        score_r = lambda fn: np.mean([v for v in (within_r(y, fn(Z), np.zeros(len(y))) for Z, y in sel_data.values())
                                      if np.isfinite(v)])
        Mu_base = np.stack([mus[b] for b in base])
        res_u = {a: score_r(lambda Z, j=j: Z @ np.mean([dirs[b][j] for b in base], 0)) for j, a in enumerate(ALPHAS)}
        res_s = {(a, t): score_r(lambda Z, j=j, t=t: simw_scores(Z, Mu_base, np.stack([dirs[b][j] for b in base]), t))
                 for j, a in enumerate(ALPHAS) for t in TAUS}
        a_u = max(res_u, key=res_u.get)
        a_s, t_s = max(res_s, key=res_s.get)
        log(f"[{fit}] selection within r: uniform {res_u[a_u]:.3f} (alpha {a_u:g}); simw {res_s[(a_s, t_s)]:.3f} "
            f"(alpha {a_s:g}, tau {t_s:g})")

        # final directions from all training benchmarks with the chosen values; predict every embedded item
        w_u = np.mean([dirs[b][ALPHAS.index(a_u)] for b in keep], 0)
        Wd = np.stack([dirs[b][ALPHAS.index(a_s)] for b in keep])
        Mu = np.stack([mus[b] for b in keep])
        rows = {"item": [], "benchmark": [], "uniform": [], "simw": []}
        for b, path in files.items():
            X, items = load_bench(path)
            Z = (X - mean) / sd
            rows["item"] += list(items)
            rows["benchmark"] += [b] * len(items)
            rows["uniform"].append(Z @ w_u)
            rows["simw"].append(simw_scores(Z, Mu, Wd, t_s))
        items, bench = np.array(rows["item"]), np.array(rows["benchmark"])
        heldout = np.array([fold_of[b] == fit for b in bench])
        y_tr = b_fit.reindex(items[~heldout]).to_numpy()
        truth = b_final.reindex(items).to_numpy()
        for v, chosen in (("uniform", {"alpha": a_u, "selection_r": res_u[a_u]}),
                          ("simw", {"alpha": a_s, "tau": t_s, "selection_r": res_s[(a_s, t_s)]})):
            s = np.concatenate(rows[v])
            c1, c0 = np.polyfit(s[~heldout], y_tr, 1)
            b_pred = c0 + c1 * s
            check = heldout_check(b_pred[heldout], truth[heldout], bench[heldout])
            os.makedirs(outs[v], exist_ok=True)
            pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout, "b_pred": b_pred}) \
                .to_parquet(os.path.join(outs[v], "pred.parquet"), index=False)
            with open(os.path.join(outs[v], "summary.json"), "w") as f:
                json.dump({"run": RUN, "fit": fit, "variant": v, "chosen": chosen, "n_directions": len(keep),
                           "selection_grid": {str(k): float(x) for k, x in (res_u if v == "uniform" else res_s).items()},
                           "affine": {"c0": c0, "c1": c1}, "heldout_vs_final_b": check,
                           "minutes": (time.time() - t0) / 60}, f, indent=1)
            open(os.path.join(outs[v], "DONE"), "w").close()
            log(f"[{fit}] {v}: held-out benchmarks within r {check['pearson_within_benchmark_mean']:.3f}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
