"""Step 3 (CPU): does a pilot probe beat the PRODUCTION ridge on the same benchmarks and items?

The pilot probes (analyze_probes.py) are trained on ~2,400 items; the production ridge (pipeline/ridge_train.py) on
every embedded item of the training folds (~200k). This script rebuilds that production ridge from the existing
embeddings (data/emb/qwen3vl8b, all items, last layer, last token of the AP prompt), with leave-one-fold-out over the
project's folds, and scores it on the pilot's benchmarks:

  * on the pilot's sampled items (same items as the probes: a paired, like-for-like comparison), and
  * on all items of those benchmarks, and on all 115 benchmarks (sanity check: should be close to 0.133).

Ridge as ridge_train.py: standardised features, ridge with an intercept, alpha from a grid chosen per outer fold by
inner leave-one-fold-out (here scored by the target metric, mean within-benchmark r, rather than R^2; the script
also prints the result for every alpha so the choice can be checked). Target = b of the final IRT fit for the
training items (the fold fits are not on the server; this is what analyze_probes.py uses too).

  python probe/compare_production.py probe/acts/pilot_l5
  python probe/compare_production.py probe/acts/pilot_l5 --compare ap__item_last:24:ridge_wc,rate__lens:36:rating_EV

Needs probe/results/<run>/per_benchmark.csv from analyze_probes.py, and ~8 GB RAM. Writes
probe/results/<run>/production_ridge.csv (per-benchmark r) and prints the paired differences.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_probes import boot_ci, per_bench_r  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [10.0 ** k for k in range(-2, 13)]   # sklearn units; ridge_train.py grows its grid up to 1e12


def fit_path(stats: dict, train_folds) -> dict:
    """Ridge (standardised features, intercept) for every alpha in sklearn units, trained on the union of
    train_folds, from per-fold sums. Returns {alpha: (coef on raw features, mean, y mean)}."""
    n = sum(stats[f]["n"] for f in train_folds)
    sx = sum(stats[f]["sx"] for f in train_folds)
    xtx = sum(stats[f]["xtx"] for f in train_folds)
    xty = sum(stats[f]["xty"] for f in train_folds)
    sy = sum(stats[f]["sy"] for f in train_folds)
    mu, ybar = sx / n, sy / n
    C = xtx - n * np.outer(mu, mu)
    c = xty - n * mu * ybar
    sd = np.sqrt(np.clip(np.diag(C) / n, 0, None))
    sd[sd < 1e-8] = np.inf
    A = C / np.outer(sd, sd) / n                 # Gram of the z-scores, per item
    bvec = c / sd / n
    s, V = np.linalg.eigh(A)
    Vb = V.T @ bvec
    out = {}
    for a in ALPHAS:
        w = V @ (Vb / (np.clip(s, 0, None) + a / n))   # (Z'Z + a I) w = Z'y, divided by n
        out[a] = (w / sd, mu, ybar)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default="ap__item_last:24:ridge_wc,ap__item_last:24:ridge,rate__lens:36:rating_EV",
                    help="key:layer:method rows of per_benchmark.csv to compare with the production ridge")
    ap.add_argument("--results", default="", help="analyze_probes output folder (default probe/results/<run name>)")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--emb_dir", default="", help="default <data_dir>/emb/qwen3vl8b")
    ap.add_argument("--irt_run", default="versionB_4parts_logitnormal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    run = Path(args.run_dir)
    res_dir = Path(args.results) if args.results else ROOT / "probe" / "results" / run.name
    data = Path(args.data_dir)
    emb_dir = Path(args.emb_dir) if args.emb_dir else data / "emb" / "qwen3vl8b"
    S = pd.read_parquet(run / "sample.parquet")
    pilot_items = set(S.item)
    pilot_benches = sorted(S.benchmark.unique())
    folds = pd.read_csv(data / "irt" / "folds.csv")
    fold_of = dict(zip(folds.benchmark, folds.fold))
    irt = pd.read_parquet(data / "irt" / args.irt_run / "final" / "items.parquet", columns=["item", "b"])
    b_of = pd.Series(irt.b.values, index=irt.item.values)
    b_of = b_of[~b_of.index.duplicated()]

    # load every embedded item with a difficulty
    Xs, items, bench = [], [], []
    files = [p for p in sorted(emb_dir.glob("*.npz")) if p.stem in fold_of]
    print(f"loading {len(files)} embedding files from {emb_dir} ...", flush=True)
    for i, p in enumerate(files):
        if i % 20 == 0:
            print(f"  {i}/{len(files)} files [{time.time() - t0:.0f}s]", flush=True)
        with np.load(p, allow_pickle=True) as z:
            keys = np.array([f"{p.stem}::{t}" for t in z["task_ids"].astype(str)])
            ok = b_of.index.get_indexer(keys) >= 0   # hash lookup; np.isin on an object Index loops in Python
            if ok.any():
                Xs.append(z["X"][ok].astype(np.float32))
                items.append(keys[ok])
                bench += [p.stem] * int(ok.sum())
    missing = sorted(set(fold_of) - set(bench))
    X = np.concatenate(Xs)
    del Xs
    items = np.concatenate(items)
    bench = np.array(bench)
    y = b_of.reindex(items).to_numpy(float)
    fold = np.array([fold_of[b] for b in bench])
    print(f"{len(y):,} embedded items with b, {len(set(bench))} benchmarks"
          + (f" (no embeddings for {len(missing)}: {missing[:5]}...)" if missing else "")
          + f"; loaded in {time.time() - t0:.0f}s", flush=True)
    absent = sorted(set(pilot_benches) - set(bench))
    if absent:
        raise SystemExit(f"pilot benchmarks without embeddings: {absent}")

    # per-fold sufficient statistics (sum x, X^T X, X^T y, sum y, n) in float64
    F = sorted(set(fold))
    d = X.shape[1]
    stats = {}
    for f in F:
        m = fold == f
        Xf = X[m].astype(np.float64)
        stats[f] = dict(n=int(m.sum()), sx=Xf.sum(0), xtx=Xf.T @ Xf, xty=Xf.T @ y[m], sy=float(y[m].sum()))
        del Xf
        print(f"  fold {f}: {stats[f]['n']:,} items, Gram done [{time.time() - t0:.0f}s]", flush=True)

    def predict(model, idx):
        coef, mu, ybar = model
        return (X[idx] - mu.astype(np.float32)) @ coef.astype(np.float32) + ybar

    pred = np.full(len(y), np.nan)
    rows_alpha = []
    for f in F:
        tr = [g for g in F if g != f]
        inner = {a: [] for a in ALPHAS}
        for gi in tr:
            path = fit_path(stats, [g for g in tr if g != gi])
            idx = np.where(fold == gi)[0]
            for a, mdl in path.items():
                inner[a].append(np.nanmean(per_bench_r(y[idx], predict(mdl, idx), bench[idx])))
        score = {a: float(np.mean(v)) for a, v in inner.items()}
        best = max(score, key=score.get)
        path = fit_path(stats, tr)
        idx = np.where(fold == f)[0]
        for a, mdl in path.items():
            p = predict(mdl, idx)
            rows_alpha.append({"fold": f, "alpha": a, "inner_within_r": score[a],
                               "heldout_within_r": float(per_bench_r(y[idx], p, bench[idx]).mean())})
            if a == best:
                pred[idx] = p
        edge = " (EDGE of the grid)" if best in (ALPHAS[0], ALPHAS[-1]) else ""
        print(f"  fold {f}: alpha {best:g}{edge} (inner within r {score[best]:.3f}) [{time.time() - t0:.0f}s]", flush=True)

    A = pd.DataFrame(rows_alpha)
    print("\nheld-out within r by alpha (mean over folds), to check the choice:")
    print(A.groupby("alpha")[["inner_within_r", "heldout_within_r"]].mean().round(3).to_string())

    r_all = per_bench_r(y, pred, bench)
    in_pilot = np.isin(items, list(pilot_items))
    r_pilot_items = per_bench_r(y[in_pilot], pred[in_pilot], bench[in_pilot])
    pb = np.isin(bench, pilot_benches)
    r_pilot_all = per_bench_r(y[pb], pred[pb], bench[pb])
    print(f"\nproduction ridge (rebuilt), held-out within-benchmark r:")
    print(f"  all {len(r_all)} benchmarks, all items:              {r_all.mean():.3f}   (EXPERIMENTS.md: 0.133)")
    print(f"  {len(r_pilot_all)} pilot benchmarks, all items:          {r_pilot_all.mean():.3f}")
    print(f"  {len(r_pilot_items)} pilot benchmarks, pilot items only:  {r_pilot_items.mean():.3f}  "
          f"(n items {int(in_pilot.sum()):,}; the paired reference below)")

    out = pd.DataFrame({"benchmark": r_pilot_items.index, "prod_ridge_pilot_items": r_pilot_items.values})
    out["prod_ridge_all_items"] = r_pilot_all.reindex(r_pilot_items.index).values
    res_dir.mkdir(parents=True, exist_ok=True)

    P = pd.read_csv(res_dir / "per_benchmark.csv") if (res_dir / "per_benchmark.csv").exists() else None
    if P is None:
        print(f"\n(no {res_dir}/per_benchmark.csv: run analyze_probes.py first for the paired comparison)")
    else:
        print(f"\npaired vs production ridge on the same items (difference of within-benchmark r, 95% bootstrap):")
        for spec in [s for s in args.compare.split(",") if s]:
            key, layer, method = spec.split(":")
            cur = P[(P.key == key) & (P.layer == int(layer)) & (P.method == method)].set_index("benchmark").r
            if cur.empty:
                print(f"  {spec}: not in per_benchmark.csv")
                continue
            dd = (cur - r_pilot_items).dropna()
            lo, hi = boot_ci(dd.values, rng)
            out[spec] = cur.reindex(out.benchmark).values
            print(f"  {spec:32s} r {cur.reindex(dd.index).mean():.3f} vs {r_pilot_items.reindex(dd.index).mean():.3f}"
                  f"  diff {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  better on {int((dd > 0).sum())} of {len(dd)}")
    out.to_csv(res_dir / "production_ridge.csv", index=False)
    A.to_csv(res_dir / "production_ridge_alphas.csv", index=False)
    print(f"\nwrote {res_dir}/production_ridge.csv, production_ridge_alphas.csv [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
