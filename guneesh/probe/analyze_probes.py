"""Step 2 (CPU): which layer / token position / prompt ranks items by IRT difficulty b on UNSEEN benchmarks?

Reads one output folder of probe/extract_activations.py and, for every saved hidden-state array
(<prompt>__last for each prompt, <first prompt>__item_last, __item_mean) and every saved layer, fits three probes
with leave-one-fold-out over the project's benchmark folds (the test benchmarks are never seen in training):

  dom        difference of means (Goodfire style, no hyperparameter). Per training benchmark: mean activation of
             the harder half minus the easier half (by b), on benchmark-centred, standardised activations;
             direction = average over training benchmarks; score = activation . direction.
  ridge      standardised activations -> ridge -> b, as pipeline/ridge_train.py (alpha by inner leave-one-fold-out
             on the training folds, scored by the target metric).
  ridge_wc   the same, but activations and b centred within each benchmark first, so the probe can only learn
             within-benchmark differences (the test benchmark is centred with its own unlabelled activations).

Metric: mean over held-out benchmarks of the within-benchmark Pearson r (pipeline/ridge_train.heldout_check),
with a 95% bootstrap interval over benchmarks, the per-fold means, the partial r after removing log token length,
and the paired difference against the reference = ridge on ap__last at the last layer (the vector of the existing
embeddings, trained on the same pilot items, so it is the like-for-like baseline; the production ridge, trained on
~200k items, gets 0.133 on all 114 benchmarks).

Also, without training:
  lens       the model's own rating read with the logit lens: E[digit] = sum_k k P(k) / P(any digit), per layer
             and prompt, correlated with b within benchmarks (positive = the model rates hard items higher;
             rate_rev should be negative). Also the mean digit mass (is it answering with a digit at all?).
  length     log number of tokens alone (the trivial baseline every probe must beat).
  cosine     how similar the per-benchmark difficulty directions are (mean pairwise cosine of the per-benchmark
             diff-of-means vectors) against the same with b shuffled within benchmarks. The existing embeddings
             give ~0.009: nearly orthogonal directions, i.e. no shared "difficulty" axis.

Outputs in <out> (default probe/results/<run folder name>/):
  probe_sweep.csv        one row per key x layer x method
  per_benchmark.csv      r of every held-out benchmark (for your own tests)
  lens.csv               logit-lens rating r and digit mass per prompt x layer
  direction_cosine.csv   per-benchmark direction similarity per key x layer
  layer_curves.png       if matplotlib is installed

  python probe/analyze_probes.py probe/acts/pilot_l5                 # ~1-5 min for 5 layers on a laptop
  python probe/analyze_probes.py probe/acts/pilot --layers 20,36 --keys ap__last
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts_io  # noqa: E402

ALPHAS = [10.0 ** k for k in range(-1, 8)]   # in units of the standardised feature scale; edges are reported
N_BOOT = 2000


# ---------------------------------------------------------------------------------------------- metrics
def per_bench_r(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pd.Series:
    """Within-benchmark Pearson r per benchmark (>= 3 items, both non-constant), as in pipeline/ridge_train."""
    out = {}
    for b in np.unique(g):
        m = g == b
        if m.sum() >= 3 and np.std(y[m]) > 0 and np.std(p[m]) > 0:
            out[b] = float(np.corrcoef(y[m], p[m])[0, 1])
    return pd.Series(out, dtype=float)


def within_r(y, p, g) -> float:
    r = per_bench_r(y, p, g)
    return float(r.mean()) if len(r) else np.nan


def boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    m = rng.choice(v, (N_BOOT, len(v)), replace=True).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def residualize_within(v: np.ndarray, z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v minus its within-benchmark linear fit on z."""
    out = np.empty_like(v, dtype=float)
    for b in np.unique(g):
        m = g == b
        zc, vc = z[m] - z[m].mean(), v[m] - v[m].mean()
        beta = (zc @ vc) / (zc @ zc) if zc @ zc > 0 else 0.0
        out[m] = vc - beta * zc
    return out


def center_within(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    X = X.copy()
    for b in np.unique(g):
        m = g == b
        X[m] -= X[m].mean(0)
    return X


# ---------------------------------------------------------------------------------------------- probes
def standardize(X: np.ndarray) -> np.ndarray:
    """Global standardisation from the features of all items (no labels used). Constant columns -> 0."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = np.inf
    return ((X - mu) / sd).astype(np.float32)


def kernel_ridge_path(G: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, alphas, intercept=True):
    """Ridge predictions on te for every alpha, from the linear kernel G = X X^T (dual form, exact for d > n)."""
    Ktr = G[np.ix_(tr, tr)].astype(np.float64)
    Kte = G[np.ix_(te, tr)].astype(np.float64)
    ytr = y[tr].astype(np.float64)
    if intercept:  # centre the features on the training rows: K -> H K H, K_te -> (K_te - 1 k^T)(H)
        rm = Ktr.mean(0)
        tm = Kte.mean(1, keepdims=True)
        Kte = Kte - rm[None, :] - tm + rm.mean()
        Ktr = Ktr - rm[None, :] - rm[:, None] + rm.mean()
        y0 = ytr.mean()
    else:
        y0 = 0.0
    s, U = np.linalg.eigh(Ktr)
    s = np.clip(s, 0, None)
    Uy = U.T @ (ytr - y0)
    KU = Kte @ U
    return {a: KU @ (Uy / (s + a)) + y0 for a in alphas}


def ridge_lofo(G, y, g, fold, alphas, intercept):
    """Leave-one-fold-out predictions; alpha chosen per outer fold by inner leave-one-fold-out within-bench r."""
    pred = np.full(len(y), np.nan)
    chosen = {}
    for f in np.unique(fold):
        te_all, tr_all = np.where(fold == f)[0], np.where(fold != f)[0]
        inner = {a: [] for a in alphas}
        for fi in np.unique(fold[tr_all]):
            tr, te = tr_all[fold[tr_all] != fi], tr_all[fold[tr_all] == fi]
            for a, p in kernel_ridge_path(G, y, tr, te, alphas, intercept).items():
                inner[a].append(within_r(y[te], p, g[te]))
        best = max(alphas, key=lambda a: np.nanmean(inner[a]))
        chosen[int(f)] = best
        pred[te_all] = kernel_ridge_path(G, y, tr_all, te_all, [best], intercept)[best]
    return pred, chosen


def dom_directions(Xc: np.ndarray, y: np.ndarray, g: np.ndarray, benches) -> np.ndarray:
    """Per-benchmark diff-of-means vectors (harder half minus easier half), rows in the order of benches."""
    D = []
    for b in benches:
        m = np.where(g == b)[0]
        med = np.median(y[m])
        hard, easy = m[y[m] > med], m[y[m] <= med]
        if len(hard) == 0 or len(easy) == 0:
            D.append(np.zeros(Xc.shape[1], np.float32))
            continue
        D.append(Xc[hard].mean(0) - Xc[easy].mean(0))
    return np.stack(D)


def dom_lofo(Xc, y, g, fold):
    pred = np.full(len(y), np.nan)
    for f in np.unique(fold):
        tr_b = np.unique(g[fold != f])
        d = dom_directions(Xc, y, g, tr_b).mean(0)
        te = fold == f
        pred[te] = Xc[te] @ d
    return pred


def mean_pairwise_cos(D: np.ndarray) -> float:
    n = np.linalg.norm(D, axis=1)
    D = D[n > 0] / n[n > 0, None]
    if len(D) < 2:
        return np.nan
    C = D @ D.T
    iu = np.triu_indices(len(D), 1)
    return float(C[iu].mean())


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="output folder of extract_activations.py, e.g. probe/acts/pilot_l5")
    ap.add_argument("--keys", default="", help="comma list of hidden-state keys (default: all saved)")
    ap.add_argument("--layers", default="", help="comma list of real layer numbers (default: all saved)")
    ap.add_argument("--methods", default="dom,ridge,ridge_wc")
    ap.add_argument("--out", default="", help="output folder (default probe/results/<run folder name>)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = acts_io.config(run)
    files = acts_io.benchmark_files(run)
    if not files:
        raise SystemExit(f"no <benchmark>.npz in {run}")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "results" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    prompts = list(cfg["prompts"])
    first = cfg.get("item_positions_from", prompts[0])
    with np.load(files[0]) as z:
        keys_avail = [k for k in z.files if k.endswith(("__last", "__item_last", "__item_mean"))]
    keys = [k for k in args.keys.split(",") if k] or keys_avail
    layers_saved = acts_io.saved_layers(run)
    layers = [int(x) for x in args.layers.split(",") if x] or layers_saved
    last_layer = cfg["n_layer_outputs"] - 1
    methods = [m for m in args.methods.split(",") if m]

    # labels, benchmarks, folds
    _, meta = acts_io.load_lens(run, f"{prompts[0]}__n_tokens")
    S = pd.read_parquet(run / "sample.parquet")
    fold_of = dict(zip(S.benchmark, S.fold))
    y = meta.b.to_numpy(float)
    g = meta.benchmark.to_numpy()
    fold = meta.benchmark.map(fold_of).fillna(-1).to_numpy(int)
    loglen = np.log(acts_io.load_lens(run, f"{prompts[0]}__n_tokens")[0].astype(float))
    n_bench = len(np.unique(g))
    print(f"{run}: {len(y):,} items, {n_bench} benchmarks, folds {sorted(int(f) for f in set(fold))}; keys {keys}; layers {layers}",
          flush=True)
    if (fold < 0).any():
        print(f"WARNING {np.unique(g[fold < 0])} have no fold; they are tested together as fold -1", flush=True)

    rows, perb, cos_rows = [], [], []

    def record(key, layer, method, pred, extra=None):
        r = per_bench_r(y, pred, g)
        lo, hi = boot_ci(r.values, rng)
        rp = per_bench_r(residualize_within(y, loglen, g), residualize_within(pred, loglen, g), g)
        row = {"key": key, "layer": layer, "method": method, "within_r": r.mean(), "ci_lo": lo, "ci_hi": hi,
               "partial_r_len": rp.mean(), "n_bench": len(r), "frac_pos": float((r > 0).mean())}
        for f in sorted(set(fold)):
            row[f"fold{f}"] = r[[b for b in r.index if fold_of.get(b, -1) == f]].mean()
        row.update(extra or {})
        rows.append(row)
        perb.extend({"key": key, "layer": layer, "method": method, "benchmark": b, "r": v} for b, v in r.items())
        return row

    # baselines without training
    record("length", -1, "log_tokens", loglen)
    lens_rows = []
    for p in prompts:
        P, _ = acts_io.load_lens(run, f"{p}__digit_probs")       # (n, n_layers, 10)
        mass = P.sum(-1)
        ev = (P * np.arange(10)).sum(-1) / np.maximum(mass, 1e-12)
        for L in range(P.shape[1]):
            r = per_bench_r(y, ev[:, L], g)
            lo, hi = boot_ci(r.values, rng)
            lens_rows.append({"prompt": p, "layer": L, "rating_within_r": r.mean(), "ci_lo": lo, "ci_hi": hi,
                              "digit_mass_mean": float(mass[:, L].mean()),
                              "digit_mass_min_bench": float(pd.Series(mass[:, L]).groupby(g).mean().min()),
                              "n_bench": len(r)})
        rp = lens_rows[-1]
        print(f"lens {p:8s} last layer: rating r {rp['rating_within_r']:+.3f} [{rp['ci_lo']:+.3f}, {rp['ci_hi']:+.3f}]"
              f"  digit mass {rp['digit_mass_mean']:.2f}", flush=True)
        record(f"{p}__lens", last_layer, "rating_EV", ev[:, -1])
    pd.DataFrame(lens_rows).to_csv(out / "lens.csv", index=False)

    # probes
    for key in keys:
        for layer in layers:
            if layer not in layers_saved:
                print(f"skip {key} L{layer}: not saved", flush=True)
                continue
            t1 = time.time()
            X, m2 = acts_io.load_layer(run, key, layer)
            assert (m2.item.to_numpy() == meta.item.to_numpy()).all(), "item order differs between arrays"
            Xs = standardize(X)
            Xc = standardize(center_within(X, g))
            del X
            msg = []
            if "dom" in methods:
                row = record(key, layer, "dom", dom_lofo(Xc, y, g, fold))
                msg.append(f"dom {row['within_r']:+.3f}")
            if "ridge" in methods:
                G = Xs @ Xs.T
                pred, ch = ridge_lofo(G, y, g, fold, ALPHAS, intercept=True)
                row = record(key, layer, "ridge", pred, {"alphas": str(ch)})
                msg.append(f"ridge {row['within_r']:+.3f}")
            if "ridge_wc" in methods:
                G = Xc @ Xc.T
                yc = residualize_within(y, np.zeros_like(y), g)   # = y centred within benchmark
                pred, ch = ridge_lofo(G, yc, g, fold, ALPHAS, intercept=False)
                row = record(key, layer, "ridge_wc", pred, {"alphas": str(ch)})
                msg.append(f"ridge_wc {row['within_r']:+.3f}")
            # are the per-benchmark difficulty directions shared? (all benchmarks, vs b shuffled within benchmark)
            benches = np.unique(g)
            ysh = y.copy()
            for b in benches:
                m = np.where(g == b)[0]
                ysh[m] = rng.permutation(y[m])
            c_real = mean_pairwise_cos(dom_directions(Xc, y, g, benches))
            c_null = mean_pairwise_cos(dom_directions(Xc, ysh, g, benches))
            cos_rows.append({"key": key, "layer": layer, "mean_cos": c_real, "mean_cos_shuffled": c_null})
            print(f"{key:16s} L{layer:<3d} " + "  ".join(msg) + f"  | cos {c_real:+.3f} (shuffled {c_null:+.3f})"
                  f"  [{time.time() - t1:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    # paired difference against the reference (ridge, ap__last or first prompt's last, last layer)
    ref_key = "ap__last" if "ap__last" in keys else f"{first}__last"
    P = pd.DataFrame(perb)
    ref = P[(P.key == ref_key) & (P.layer == last_layer) & (P.method == "ridge")].set_index("benchmark").r
    if len(ref):
        d_cols = []
        for _, row in res.iterrows():
            cur = P[(P.key == row.key) & (P.layer == row.layer) & (P.method == row.method)].set_index("benchmark").r
            d = (cur - ref).dropna()
            lo, hi = boot_ci(d.values, rng)
            d_cols.append({"d_vs_ref": d.mean(), "d_ci_lo": lo, "d_ci_hi": hi, "better_on": int((d > 0).sum()),
                           "of": len(d)})
        res = pd.concat([res, pd.DataFrame(d_cols)], axis=1)
    res.to_csv(out / "probe_sweep.csv", index=False)
    P.to_csv(out / "per_benchmark.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(out / "direction_cosine.csv", index=False)

    pd.set_option("display.width", 200)
    show = [c for c in ["key", "layer", "method", "within_r", "ci_lo", "ci_hi", "partial_r_len", "d_vs_ref",
                        "d_ci_lo", "d_ci_hi", "better_on", "of"] if c in res]
    print(f"\nreference: ridge on {ref_key} L{last_layer} (same items). Top 15 by within-benchmark r:")
    print(res.sort_values("within_r", ascending=False)[show].head(15).round(3).to_string(index=False))
    print(f"\nwrote {out}/probe_sweep.csv, per_benchmark.csv, lens.csv, direction_cosine.csv "
          f"[{(time.time() - t0) / 60:.1f} min]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no plot)")
        return
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4.2 * (len(methods) + 1), 3.6), sharey=True)
    probe_res = res[res.layer >= 0]
    for ax, mth in zip(axes, methods):
        for key in keys:
            d = probe_res[(probe_res.key == key) & (probe_res.method == mth)].sort_values("layer")
            if len(d):
                ax.plot(d.layer, d.within_r, marker="o", label=key)
                ax.fill_between(d.layer, d.ci_lo, d.ci_hi, alpha=0.15)
        ax.set_title(mth)
        ax.set_xlabel("layer")
        ax.axhline(0, color="grey", lw=0.6)
    axes[0].set_ylabel("held-out within-benchmark r")
    axes[0].legend(fontsize=7)
    L = pd.DataFrame(lens_rows)
    for p in prompts:
        d = L[L.prompt == p]
        axes[-1].plot(d.layer, d.rating_within_r, label=f"{p} (mass {d.digit_mass_mean.iloc[-1]:.2f})")
    axes[-1].set_title("logit-lens rating")
    axes[-1].set_xlabel("layer")
    axes[-1].axhline(0, color="grey", lw=0.6)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "layer_curves.png", dpi=130)
    print(f"wrote {out}/layer_curves.png")


if __name__ == "__main__":
    main()
