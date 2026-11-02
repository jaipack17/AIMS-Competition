"""Step 2 (CPU or GPU): which layer / token position / prompt ranks items by IRT difficulty b on UNSEEN benchmarks?

Reads one output folder of probe/extract_activations.py (pilot or full) and, for every saved hidden-state array
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
embeddings, trained on the same items, so it is the like-for-like baseline; the production ridge, trained on
~200k items, gets 0.133 on all 114 benchmarks; see compare_production.py).

Also, without training:
  lens       the model's own rating read with the logit lens: E[digit] = sum_k k P(k) / P(any digit), per layer
             and prompt, correlated with b within benchmarks. Also the mean digit mass.
  length     log number of tokens alone (the trivial baseline every probe must beat).
  cosine     how similar the per-benchmark difficulty directions are (mean pairwise cosine of the per-benchmark
             diff-of-means vectors) against the same with b shuffled within benchmarks.

Speed (written for the full run, ~34,500 items x 4096): the ridge is solved in feature space from per-fold sums
(X'X, X'y), so its cost does not grow with the number of items beyond one pass over the data per fold; the 15
distinct training sets (5 outer + 10 inner pairs of held-out folds) each need one 4096 x 4096 eigendecomposition,
after which every alpha is free. Runs on the GPU when one is free (--device auto), in float64.
Pilot (3,000 items): identical results to the earlier kernel version (tested).

Outputs in <out> (default probe/results/<run folder name>/):
  probe_sweep.csv          one row per key x layer x method
  per_benchmark.csv        r of every held-out benchmark (input of compare_production.py)
  oof_predictions.parquet  held-out prediction of every item for every key x layer x method (+ lens, length)
  probes.npz               each ridge / ridge_wc / dom probe refitted on ALL benchmarks (alpha = the most chosen)
  lens.csv, direction_cosine.csv, layer_curves.png

  python probe/analyze_probes.py probe/acts/full_l8                      # everything
  python probe/analyze_probes.py probe/acts/full_l8 --keys ap__item_last --layers 22,24,26
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts_io  # noqa: E402

ALPHAS = [10.0 ** k for k in range(-1, 11)]   # sklearn units (penalty on the sum of squares); edges are flagged
N_BOOT = 2000


# ---------------------------------------------------------------------------------------------- metrics
def per_bench_r(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> pd.Series:
    """Within-benchmark Pearson r per benchmark (>= 3 items, both non-constant), as in pipeline/ridge_train.
    Vectorised with bincount (one pass, no loop over benchmarks)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    codes, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(codes)).astype(float)
    dy = y - (np.bincount(inv, y, len(codes)) / cnt)[inv]
    dp = p - (np.bincount(inv, p, len(codes)) / cnt)[inv]
    vy = np.bincount(inv, dy * dy, len(codes))
    vp = np.bincount(inv, dp * dp, len(codes))
    cov = np.bincount(inv, dy * dp, len(codes))
    ok = (cnt >= 3) & (vy > 1e-12 * cnt) & (vp > 1e-12 * cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vy * vp)
    return pd.Series(r[ok], index=codes[ok], dtype=float)


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


class Groups:
    """Benchmark index lists, computed once."""

    def __init__(self, g: np.ndarray):
        self.codes, self.inv = np.unique(g, return_inverse=True)
        order = np.argsort(self.inv, kind="stable")
        cuts = np.flatnonzero(np.diff(self.inv[order])) + 1
        self.idx = np.split(order, cuts)

    def center(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for ix in self.idx:
            X[ix] -= X[ix].mean(0)
        return X

    def residualize(self, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """v minus its within-benchmark linear fit on z (z = 0: v centred within benchmark)."""
        out = np.empty(len(v), float)
        for ix in self.idx:
            zc, vc = z[ix] - z[ix].mean(), v[ix] - v[ix].mean()
            zz = zc @ zc
            out[ix] = vc - ((zc @ vc) / zz if zz > 0 else 0.0) * zc
        return out


# ---------------------------------------------------------------------------------------------- probes
def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global standardisation from the features of all items (no labels used). Constant columns -> 0."""
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    sd[sd < 1e-6] = np.inf
    return ((X - mu) / sd).astype(np.float32), mu, sd


class FoldRidge:
    """Ridge for every alpha on any union of folds, from per-fold sums (feature space, exact).

    With intercept: penalised least squares on features centred with the training rows' mean (as sklearn Ridge).
    One eigendecomposition per distinct training set, cached; all alphas then cost one small product."""

    def __init__(self, Z: np.ndarray, y: np.ndarray, fold: np.ndarray, alphas, intercept: bool, torch, device):
        self.torch, self.dev, self.alphas, self.intercept = torch, device, list(alphas), intercept
        self.Z = torch.from_numpy(Z).to(device)                       # float32 on the device
        self.y = torch.from_numpy(np.asarray(y, np.float64)).to(device)
        self.fold = fold
        self.stats = {}
        for f in np.unique(fold):
            ix = torch.from_numpy(np.flatnonzero(fold == f)).to(device)
            Zf = self.Z[ix].double()
            yf = self.y[ix]
            self.stats[f] = (len(ix), Zf.sum(0), Zf.T @ Zf, Zf.T @ yf, float(yf.sum()))
            del Zf
        self.cache = {}

    def path(self, train_folds) -> tuple:
        """(W: d x n_alpha coefficients, mu: d, ybar) trained on the union of train_folds."""
        key = frozenset(int(f) for f in train_folds)
        if key in self.cache:
            return self.cache[key]
        t = self.torch
        n = sum(self.stats[f][0] for f in key)
        sz = sum(self.stats[f][1] for f in key)
        G = sum(self.stats[f][2] for f in key)
        zy = sum(self.stats[f][3] for f in key)
        sy = sum(self.stats[f][4] for f in key)
        if self.intercept:
            mu, ybar = sz / n, sy / n
            G = G - n * t.outer(mu, mu)
            zy = zy - n * mu * ybar
        else:
            mu, ybar = t.zeros_like(sz), 0.0
        s, V = t.linalg.eigh(G)
        s = s.clamp_min(0)
        Vc = V.T @ zy
        a = t.tensor(self.alphas, dtype=t.float64, device=self.dev)
        W = V @ (Vc[:, None] / (s[:, None] + a[None, :]))           # d x n_alpha
        self.cache[key] = (W, mu, ybar)
        return self.cache[key]

    def predict(self, model, rows: np.ndarray) -> np.ndarray:
        W, mu, ybar = model
        ix = self.torch.from_numpy(rows).to(self.dev)
        return ((self.Z[ix].double() - mu) @ W + ybar).cpu().numpy()   # n_rows x n_alpha

    def lofo(self, y: np.ndarray, g: np.ndarray):
        """Leave-one-fold-out predictions; alpha per outer fold by inner leave-one-fold-out mean within-bench r."""
        F = sorted(int(f) for f in np.unique(self.fold))
        pred = np.full(len(y), np.nan)
        chosen = {}
        for f in F:
            tr = [h for h in F if h != f]
            score = np.zeros(len(self.alphas))
            for gi in tr:
                rows = np.flatnonzero(self.fold == gi)
                P = self.predict(self.path([h for h in tr if h != gi]), rows)
                score += [within_r(y[rows], P[:, j], g[rows]) for j in range(len(self.alphas))]
            j = int(np.nanargmax(score))
            chosen[f] = self.alphas[j]
            rows = np.flatnonzero(self.fold == f)
            pred[rows] = self.predict(self.path(tr), rows)[:, j]
        return pred, chosen

    def final(self, alpha):
        W, mu, ybar = self.path(np.unique(self.fold))
        j = self.alphas.index(alpha)
        return W[:, j].cpu().numpy(), mu.cpu().numpy(), float(ybar)


def dom_directions(Xc: np.ndarray, y: np.ndarray, groups: Groups, which=None) -> np.ndarray:
    """Per-benchmark diff-of-means vectors (harder half minus easier half), one row per benchmark in `which`."""
    which = range(len(groups.idx)) if which is None else which
    D = []
    for k in which:
        m = groups.idx[k]
        med = np.median(y[m])
        hard, easy = m[y[m] > med], m[y[m] <= med]
        D.append(Xc[hard].mean(0) - Xc[easy].mean(0) if len(hard) and len(easy) else np.zeros(Xc.shape[1], np.float32))
    return np.stack(D)


def dom_lofo(Xc, y, groups: Groups, fold):
    pred = np.full(len(y), np.nan)
    bench_fold = np.array([fold[ix[0]] for ix in groups.idx])
    for f in np.unique(fold):
        d = dom_directions(Xc, y, groups, np.flatnonzero(bench_fold != f)).mean(0)
        te = fold == f
        pred[te] = Xc[te] @ d
    return pred


def mean_pairwise_cos(D: np.ndarray) -> float:
    n = np.linalg.norm(D, axis=1)
    D = D[n > 0] / n[n > 0, None]
    if len(D) < 2:
        return np.nan
    s = D.sum(0)
    return float((s @ s - len(D)) / (len(D) * (len(D) - 1)))   # mean of the off-diagonal cosines, O(n d)


def load_key_u16(run_dir, key: str):
    """All saved layers of one key as raw bf16 bits (n, n_layers, d) uint16: half the memory of float32."""
    parts = []
    for p in acts_io.benchmark_files(run_dir):
        with np.load(p) as z:
            parts.append(np.asarray(z[key]))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="output folder of extract_activations.py, e.g. probe/acts/full_l8")
    ap.add_argument("--keys", default="", help="comma list of hidden-state keys (default: all saved)")
    ap.add_argument("--layers", default="", help="comma list of real layer numbers (default: all saved)")
    ap.add_argument("--methods", default="dom,ridge,ridge_wc")
    ap.add_argument("--alphas", default="", help="comma list (default 1e-1 .. 1e10)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda[:i]")
    ap.add_argument("--out", default="", help="output folder (default probe/results/<run folder name>)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if args.device == "auto" else args.device))
    alphas = [float(a) for a in args.alphas.split(",") if a] or ALPHAS

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
    ntok, meta = acts_io.load_lens(run, f"{prompts[0]}__n_tokens")
    S = pd.read_parquet(run / "sample.parquet")
    fold_of = dict(zip(S.benchmark, S.fold))
    y = meta.b.to_numpy(float)
    g = meta.benchmark.to_numpy()
    fold = meta.benchmark.map(fold_of).fillna(-1).to_numpy(int)
    loglen = np.log(ntok.astype(float))
    groups = Groups(g)
    y_wc = groups.residualize(y, np.zeros_like(y))          # b centred within benchmark
    len_res = None
    print(f"{run}: {len(y):,} items, {len(groups.codes)} benchmarks, folds {sorted(int(f) for f in set(fold))}; "
          f"keys {keys}; layers {layers}; device {dev}", flush=True)
    if (fold < 0).any():
        print(f"WARNING {np.unique(g[fold < 0])} have no fold; they are tested together as fold -1", flush=True)

    rows, perb, cos_rows = [], [], []
    oof = {"item": meta.item.to_numpy(), "benchmark": g, "fold": fold, "b": y}
    y_len = groups.residualize(y, loglen)

    def record(key, layer, method, pred, extra=None):
        oof[f"{key}|L{layer}|{method}"] = pred.astype(np.float32)
        r = per_bench_r(y, pred, g)
        lo, hi = boot_ci(r.values, rng)
        rp = per_bench_r(y_len, groups.residualize(pred, loglen), g)
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
        del P
    pd.DataFrame(lens_rows).to_csv(out / "lens.csv", index=False)

    # probes
    probes = {}
    for key in keys:
        t_key = time.time()
        U = load_key_u16(run, key)                           # (n, n_saved_layers, d) uint16
        with np.load(files[0]) as z:
            file_layers = [int(v) for v in z["layers"]] if "layers" in z.files else list(range(U.shape[1]))
        print(f"{key}: loaded {U.shape} ({U.nbytes / 1e9:.1f} GB) in {time.time() - t_key:.0f}s", flush=True)
        for layer in layers:
            if layer not in file_layers:
                print(f"skip {key} L{layer}: not saved", flush=True)
                continue
            t1 = time.time()
            X = acts_io.bf16_to_f32(U[:, file_layers.index(layer), :])
            Xs, mu_s, sd_s = standardize(X)
            Xc, mu_c, sd_c = standardize(groups.center(X))
            del X
            msg = []
            tag = f"{key}|L{layer}"
            if "dom" in methods:
                row = record(key, layer, "dom", dom_lofo(Xc, y, groups, fold))
                probes[f"{tag}|dom|w"] = dom_directions(Xc, y, groups).mean(0)
                msg.append(f"dom {row['within_r']:+.3f}")
            for method, Z, yy, icpt, mu_, sd_ in (("ridge", Xs, y, True, mu_s, sd_s),
                                                  ("ridge_wc", Xc, y_wc, False, mu_c, sd_c)):
                if method not in methods:
                    continue
                fr = FoldRidge(Z, yy, fold, alphas, icpt, torch, dev)
                pred, ch = fr.lofo(yy, g)
                edge = [f for f, a in ch.items() if a in (alphas[0], alphas[-1])]
                row = record(key, layer, method, pred, {"alphas": str(ch), "alpha_at_edge": len(edge)})
                a_final = Counter(ch.values()).most_common(1)[0][0]
                w, mu_z, ybar = fr.final(a_final)
                probes.update({f"{tag}|{method}|w": w, f"{tag}|{method}|mu_z": mu_z, f"{tag}|{method}|ybar": ybar,
                               f"{tag}|{method}|feat_mean": mu_, f"{tag}|{method}|feat_sd": sd_,
                               f"{tag}|{method}|alpha": a_final})
                msg.append(f"{method} {row['within_r']:+.3f}" + (" (alpha at grid edge)" if edge else ""))
                del fr
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            c_real = mean_pairwise_cos(dom_directions(Xc, y, groups))
            ysh = y.copy()
            for ix in groups.idx:
                ysh[ix] = rng.permutation(y[ix])
            c_null = mean_pairwise_cos(dom_directions(Xc, ysh, groups))
            cos_rows.append({"key": key, "layer": layer, "mean_cos": c_real, "mean_cos_shuffled": c_null})
            print(f"{key:16s} L{layer:<3d} " + "  ".join(msg) + f"  | cos {c_real:+.3f} (shuffled {c_null:+.3f})"
                  f"  [{time.time() - t1:.0f}s]", flush=True)
            del Xs, Xc
        del U

    res = pd.DataFrame(rows)
    ref_key = "ap__last" if "ap__last" in keys else f"{first}__last"
    P = pd.DataFrame(perb)
    ref = P[(P.key == ref_key) & (P.layer == last_layer) & (P.method == "ridge")].set_index("benchmark").r
    if len(ref):
        d_cols = []
        for (key, layer, method), cur in P.groupby(["key", "layer", "method"], sort=False):
            d = (cur.set_index("benchmark").r - ref).dropna()
            lo, hi = boot_ci(d.values, rng)
            d_cols.append({"key": key, "layer": layer, "method": method, "d_vs_ref": d.mean(), "d_ci_lo": lo,
                           "d_ci_hi": hi, "better_on": int((d > 0).sum()), "of": len(d)})
        res = res.merge(pd.DataFrame(d_cols), on=["key", "layer", "method"], how="left")
    res.to_csv(out / "probe_sweep.csv", index=False)
    P.to_csv(out / "per_benchmark.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(out / "direction_cosine.csv", index=False)
    pd.DataFrame(oof).to_parquet(out / "oof_predictions.parquet", index=False)
    np.savez(out / "probes.npz", **{k: np.asarray(v) for k, v in probes.items()})

    pd.set_option("display.width", 200)
    show = [c for c in ["key", "layer", "method", "within_r", "ci_lo", "ci_hi", "partial_r_len", "d_vs_ref",
                        "d_ci_lo", "d_ci_hi", "better_on", "of"] if c in res]
    print(f"\nreference: ridge on {ref_key} L{last_layer} (same items). Top 15 by within-benchmark r:")
    print(res.sort_values("within_r", ascending=False)[show].head(15).round(3).to_string(index=False))
    print(f"\nwrote {out}/probe_sweep.csv, per_benchmark.csv, oof_predictions.parquet, probes.npz, lens.csv, "
          f"direction_cosine.csv [{(time.time() - t0) / 60:.1f} min]")

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
