"""Can a small MLP (C-MIRT stage 2) predict MIRT item parameters from the Qwen3-VL-8B embedding where the ridge cannot?

C-MIRT (Shang, Tang, He, arXiv 2608.22295, Appendix A) stage 2: phi(e_j) is "a multilayer perceptron" fitted by
minimising the mean squared error to the stage-1 item factors over mini-batches. No architecture or optimiser
details and no code are published, so the choices below are ours:
  * inputs  standardised embeddings (StandardScaler on the training rows, as for the ridge)
  * targets [b, w1..wK] of one MIRT fit (w = v for nested, a for m2pl), each standardised on the training rows
  * split   GROUPED by benchmark: 20% of the fit's training benchmarks (random, seed 0) are validation benchmarks,
            never trained on; half of them choose the early-stopping epoch, the other half give the reported
            scores. A random item split would let the MLP recognise benchmarks and overstate cold-start accuracy.
  * designs a fixed set of three MLPs, all reported (no tuning on the reported benchmarks), each with every
            regularisation below: Gaussian input noise (sd 0.1) and input dropout 0.2; Linear -> LayerNorm -> GELU
            -> dropout 0.5 hidden layers; AdamW (lr 1e-4, weight decay 0.05); gradient clipping at norm 1;
            learning rate halved after 2 epochs without improvement; early stopping (patience 6, at most 40
            epochs) on the mean squared error of the weights w on unseen benchmarks, best epoch restored;
            prediction = average of 3 seeds. One design samples training items so every benchmark weighs the
            same (large benchmarks cannot dominate).
            A first version (lr 1e-3, dropout 0.2) overfit from the first epoch, so early stopping kept the
            untrained network; the designs below use lower learning rates and stronger regularisation.
Compared on the same validation benchmarks with
  * the mean of the training targets (R^2 = 0 by construction on the training rows; slightly negative or positive
    on validation),
  * the ridge (fast_ridge, exact), its alpha per target chosen on the training benchmarks only by an inner
    GroupKFold(4), from the AP grid extended to 1e12.
Reported per target on the validation benchmarks: R^2, and the mean within-benchmark Pearson correlation.
Only the fold's own fit is used (the fit never saw the fold's held-out benchmarks; we do not touch those).

Run:  python analysis/mirt_mlp_check.py --run mirt_nested_K2 [--fit fold0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from fast_ridge import Scaler, fold_sums, ridge_cv  # noqa: E402
from ridge_train import ALPHAS  # noqa: E402


def load_embeddings_lean(emb_dir: str, benchmarks: set, cap: int | None = None):
    """Same rows and order as ridge_train.load_embeddings, but filled into one pre-allocated float32 array
    (peak memory = the array + one benchmark file, instead of twice the array)."""
    import glob
    paths = [p for p in sorted(glob.glob(os.path.join(emb_dir, "*.npz"))) if os.path.basename(p)[:-4] in benchmarks]
    sizes = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            sizes.append(min(len(z["task_ids"]), cap or 10 ** 9))
    X = np.empty((sum(sizes), 4096), dtype=np.float32)
    items, bench, at = [], [], 0
    for p, n in zip(paths, sizes):
        b = os.path.basename(p)[:-4]
        with np.load(p, allow_pickle=True) as z:
            X[at:at + n] = z["X"][:n]
            items += [f"{b}::{i}" for i in z["task_ids"][:n].astype(str)]
        bench += [b] * n
        at += n
    return X, np.array(items), np.array(bench)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def within_r(y, p, g):
    rs = [np.corrcoef(y[g == b], p[g == b])[0, 1] for b in np.unique(g)
          if (g == b).sum() >= 3 and np.std(y[g == b]) > 0 and np.std(p[g == b]) > 0]
    return float(np.mean(rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", default="fold0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--debug_benchmarks", type=int, default=0, help="code check only: use this many benchmarks")
    ap.add_argument("--debug_cap", type=int, default=0, help="code check only: at most this many items each")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    t0 = time.time()

    folds = pd.read_csv(os.path.join(ROOT, "data", "irt", "folds.csv"))
    train_b = sorted(folds.benchmark[folds.fold != int(args.fit[-1])]) if args.fit != "final" else sorted(folds.benchmark)
    if args.debug_benchmarks:
        train_b = train_b[:args.debug_benchmarks]
    X, items, bench = load_embeddings_lean(os.path.join(ROOT, "data", "emb", "qwen3vl8b"), set(train_b),
                                           args.debug_cap or None)
    it = pd.read_parquet(os.path.join(ROOT, "data", "irt", args.run, args.fit, "items.parquet")).set_index("item")
    tcols = ["b"] + sorted([c for c in it.columns if c[0] in "va" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    Y = it[tcols].reindex(items).to_numpy(np.float64)
    assert not np.isnan(Y).any(), "embedded training items without MIRT parameters"
    val_list = list(rng.choice(train_b, size=round(0.2 * len(train_b)), replace=False))
    sel_b, rep_b = set(val_list[::2]), set(val_list[1::2])   # early stopping on one half, reported on the other
    va = np.array([b in sel_b or b in rep_b for b in bench])
    sel = np.array([b in sel_b for b in bench])
    rep = np.array([b in rep_b for b in bench])
    tr = ~va
    print(f"{args.run} {args.fit}: {tr.sum():,} train items ({len(train_b) - len(val_list)} benchmarks); validation "
          f"{sel.sum():,} items ({len(sel_b)} benchmarks) for early stopping, {rep.sum():,} items ({len(rep_b)} "
          f"benchmarks) for the reported scores; targets {tcols} ({time.time() - t0:.0f} s)", flush=True)

    # ---- standardise inputs (training rows) in place, float32, chunked: no second copy of X
    sc = Scaler.fit(X, np.flatnonzero(tr))
    mean, scale = sc.mean.astype(np.float32), sc.scale.astype(np.float32)
    for i in range(0, len(X), 8192):
        X[i:i + 8192] -= mean
        X[i:i + 8192] /= scale
    ym, ys = Y[tr].mean(0), Y[tr].std(0)
    Ys = ((Y - ym) / ys).astype(np.float32)

    # ---- ridge on the same split (X is already standardised: identity scaler)
    ident = Scaler(mean=np.zeros(X.shape[1]), scale=np.ones(X.shape[1]))
    rows = np.flatnonzero(tr)
    gb = bench[rows]
    ub = np.unique(gb)
    inner = np.searchsorted(ub, gb) % 4                     # 4 inner groups of whole benchmarks
    sums = fold_sums(X, Ys[:, :].astype(np.float64), rows, inner, ident)
    res = ridge_cv(sums, ALPHAS + [1e8, 1e9, 1e10, 1e11, 1e12], score="r2")
    Xva = X[rep]                                            # one copy of the reported rows (~0.3 GB)
    P_ridge = np.concatenate([Xva[i:i + 8192].astype(np.float64) @ res.coef.T for i in range(0, len(Xva), 8192)]) \
        + res.intercept
    del Xva
    print(f"ridge done ({(time.time() - t0) / 60:.1f} min); alphas " +
          ", ".join(f"{t} {res.alphas[res.best[j]]:g}" for j, t in enumerate(tcols)), flush=True)

    # ---- MLPs: a fixed set of designs, all reported (no tuning on the reported benchmarks)
    # Regularisation in every design: Gaussian input noise and input dropout, LayerNorm + dropout in each hidden
    # layer, AdamW weight decay, gradient clipping, learning rate halved when the early-stopping loss stalls,
    # early stopping on benchmarks never trained on (best epoch restored), and an average over 3 seeds.
    T = len(tcols)
    wcols = [j for j, t in enumerate(tcols) if t != "b"] or list(range(T))   # early stopping on the weights
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Ys)
    tri = torch.from_numpy(np.flatnonzero(tr))
    sel_i, rep_i = torch.from_numpy(np.flatnonzero(sel)), torch.from_numpy(np.flatnonzero(rep))
    _, inv, cnt = np.unique(bench[tr], return_inverse=True, return_counts=True)
    balanced_w = torch.from_numpy(1.0 / cnt[inv])            # each training benchmark weighs the same
    designs = {   # name: (hidden sizes, input noise sd, input dropout, hidden dropout, lr, weight decay, sampling)
        "mlp_256x2": ((256, 256), 0.1, 0.2, 0.5, 1e-4, 0.05, "item"),
        "mlp_256x2_bench_balanced": ((256, 256), 0.1, 0.2, 0.5, 1e-4, 0.05, "benchmark"),
        "mlp_64x1": ((64,), 0.1, 0.2, 0.5, 1e-4, 0.05, "item"),
    }
    SEEDS = (0, 1, 2)

    class Net(torch.nn.Module):
        def __init__(self, hidden, noise, in_drop, drop):
            super().__init__()
            self.noise = noise
            layers, d_in = [torch.nn.Dropout(in_drop)], X.shape[1]
            for h in hidden:
                layers += [torch.nn.Linear(d_in, h), torch.nn.LayerNorm(h), torch.nn.GELU(), torch.nn.Dropout(drop)]
                d_in = h
            self.body = torch.nn.Sequential(*layers, torch.nn.Linear(d_in, T))

        def forward(self, x):
            if self.training and self.noise:
                x = x + self.noise * torch.randn_like(x)
            return self.body(x)

    def predict(net, idx):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(Xt[idx[i:i + 8192]]) for i in range(0, len(idx), 8192)])

    P_mlps = {}
    for name, (hidden, noise, in_drop, drop, lr, wd, sampling) in designs.items():
        preds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = Net(hidden, noise, in_drop, drop)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
            sel_loss = lambda: float(((predict(net, sel_i) - Yt[sel_i])[:, wcols] ** 2).mean())
            best = start_loss = sel_loss()
            best_ep, best_state, bad = 0, {k: v.clone() for k, v in net.state_dict().items()}, 0
            for ep in range(1, args.epochs + 1):
                net.train()
                if sampling == "benchmark":
                    perm = tri[torch.multinomial(balanced_w, len(tri), replacement=True)]
                else:
                    perm = tri[torch.randperm(len(tri))]
                for i in range(0, len(perm), 512):
                    b = perm[i:i + 512]
                    loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                v = sel_loss()
                sched.step(v)
                if v < best - 1e-5:
                    best, best_ep, best_state, bad = v, ep, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    bad += 1
                if bad >= 6:
                    break
            net.load_state_dict(best_state)
            preds.append(predict(net, rep_i).numpy().astype(np.float64))
            print(f"{name} seed {seed}: stopped after {ep} epochs, best epoch {best_ep}; early-stopping weight MSE "
                  f"{start_loss:.4f} (untrained) -> {best:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        P_mlps[name] = np.mean(preds, axis=0)

    # ---- compare on the reported validation benchmarks (never used for training or early stopping)
    Yv, gv = Y[rep], bench[rep]
    P_r = P_ridge * ys + ym
    rows_out = []
    for j, t in enumerate(tcols):
        row = {"target": t, "R2_mean": r2(Yv[:, j], np.full(len(Yv), ym[j])), "R2_ridge": r2(Yv[:, j], P_r[:, j]),
               "within_r_ridge": within_r(Yv[:, j], P_r[:, j], gv)}
        for name, P in P_mlps.items():
            pm = P[:, j] * ys[j] + ym[j]
            row[f"R2_{name}"] = r2(Yv[:, j], pm)
            row[f"within_r_{name}"] = within_r(Yv[:, j], pm, gv)
        rows_out.append(row)
    R = pd.DataFrame(rows_out)
    out = os.path.join(ROOT, "analysis", f"mirt_mlp_check_{args.run}_{args.fit}{'_debug' if args.debug_benchmarks else ''}.csv")
    R.to_csv(out, index=False)
    print("\nreported validation benchmarks (never used for training or early stopping):")
    print(R.round(4).set_index("target").T.to_string())
    print(f"saved {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
