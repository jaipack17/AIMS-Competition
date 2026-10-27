"""Stage 2 with a small MLP instead of the ridge: Qwen3-VL-8B embedding -> 1D IRT difficulty b (version B).

Same rows and targets as pipeline/ridge_train.py (embedded items of each fit's training benchmarks, target = that
fit's b), written in the same format (pred.parquet: item, benchmark, heldout, b_pred), so analysis/simulate_alc.py
can use it in place of the ridge (--ridge_dir data/mlp_b).

Network (one output), the regularisation found necessary in analysis/mirt_mlp_check.py:
  standardised input (StandardScaler of the training rows, applied per batch: no copy of the embeddings)
  -> Gaussian input noise (sd 0.1, training only) -> input dropout 0.2
  -> [Linear -> LayerNorm -> GELU -> dropout 0.5] per hidden layer -> Linear -> b (standardised target)
  AdamW (lr 1e-4, weight decay 0.05), batch 512, gradient clipping at norm 1, learning rate halved after 2 epochs
  without improvement, at most 40 epochs, early stopping (patience 6) on the mean within-benchmark Pearson r of
  b on 20% of the training benchmarks that are never trained on (seed 0); best epoch restored.
Prediction = average of 6 networks: 2 designs (64 and 256x256 hidden) x 3 seeds, fixed in advance.

Outputs, data/mlp_b/<run>/<fit>/: pred.parquet, summary.json, DONE   (finished fits are skipped)
  and mlp.npz: the whole ensemble for the submission (pack_ensemble; core.py runs it in numpy), written when every
  network's weights were saved.  The "final" fit (all benchmarks, no held-out ones) is the one bundled.
Progress is saved after every network (nets/<design>_s<seed>.npy + .json + .weights.npz): after a crash, shutdown
or kill, a rerun reuses the finished networks of the fold. While it runs, Windows is asked not to sleep (SetThreadExecutionState;
the request ends with the process).
Run:  python pipeline/mlp_b_train.py [--fits fold0,fold1,fold2,fold3,fold4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from fast_ridge import Scaler
from ridge_train import EMB_DIR, IRT_DIR, heldout_check

RUN = "versionB_4parts_logitnormal"
OUT_DIR = "data/mlp_b"
DESIGNS = {"h64": (64,), "h256x2": (256, 256)}
# deep residual design (--designs deep7): stem 4096 -> 1024, then residual blocks 1024 -> 512 -> 256 -> 256 -> 128
# -> 64 -> 32 (7 hidden layers), each Linear -> LayerNorm -> GELU -> dropout DEEP_DROP, plus a skip connection
# (identity, or a bias-free Linear when the width changes); then Linear 32 -> 1
DEEP_WIDTHS = (1024, 512, 256, 256, 128, 64, 32)
DEEP_DROP = 0.3
SEEDS = (0, 1, 2)
NOISE, IN_DROP, DROP, LR, WD, BATCH, PATIENCE = 0.1, 0.2, 0.5, 1e-4, 0.05, 512, 6


def load_all(emb_dir: str = EMB_DIR, keep: set | None = None):
    """All embeddings (or only the benchmarks in `keep`) into one pre-allocated float32 array, in the same rows and
    order as ridge_train.load_embeddings."""
    paths = [p for p in sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
             if keep is None or os.path.basename(p)[:-4] in keep]
    sizes = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            sizes.append(len(z["task_ids"]))
    X = np.empty((sum(sizes), 4096), dtype=np.float32)
    items, bench, at = [], [], 0
    for p, n in zip(paths, sizes):
        b = os.path.basename(p)[:-4]
        with np.load(p, allow_pickle=True) as z:
            X[at:at + n] = z["X"]
            items += [f"{b}::{i}" for i in z["task_ids"].astype(str)]
        bench += [b] * n
        at += n
    return X, np.array(items), np.array(bench)


def within_r(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> float:
    rs = [np.corrcoef(y[g == b], p[g == b])[0, 1] for b in np.unique(g)
          if (g == b).sum() >= 3 and np.std(y[g == b]) > 0 and np.std(p[g == b]) > 0]
    return float(np.mean(rs))


class Net(torch.nn.Module):
    def __init__(self, hidden: tuple, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        layers, d_in = [torch.nn.Dropout(IN_DROP)], len(mean)
        for h in hidden:
            layers += [torch.nn.Linear(d_in, h), torch.nn.LayerNorm(h), torch.nn.GELU(), torch.nn.Dropout(DROP)]
            d_in = h
        self.body = torch.nn.Sequential(*layers, torch.nn.Linear(d_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.scale
        if self.training:
            x = x + NOISE * torch.randn_like(x)
        return self.body(x).squeeze(-1)


def predict(net: Net, Xt: torch.Tensor, idx: np.ndarray) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(Xt[torch.from_numpy(idx[i:i + 8192])]).numpy() for i in range(0, len(idx), 8192)])


class ResNet(torch.nn.Module):
    """Deep residual MLP with the same input handling as Net (standardise, Gaussian noise, input dropout)."""

    def __init__(self, widths: tuple, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.in_drop = torch.nn.Dropout(IN_DROP)
        block = lambda a, b: torch.nn.Sequential(torch.nn.Linear(a, b), torch.nn.LayerNorm(b), torch.nn.GELU(),
                                                 torch.nn.Dropout(DEEP_DROP))
        self.stem = block(len(mean), widths[0])
        self.blocks = torch.nn.ModuleList(block(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.skips = torch.nn.ModuleList(torch.nn.Identity() if a == b else torch.nn.Linear(a, b, bias=False)
                                         for a, b in zip(widths[:-1], widths[1:]))
        self.head = torch.nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.scale
        if self.training:
            x = x + NOISE * torch.randn_like(x)
        h = self.stem(self.in_drop(x))
        for block, skip in zip(self.blocks, self.skips):
            h = block(h) + skip(h)
        return self.head(h).squeeze(-1)


def train_one(hidden, seed, Xt, ys, tr_idx, sel_idx, sel_bench, sel_y, mean, scale, log):
    torch.manual_seed(seed)
    net = ResNet(hidden, mean, scale) if hidden == DEEP_WIDTHS else Net(hidden, mean, scale)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    Yt = torch.from_numpy(ys)
    score = lambda: within_r(sel_y, predict(net, Xt, sel_idx), sel_bench)
    best, best_ep, best_state, bad = score(), 0, {k: v.clone() for k, v in net.state_dict().items()}, 0
    start = best
    tri = torch.from_numpy(tr_idx)
    for ep in range(1, 41):
        net.train()
        perm = tri[torch.randperm(len(tri))]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        s = score()
        sched.step(s)
        if s > best + 1e-4:
            best, best_ep, best_state, bad = s, ep, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    net.load_state_dict(best_state)
    log(f"    {hidden} seed {seed}: {ep} epochs, best epoch {best_ep}, early-stopping within-benchmark r "
        f"{start:.3f} (untrained) -> {best:.3f}")
    return net, {"hidden": list(hidden), "seed": seed, "epochs": ep, "best_epoch": best_ep, "sel_within_r": best}


def net_weights(net: Net) -> dict:
    """The eval-mode weights of a Net (dropout and input noise are off at prediction time), float64:
    W0, b0, g0, beta0 (Linear and LayerNorm of hidden layer 0), ..., Wout, bout."""
    lin = [m for m in net.body if isinstance(m, torch.nn.Linear)]
    ln = [m for m in net.body if isinstance(m, torch.nn.LayerNorm)]
    assert len(lin) == len(ln) + 1 and all(m.eps == 1e-5 for m in ln)
    f = lambda t: t.detach().numpy().astype(np.float64)
    w = {}
    for j, (a, n) in enumerate(zip(lin[:-1], ln)):
        w.update({f"W{j}": f(a.weight), f"b{j}": f(a.bias), f"g{j}": f(n.weight), f"beta{j}": f(n.bias)})
    w.update(Wout=f(lin[-1].weight), bout=f(lin[-1].bias))
    return w


def pack_ensemble(weights: list[dict], mean: np.ndarray, scale: np.ndarray, ym: float, ysd: float,
                  center: float) -> dict:
    """One npz for the submission: shared scaler, target scale, and net i's arrays as n{i}_<name>.
    b_hat = mean over nets of (net(standardised x) * y_sd + y_mean); center = mean training b (the smoke-test b_hat
    and the stretch centre, as the ridge intercept is for the ridge)."""
    out = {"mean": mean.astype(np.float64), "scale": scale.astype(np.float64), "y_mean": np.float64(ym),
           "y_sd": np.float64(ysd), "center": np.float64(center), "n_nets": np.int64(len(weights))}
    for i, w in enumerate(weights):
        out.update({f"n{i}_{k}": v for k, v in w.items()})
    return out


def keep_awake() -> None:
    """Windows: ask the system not to sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def main() -> None:
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="fold0,fold1,fold2,fold3,fold4")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--designs", default="h64,h256x2", help="comma-separated: h64, h256x2, deep7")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    designs = {n: (DEEP_WIDTHS if n == "deep7" else DESIGNS[n]) for n in args.designs.split(",")}
    t0 = time.time()
    log = lambda m: print(f"{m}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    X, items, bench = load_all()
    Xt = torch.from_numpy(X)
    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"]) \
        .set_index("item").b.reindex(items).to_numpy()
    log(f"embeddings {X.shape}")

    for fit in args.fits.split(","):
        out = os.path.join(args.out_dir, RUN, fit)
        if os.path.exists(os.path.join(out, "DONE")):
            log(f"[{fit}] already done, skipped")
            continue
        heldout = np.array([fold_of[b] == fit for b in bench])
        y = pd.read_parquet(os.path.join(IRT_DIR, RUN, fit, "items.parquet"), columns=["item", "b"]) \
            .set_index("item").b.reindex(items).to_numpy()
        train_b = sorted(set(bench[~heldout]))
        rng = np.random.default_rng(0)
        sel_b = set(rng.choice(train_b, size=round(0.2 * len(train_b)), replace=False))
        sel = ~heldout & np.isin(bench, list(sel_b))
        tr = ~heldout & ~sel
        assert not np.isnan(y[~heldout]).any()
        sc = Scaler.fit(X, np.flatnonzero(tr))
        ym, ysd = float(y[tr].mean()), float(y[tr].std())
        ys = np.nan_to_num((y - ym) / ysd).astype(np.float32)
        log(f"[{fit}] train {tr.sum():,} items, early-stopping {sel.sum():,} items ({len(sel_b)} benchmarks), "
            f"held out {heldout.sum():,}")
        tr_idx, sel_idx, all_idx = np.flatnonzero(tr), np.flatnonzero(sel), np.arange(len(X))
        preds, nets, weights = [], [], []
        net_dir = os.path.join(out, "nets")
        os.makedirs(net_dir, exist_ok=True)
        for name, hidden in designs.items():
            for seed in SEEDS:
                f_pred, f_info, f_w = (os.path.join(net_dir, f"{name}_s{seed}{ext}")
                                       for ext in (".npy", ".json", ".weights.npz"))
                if os.path.exists(f_pred) and os.path.exists(f_info):        # saved by an earlier, interrupted run
                    preds.append(np.load(f_pred))
                    nets.append(json.load(open(f_info)))
                    weights.append(dict(np.load(f_w)) if os.path.exists(f_w) else None)
                    log(f"    {hidden} seed {seed}: reused the saved network predictions")
                    continue
                net, info = train_one(hidden, seed, Xt, ys, tr_idx, sel_idx, bench[sel_idx], y[sel_idx],
                                      sc.mean, sc.scale, log)
                p = predict(net, Xt, all_idx) * ysd + ym
                w = net_weights(net) if isinstance(net, Net) else None
                if w is not None:
                    np.savez(f_w + ".tmp.npz", **w)
                    os.replace(f_w + ".tmp.npz", f_w)
                np.save(f_pred + ".tmp.npy", p)                                # write, then rename: never half a file
                os.replace(f_pred + ".tmp.npy", f_pred)
                with open(f_info, "w") as f:
                    json.dump(info, f)
                preds.append(p)
                nets.append(info)
                weights.append(w)
        b_pred = np.mean(preds, axis=0)
        if all(w is not None for w in weights):
            center = float(y[~heldout].mean())  # mean training b of this fit (all its training rows)
            np.savez(os.path.join(out, "mlp.npz"), **pack_ensemble(weights, sc.mean, sc.scale, ym, ysd, center))
        # the final fit has no held-out benchmarks
        check = heldout_check(b_pred[heldout], b_final[heldout], bench[heldout]) if heldout.any() else None
        sel_r = within_r(y[sel_idx], b_pred[sel_idx], bench[sel_idx])
        os.makedirs(out, exist_ok=True)
        pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout, "b_pred": b_pred}) \
            .to_parquet(os.path.join(out, "pred.parquet"), index=False)
        summary = {"run": RUN, "fit": fit, "designs": {k: list(v) for k, v in designs.items()}, "seeds": list(SEEDS),
                   "regularisation": {"input_noise_sd": NOISE, "input_dropout": IN_DROP, "dropout": DROP, "lr": LR,
                                      "weight_decay": WD, "batch": BATCH, "patience": PATIENCE, "grad_clip": 1.0},
                   "nets": nets, "ensemble_sel_within_r": sel_r, "heldout_vs_final_b": check,
                   "minutes": (time.time() - t0) / 60}
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        open(os.path.join(out, "DONE"), "w").close()
        log(f"[{fit}] ensemble: early-stopping within r {sel_r:.3f}" + ("" if check is None else
            f"; held-out benchmarks vs final b: within r {check['pearson_within_benchmark_mean']:.3f}, "
            f"overall r {check['pearson_all']:.3f}, rmse {check['rmse']:.3f}"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
