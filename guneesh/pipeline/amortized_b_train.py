"""Amortized difficulty (Truong et al., ICML 2025, "Reliable and Efficient Amortized Model-based Evaluation",
arXiv 2503.13335; code github.com/sangttruong/reeval, additional_analysis/amortized_irt/irt.py):
the map  b = f(embedding)  is trained directly on the responses, not on fitted difficulties.

Their calibration maximises the response likelihood with difficulty = f_phi(item embedding) (a linear layer by
default, an ELU MLP for large datasets; Adam lr 1e-3, weight decay 1e-4), abilities integrated over N(0, 1).
Here the abilities are already known from our 1D version-B fit of the same fold (theta of the 4 parts + zeta of the
subject-benchmark pair), so f is trained with them fixed:

    loss = - sum over binary cells (pair, item) of [ k log s(a_pair - f(x_item)) + (n - k) log s(-(a_pair - f(x_item))) ]
           / sum of n,         a_pair = theta(pair) + zeta(pair) from data/irt/<RUN>/<fit>/,  s = sigmoid

So an item with many responses pulls on f in proportion to its evidence, while AP's two-stage ridge copies every
fitted b (noisy for items with few responses) with the same weight. Their dataset-description prefix is not used:
test items only carry an anonymous benchmark_id.

Heads (both read the standardised embedding, StandardScaler of the training rows applied per batch):
  linear  the paper's default head, one Linear layer. Their optimiser (Adam lr 1e-3, weight decay 1e-4) overfits
          our training benchmarks from the first epoch (they test on random questions of seen datasets, we on
          unseen benchmarks), so the head is a penalised logistic ridge: loss + l2 * ||w||^2, Adam lr 1e-4, with
          l2 chosen from L2_GRID by the early-stopping log-likelihood (as the ridge chooses alpha by CV)
  mlp     pipeline/mlp_b_train.Net (64 hidden: noise, input dropout, LayerNorm, dropout 0.5), AdamW (lr 1e-4,
          weight decay 0.05), gradient clipping at 1; average of 3 seeds
Training: batches of 1024 items with all their cells, at most 40 epochs, early stopping (patience 5), best epoch
restored. Early stopping (and the linear head's l2) use the LEVEL-FREE log-likelihood per response of the cells of
20% of the training benchmarks never trained on (seed 0, the same benchmarks as mlp_b_train): before scoring, each
of those benchmarks gets its own best constant shift of b (one Newton fit per benchmark), as the submission's
benchmark shift delta does from the first labels. The plain log-likelihood is dominated by the level of each new
benchmark and rejected every model that ranked items better but sat a little high or low.

Rows, targets for the checks and output format as pipeline/ridge_train.py:
  data/amortized_b_<head>/<RUN>/<fit>/pred.parquet (item, benchmark, heldout, b_pred), summary.json, DONE
Run from the project root:  python pipeline/amortized_b_train.py --head linear|mlp [--fits fold0,...]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from fast_ridge import Scaler
from mlp_b_train import Net, load_all, within_r
from ridge_train import IRT_DIR, heldout_check

RUN = "versionB_4parts_logitnormal"
PARTS = {"name": "name", "harness": "harness_n", "provider": "provider_n", "effort": "effort_n"}
BATCH_ITEMS, MAX_EPOCHS, PATIENCE = 1024, 40, 5
L2_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)                # penalty per unit of the mean per-response loss


def pair_ability(fit: str) -> pd.Series:
    """theta(pair) + zeta(pair) of the 1D fit, with the fit's own (mean-centred) part tables; unseen keys add 0."""
    d = os.path.join(IRT_DIR, RUN, fit)
    S = pd.read_parquet(os.path.join(IRT_DIR, "subjects.parquet")).set_index("pair")
    a = pd.read_parquet(os.path.join(d, "zeta.parquet")).set_index("pair").zeta
    for p, col in PARTS.items():
        t = pd.read_parquet(os.path.join(d, f"theta_{p}.parquet")).set_index("key").theta
        keys = S[col].reindex(a.index).fillna("").astype(str)
        a = a + keys.map(t).fillna(0.0).to_numpy()
    return a


class Linear(torch.nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, bias: float):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.lin = torch.nn.Linear(len(mean), 1)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.constant_(self.lin.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin((x - self.mean) / self.scale).squeeze(-1)


class Cells:
    """Binary cells grouped by item row (CSR), so a batch of items gathers all of its cells."""

    def __init__(self, rows: np.ndarray, ability: np.ndarray, n: np.ndarray, k: np.ndarray, group: np.ndarray):
        order = np.argsort(rows, kind="stable")
        self.rows, self.a, self.n, self.k = rows[order], ability[order], n[order], k[order]
        self.group = group[order]                         # benchmark code of each cell
        self.items = np.unique(self.rows)
        self.start = np.searchsorted(self.rows, self.items)
        self.end = np.searchsorted(self.rows, self.items, side="right")

    def batch(self, item_pos: np.ndarray):
        idx = np.concatenate([np.arange(self.start[i], self.end[i]) for i in item_pos])
        local = np.repeat(np.arange(len(item_pos)), self.end[item_pos] - self.start[item_pos])
        t = lambda v, d=torch.float32: torch.from_numpy(np.ascontiguousarray(v)).to(d)
        return t(self.items[item_pos], torch.long), t(local, torch.long), t(self.a[idx]), t(self.n[idx]), t(self.k[idx])


def nll(net, Xt, cells: Cells, item_pos: np.ndarray) -> torch.Tensor:
    rows, local, a, n, k = cells.batch(item_pos)
    mu = a - net(Xt[rows])[local]
    ls = torch.nn.functional.logsigmoid
    return -(k * ls(mu) + (n - k) * ls(-mu)).sum() / n.sum()


def eval_ll(net, Xt, cells: Cells) -> float:
    """Level-free log-likelihood per response: each benchmark's cells get their own best shift d of b (Newton)."""
    net.eval()
    with torch.no_grad():
        b = np.concatenate([net(Xt[torch.from_numpy(cells.items[i:i + 8192])]).numpy()
                            for i in range(0, len(cells.items), 8192)]).astype(np.float64)
    mu0 = cells.a - b[np.searchsorted(cells.items, cells.rows)]
    tot = 0.0
    for g in np.unique(cells.group):
        m = cells.group == g
        z, n, k, d = mu0[m], cells.n[m].astype(np.float64), cells.k[m].astype(np.float64), 0.0
        for _ in range(50):                               # maximise sum k log s(z-d) + (n-k) log s(-(z-d)) over d
            p = 1 / (1 + np.exp(-(z - d)))
            step = (k - n * p).sum() / max((n * p * (1 - p)).sum(), 1e-12)
            d -= step
            if abs(step) < 1e-10:
                break
        mu = z - d
        tot += float((k * -np.logaddexp(0, -mu) + (n - k) * -np.logaddexp(0, mu)).sum())
    return tot / float(cells.n.sum())


def train(head: str, seed: int, Xt, tr: Cells, sel: Cells, mean, scale, bias, log, l2: float = 0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if head == "linear":
        net = Linear(mean, scale, bias)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        clip = None
    else:
        net = Net((64,), mean, scale)
        with torch.no_grad():
            net.body[-1].bias.fill_(bias)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
        clip = 1.0
    best = start = eval_ll(net, Xt, sel)
    best_ep, best_state, bad = 0, {k: v.clone() for k, v in net.state_dict().items()}, 0
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = rng.permutation(len(tr.items))
        for i in range(0, len(perm), BATCH_ITEMS):
            loss = nll(net, Xt, tr, perm[i:i + BATCH_ITEMS])
            if head == "linear":
                loss = loss + l2 * (net.lin.weight ** 2).sum()
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
        s = eval_ll(net, Xt, sel)
        if s > best + 1e-6:
            best, best_ep, best_state, bad = s, ep, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    net.load_state_dict(best_state)
    log(f"    {head}{f' l2={l2:g}' if head == 'linear' else ''} seed {seed}: {ep} epochs, best epoch {best_ep}; "
        f"early-stopping level-free log-lik per response {start:.5f} (start) -> {best:.5f}")
    return net, {"seed": seed, "l2": l2, "epochs": ep, "best_epoch": best_ep, "sel_ll_start": start, "sel_ll": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=["linear", "mlp"])
    ap.add_argument("--fits", default="fold0,fold1,fold2,fold3,fold4")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--debug_benchmarks", type=int, default=0, help="code check only: keep this many benchmarks")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = f"data/amortized_b_{args.head}{'_debug' if args.debug_benchmarks else ''}"
    seeds = (0,) if args.head == "linear" else (0, 1, 2)
    t0 = time.time()
    log = lambda m: print(f"{m}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    folds = pd.read_csv(os.path.join(IRT_DIR, "folds.csv"))
    keep = set(sorted(folds.benchmark)[:args.debug_benchmarks]) if args.debug_benchmarks else None
    X, items, bench = load_all(keep=keep)
    Xt = torch.from_numpy(X)
    row_of = pd.Series(np.arange(len(items)), index=items)
    fold_of = dict(zip(folds.benchmark, "fold" + folds.fold.astype(str)))
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"]) \
        .set_index("item").b.reindex(items).to_numpy()
    cells_all = pd.read_parquet(os.path.join(IRT_DIR, "cells_binary.parquet"), columns=["pair", "item", "n", "k", "benchmark"],
                                filters=[("benchmark", "in", sorted(keep))] if keep else None)
    cells_all = cells_all[cells_all.item.isin(row_of.index)]
    log(f"embeddings {X.shape}, binary cells on embedded items {len(cells_all):,}")

    for fit in args.fits.split(","):
        out = os.path.join(out_dir, RUN, fit)
        if os.path.exists(os.path.join(out, "DONE")):
            log(f"[{fit}] already done, skipped")
            continue
        heldout = np.array([fold_of[b] == fit for b in bench])
        train_b = sorted(set(bench[~heldout]))
        sel_b = set(np.random.default_rng(0).choice(train_b, size=round(0.2 * len(train_b)), replace=False))
        tr_rows = np.flatnonzero(~heldout & ~np.isin(bench, list(sel_b)))
        ab = pair_ability(fit)
        C = cells_all[~cells_all.benchmark.map(fold_of).eq(fit)]
        a = C.pair.map(ab).to_numpy(np.float32)
        assert not np.isnan(a).any(), "training cells whose pair has no ability"
        r = row_of[C.item].to_numpy()
        in_sel = C.benchmark.isin(sel_b).to_numpy()
        g = pd.factorize(C.benchmark)[0]
        n_, k_ = C.n.to_numpy(np.float32), C.k.to_numpy(np.float32)
        tr = Cells(r[~in_sel], a[~in_sel], n_[~in_sel], k_[~in_sel], g[~in_sel])
        sel = Cells(r[in_sel], a[in_sel], n_[in_sel], k_[in_sel], g[in_sel])
        sc = Scaler.fit(X, tr_rows)
        b_fit = pd.read_parquet(os.path.join(IRT_DIR, RUN, fit, "items.parquet"), columns=["item", "b"]) \
            .set_index("item").b.reindex(items).to_numpy()
        bias = float(np.nanmean(b_fit[tr_rows]))
        log(f"[{fit}] train {len(tr.items):,} items / {len(tr.rows):,} cells; early-stopping {len(sel.items):,} items "
            f"({len(sel_b)} benchmarks); held out {heldout.sum():,} items")
        preds, runs = [], []
        if args.head == "linear":   # choose the penalty on the early-stopping benchmarks, keep that model
            fitted = [train("linear", 0, Xt, tr, sel, sc.mean, sc.scale, bias, log, l2) for l2 in L2_GRID]
            runs = [info for _, info in fitted]
            nets = [max(fitted, key=lambda f: f[1]["sel_ll"])[0]]
        else:
            nets = []
            for seed in seeds:
                net, info = train(args.head, seed, Xt, tr, sel, sc.mean, sc.scale, bias, log)
                nets.append(net)
                runs.append(info)
        for net in nets:
            net.eval()
            with torch.no_grad():
                preds.append(np.concatenate([net(Xt[i:i + 8192]).numpy() for i in range(0, len(X), 8192)]))
        b_pred = np.mean(preds, axis=0).astype(np.float64)
        check = heldout_check(b_pred[heldout], b_final[heldout], bench[heldout])
        sel_rows = np.flatnonzero(np.isin(bench, list(sel_b)))
        sel_r = within_r(b_fit[sel_rows], b_pred[sel_rows], bench[sel_rows])
        os.makedirs(out, exist_ok=True)
        pd.DataFrame({"item": items, "benchmark": bench, "heldout": heldout, "b_pred": b_pred}) \
            .to_parquet(os.path.join(out, "pred.parquet"), index=False)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"run": RUN, "fit": fit, "head": args.head, "runs": runs, "sel_within_r_vs_fit_b": sel_r,
                       "heldout_vs_final_b": check, "minutes": (time.time() - t0) / 60}, f, indent=1)
        open(os.path.join(out, "DONE"), "w").close()
        log(f"[{fit}] early-stopping benchmarks: within r vs the fit's b {sel_r:.3f}; held-out benchmarks vs final b: "
            f"within r {check['pearson_within_benchmark_mean']:.3f}, overall r {check['pearson_all']:.3f}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
