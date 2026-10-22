"""Version B IRT: LEGO-MB-style mixed binary/continuous 1PL, fitted by SVI exactly as Agent Psychometrics does.

Based on agent-psychometrics swebench_irt/train_model_scaffold_shared.py (MIT License, Copyright (c) 2026 Fulcrum):
ModelScaffold1PL (HalfNormal(1) hierarchical scales with Delta guides, Normal mean-field guides, mean-centred
abilities, Normal(0, sigma_b) item difficulty) and train_svi (Trace_ELBO, ClippedAdam lr 0.01, betas
(0.9, 0.999), clip_norm 5, full batch, 2000 epochs). Changes, all for this competition:
  * ability = theta_name + theta_harness + theta_provider + theta_effort (+ zeta per subject-benchmark pair).
    AP has 2 parts (model + scaffold); blank harness / provider / effort contribute 0.
  * zeta_{pair} ~ Normal(0, sigma_zeta): the per-benchmark ability deviation of LEGO-MB (theta_{i,m} = theta_i +
    zeta_{i,m}), with independent priors instead of LEGO's LKJ covariance (new benchmarks have no learned
    correlations). sigma_zeta is the spread a test-time update from revealed labels should expect.
  * likelihood (LEGO-MB with a_j = 1, i.e. its Rasch case):
      binary items      Bernoulli(sigmoid(mu)), repeated attempts grouped as Binomial(n, .)  (identical likelihood)
      continuous items  --cont logitnormal : logit(y) ~ Normal(mu, sigma_m^2), sigma_m per benchmark
                                             (LEGO uses per item; most of our items have too few responses)
                        --cont soft        : y * log p + (1 - y) * log(1 - p)  (fractional-Bernoulli alternative)
      mu = ability - b_item.
Fits: --fold 0..4 leaves that fold's binary benchmarks out entirely; --fold final uses everything; --fold all
runs 0,1,2,3,4,final in turn, skipping finished ones. Each fit checkpoints every 200 epochs and resumes.

Outputs, on this computer, in data/irt/<run>/<fit>/:
  items.parquet (item, b, b_scale), theta_<part>.parquet (key, theta, scale), zeta.parquet (pair, zeta, scale),
  hyper.json (sigmas, per-benchmark sigma_m, config, data sizes, minutes), losses.csv, DONE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

PARTS = ["name", "harness", "provider", "effort"]
PART_COL = {"name": "name", "harness": "harness_n", "provider": "provider_n", "effort": "effort_n"}


def set_seed(seed: int) -> None:  # verbatim from AP
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_fit_data(irt_dir: str, fold: str, parts: list[str]):
    """Index the cells used by one fit. Held-out binary benchmarks are dropped completely."""
    S = pd.read_parquet(f"{irt_dir}/subjects.parquet")
    B = pd.read_parquet(f"{irt_dir}/cells_binary.parquet")
    C = pd.read_parquet(f"{irt_dir}/cells_cont.parquet")
    if fold != "final":
        F = pd.read_csv(f"{irt_dir}/folds.csv")
        held = set(F.benchmark[F.fold == int(fold)])
        B = B[~B.benchmark.isin(held)]
    pairs = pd.Index(sorted(set(B.pair) | set(C.pair)))
    items = pd.Index(sorted(set(B.item) | set(C.item)))
    Sp = S.set_index("pair").loc[pairs]
    part_keys, part_idx = {}, {}
    for p in parts:
        v = Sp[PART_COL[p]].fillna("").astype(str)
        keys = pd.Index(sorted(set(v) - {""}))
        part_keys[p] = keys
        # blank -> index len(keys), which points at a fixed 0
        idx = keys.get_indexer(v)
        part_idx[p] = torch.tensor(np.where(idx < 0, len(keys), idx), dtype=torch.long)
    cont_b = pd.Index(sorted(C.benchmark.unique()))
    t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
    data = {
        "b_pair": t(pairs.get_indexer(B.pair), torch.long), "b_item": t(items.get_indexer(B.item), torch.long),
        "b_n": t(B.n), "b_k": t(B.k),
        "c_pair": t(pairs.get_indexer(C.pair), torch.long), "c_item": t(items.get_indexer(C.item), torch.long),
        "c_bench": t(cont_b.get_indexer(C.benchmark), torch.long),
        "c_n": t(C.n), "c_sy": t(C.sy), "c_sz": t(C.sz), "c_sz2": t(C.sz2),
        "part_idx": part_idx,
    }
    meta = {"pairs": pairs, "items": items, "part_keys": part_keys, "cont_benchmarks": cont_b,
            "n_binary_cells": len(B), "n_binary_responses": int(B.n.sum()),
            "n_cont_cells": len(C), "n_cont_responses": int(C.n.sum())}
    return data, meta


class MixedScaffold1PL:
    def __init__(self, n_parts: dict, n_pairs: int, n_items: int, n_cont_bench: int, cont: str):
        self.n_parts, self.n_pairs, self.n_items, self.n_cont_bench, self.cont = n_parts, n_pairs, n_items, n_cont_bench, cont

    def _ability(self, thetas: dict, zeta, d):
        a = zeta
        for p, th in thetas.items():
            a = a + torch.cat([th - th.mean(), th.new_zeros(1)])[d["part_idx"][p]]
        return a  # one ability per subject-benchmark pair

    def model(self, d):
        one = d["b_n"].new_tensor(1.0)
        zero = d["b_n"].new_tensor(0.0)
        thetas = {}
        for p, n in self.n_parts.items():
            sig = pyro.sample(f"sigma_theta_{p}", dist.HalfNormal(one))
            with pyro.plate(f"{p}_plate", n):
                thetas[p] = pyro.sample(f"theta_{p}_raw", dist.Normal(zero, sig))
        sigma_zeta = pyro.sample("sigma_zeta", dist.HalfNormal(one))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(one))
        with pyro.plate("pairs", self.n_pairs):
            zeta = pyro.sample("zeta", dist.Normal(zero, sigma_zeta))
        with pyro.plate("items", self.n_items):
            b = pyro.sample("b", dist.Normal(zero, sigma_b))
        ability = self._ability(thetas, zeta, d)
        mu_b = ability[d["b_pair"]] - b[d["b_item"]]
        with pyro.plate("obs_binary", mu_b.size(0)):
            pyro.sample("y_binary", dist.Binomial(total_count=d["b_n"], logits=mu_b), obs=d["b_k"])
        if d["c_n"].numel():
            mu_c = ability[d["c_pair"]] - b[d["c_item"]]
            n = d["c_n"]
            if self.cont == "logitnormal":
                with pyro.plate("cont_benchmarks", self.n_cont_bench):
                    sigma_m = pyro.sample("sigma_cont", dist.HalfNormal(one))
                s = sigma_m[d["c_bench"]]
                # sum over the cell's n responses of log Normal(z | mu, s), from sum z and sum z^2
                ll = (-n * (torch.log(s) + 0.5 * math.log(2 * math.pi))
                      - (d["c_sz2"] - 2 * mu_c * d["c_sz"] + n * mu_c ** 2) / (2 * s ** 2))
            else:
                ll = d["c_sy"] * torch.nn.functional.logsigmoid(mu_c) + (n - d["c_sy"]) * torch.nn.functional.logsigmoid(-mu_c)
            pyro.factor("y_cont", ll.sum())

    def guide(self, d):
        dev = d["b_n"].device
        for p, n in self.n_parts.items():
            sq = pyro.param(f"sigma_theta_{p}_q", torch.tensor(1.0, device=dev), constraint=constraints.positive)
            pyro.sample(f"sigma_theta_{p}", dist.Delta(sq))
            loc = pyro.param(f"loc_theta_{p}_raw", torch.zeros(n, device=dev))
            scale = pyro.param(f"scale_theta_{p}_raw", torch.ones(n, device=dev), constraint=constraints.positive)
            with pyro.plate(f"{p}_plate", n):
                pyro.sample(f"theta_{p}_raw", dist.Normal(loc, scale))
        for nm in ("sigma_zeta", "sigma_b"):
            q = pyro.param(f"{nm}_q", torch.tensor(1.0, device=dev), constraint=constraints.positive)
            pyro.sample(nm, dist.Delta(q))
        loc_z = pyro.param("loc_zeta", torch.zeros(self.n_pairs, device=dev))
        scale_z = pyro.param("scale_zeta", torch.ones(self.n_pairs, device=dev), constraint=constraints.positive)
        with pyro.plate("pairs", self.n_pairs):
            pyro.sample("zeta", dist.Normal(loc_z, scale_z))
        loc_b = pyro.param("loc_b", torch.zeros(self.n_items, device=dev))
        scale_b = pyro.param("scale_b", torch.ones(self.n_items, device=dev), constraint=constraints.positive)
        with pyro.plate("items", self.n_items):
            pyro.sample("b", dist.Normal(loc_b, scale_b))
        if self.cont == "logitnormal" and d["c_n"].numel():
            sm = pyro.param("sigma_cont_q", torch.ones(self.n_cont_bench, device=dev), constraint=constraints.positive)
            with pyro.plate("cont_benchmarks", self.n_cont_bench):
                pyro.sample("sigma_cont", dist.Delta(sm))


def fit_one(args, fold: str) -> None:
    out = os.path.join(args.irt_dir, args.run, f"fold{fold}" if fold != "final" else "final")
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"[{fold}] already done -> {out}")
        return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    set_seed(args.seed)
    pyro.clear_param_store()
    data, meta = load_fit_data(args.irt_dir, fold, args.parts)
    used_parts = [p for p in args.parts if len(meta["part_keys"][p])]  # a part with no values adds 0
    m = MixedScaffold1PL({p: len(meta["part_keys"][p]) for p in used_parts}, len(meta["pairs"]), len(meta["items"]),
                         len(meta["cont_benchmarks"]), args.cont)
    print(f"[{fold}] binary cells {meta['n_binary_cells']:,} ({meta['n_binary_responses']:,} responses), "
          f"continuous cells {meta['n_cont_cells']:,} ({meta['n_cont_responses']:,}), items {len(meta['items']):,}, "
          f"pairs {len(meta['pairs']):,}, " + ", ".join(f"{p} {len(meta['part_keys'][p])}" for p in args.parts), flush=True)

    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    optimizer = ClippedAdam({"lr": args.lr, "betas": (0.9, 0.999), "clip_norm": 5.0})
    svi = SVI(m.model, m.guide, optimizer, loss=Trace_ELBO())
    ck_params, ck_optim, ck_losses = (os.path.join(out, f) for f in ("ckpt_params.pt", "ckpt_optim.pt", "losses.csv"))
    losses, start = [], 1
    if os.path.exists(ck_params) and os.path.exists(ck_optim) and os.path.exists(ck_losses):
        svi.step(data)  # creates the params and optimizer slots, then overwrite them with the checkpoint
        pyro.get_param_store().load(ck_params)
        optimizer.load(ck_optim)
        losses = pd.read_csv(ck_losses).loss.tolist()
        start = len(losses) + 1
        print(f"[{fold}] resumed at epoch {start}", flush=True)
    for ep in range(start, args.epochs + 1):
        losses.append(float(svi.step(data)))
        if ep == 1 or ep % 200 == 0 or ep == args.epochs:
            print(f"[{fold}] epoch {ep:5d}  loss={losses[-1]:,.2f}  {(time.time() - t0) / 60:.1f} min", flush=True)
            pyro.get_param_store().save(ck_params)
            optimizer.save(ck_optim)
            pd.DataFrame({"loss": losses}).to_csv(ck_losses, index=False)

    ps = pyro.get_param_store()
    g = lambda k: ps[k].detach().cpu().numpy()
    pd.DataFrame({"item": meta["items"], "b": g("loc_b"), "b_scale": g("scale_b")}).to_parquet(os.path.join(out, "items.parquet"))
    for p in args.parts:
        if p in used_parts:
            loc = g(f"loc_theta_{p}_raw")
            th = pd.DataFrame({"key": meta["part_keys"][p], "theta": loc - loc.mean(), "scale": g(f"scale_theta_{p}_raw")})
        else:
            th = pd.DataFrame({"key": pd.Series(dtype=str), "theta": pd.Series(dtype=float), "scale": pd.Series(dtype=float)})
        th.to_parquet(os.path.join(out, f"theta_{p}.parquet"))
    pd.DataFrame({"pair": meta["pairs"], "zeta": g("loc_zeta"), "scale": g("scale_zeta")}).to_parquet(os.path.join(out, "zeta.parquet"))
    hyper = {k: float(g(f"{k}_q")) for k in [f"sigma_theta_{p}" for p in used_parts] + ["sigma_zeta", "sigma_b"]}
    if args.cont == "logitnormal" and "sigma_cont_q" in ps.keys():
        hyper["sigma_cont"] = dict(zip(meta["cont_benchmarks"], g("sigma_cont_q").tolist()))
    hyper.update(config=vars(args) | {"fold": fold}, minutes=(time.time() - t0) / 60,
                 **{k: v for k, v in meta.items() if k.startswith("n_")})
    with open(os.path.join(out, "hyper.json"), "w") as f:
        json.dump(hyper, f, indent=1, default=str)
    for f in (ck_params, ck_optim):
        os.remove(f)
    open(os.path.join(out, "DONE"), "w").close()
    print(f"[{fold}] saved -> {out} ({hyper['minutes']:.0f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--irt_dir", default="data/irt")
    ap.add_argument("--run", default="versionB_4parts_logitnormal")
    ap.add_argument("--fold", default="all", help="0..4, final, or all")
    ap.add_argument("--parts", default="name,harness,provider,effort")
    ap.add_argument("--cont", default="logitnormal", choices=["logitnormal", "soft"])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.parts = [p for p in args.parts.split(",") if p]
    if args.threads:
        torch.set_num_threads(args.threads)
    pyro.enable_validation(False)  # skips per-step support checks on ~29M cells; does not change results
    for fold in (["0", "1", "2", "3", "4", "final"] if args.fold == "all" else [args.fold]):
        fit_one(args, fold)


if __name__ == "__main__":
    main()
