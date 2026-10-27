"""Multidimensional IRT (MIRT) fits: version B (pipeline/irt_fit.py) extended to K latent dimensions.

Same data (data/irt: cells_binary, cells_cont, subjects, folds), same likelihoods (Binomial for binary cells,
logit-normal for continuous cells), same subject parts (name, harness, provider, effort; blank -> 0, mean-centred),
same fitting (Pyro SVI, Trace_ELBO, ClippedAdam lr 0.01, betas (0.9, 0.999), clip_norm 5, full batch, 2000 epochs,
mean-field Normal guides, Delta guides for the HalfNormal(1) scales), same folds. Only the logit mu changes.

--form nested   (general 1D ability plus K extra dimensions; K = 0 is exactly version B; cf. arXiv 2604.01418)
    mu = theta(pair) + zeta(pair) + u(pair) . v(item) - b(item)
    theta(pair) = sum over parts of theta_part[key]                          scalar, as in version B
    u(pair)     = sum over parts of u_part[key]                              K-vector
    priors:  theta_part ~ N(0, s_part^2), zeta ~ N(0, s_zeta^2), b ~ N(0, s_b^2)          (version B)
             u_part[key, d] ~ N(0, s_u_part^2)   (s_u_part ~ HalfNormal(1), one per part)
             v[item, d] ~ N(0, 1)                (fixed unit scale: fixes the scale trade-off between u and v)

--form m2pl     (standard compensatory multidimensional 2PL; K = 1 is the 2PL; Reckase 2009, tinyBenchmarks)
    mu = a(item) . theta(pair) - b(item)
    theta(pair) = sum over parts of theta_part[key] + zeta(pair)            K-vectors
    priors:  theta_part[key, d] ~ N(0, s_part^2), zeta[pair, d] ~ N(0, s_zeta^2), b ~ N(0, s_b^2)
             a[item, d] ~ N(0, 1)

Rotation: u.v and a.theta are unchanged by any rotation of the latent space, so predictions do not depend on it;
nothing is rotated here (BenchMIRT rotates only to interpret the dimensions).

Initialisation (AP's MIRT uses a PCA warm start, py_irt "mirt_pca"): the 1D version-B fit of the same fold gives the
start of theta, zeta and b (nested), or of dimension 1 of theta / zeta with a[:, 0] = 1 (m2pl). The other item
loadings start from a truncated SVD of the binary residuals k/n - sigmoid(1D logit) (items x pairs, sparse), scaled
to RMS --init_scale; their abilities start at 0. Guide scales of the new vectors start at --init_guide_scale.

Model selection numbers saved in hyper.json (AP swebench_irt/compare_dims.py computes AIC / BIC the same way):
log-likelihood of all training responses at the posterior means, number of parameters, AIC = 2k - 2LL,
BIC = k ln(N) - 2LL with N = number of responses. The final choice of K is made on held-out benchmarks
(analysis/simulate_alc.py), not on these.

Outputs, data/irt/mirt_<form>_K<K>/<fit>/:
  items.parquet   item, b, b_scale, v1..vK (nested) or a1..aK (m2pl), and their scales
  theta_<part>.parquet  key, theta (nested only), u1..uK (nested) or theta1..thetaK (m2pl)
  zeta.parquet    pair, zeta (nested) or zeta1..zetaK (m2pl)
  hyper.json, losses.csv, DONE, ckpt_params.pt / ckpt_optim.pt (kept: rerun with a larger --epochs to extend)

Run:  python pipeline/mirt_fit.py --form nested --K 2 --fold 0 [--epochs 2000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irt_fit import PARTS, load_fit_data  # noqa: E402

BASE_RUN = "versionB_4parts_logitnormal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only: some CUDA backward ops (index_add for the gathers) have no deterministic kernel
    torch.use_deterministic_algorithms(True, warn_only=True)


def centred(x: torch.Tensor) -> torch.Tensor:
    """Mean-centre over keys and append a zero row for the blank key (index n), as version B does."""
    return torch.cat([x - x.mean(0), x.new_zeros((1,) + x.shape[1:])])


class MixedMIRT:
    def __init__(self, form: str, K: int, n_parts: dict, n_pairs: int, n_items: int, n_cont_bench: int):
        self.form, self.K = form, K
        self.n_parts, self.n_pairs, self.n_items, self.n_cont_bench = n_parts, n_pairs, n_items, n_cont_bench

    # ------------------------------------------------------------------ logit
    def logits(self, lat: dict, d: dict, pair: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        if self.form == "nested":
            th = lat["zeta"]
            u = 0.0
            for p in self.n_parts:
                th = th + centred(lat[f"theta_{p}"])[d["part_idx"][p]]
                if self.K:
                    u = u + centred(lat[f"u_{p}"])[d["part_idx"][p]]
            mu = th[pair] - lat["b"][item]
            if self.K:
                mu = mu + (u[pair] * lat["v"][item]).sum(-1)
            return mu
        th = lat["zeta"]                                  # [pairs, K]
        for p in self.n_parts:
            th = th + centred(lat[f"theta_{p}"])[d["part_idx"][p]]
        return (lat["a"][item] * th[pair]).sum(-1) - lat["b"][item]

    def _shapes(self) -> dict:
        """name -> (shape, prior scale name or None for the fixed N(0, 1))"""
        K, s = self.K, {}
        for p, n in self.n_parts.items():
            if self.form == "nested":
                s[f"theta_{p}"] = ((n,), f"sigma_theta_{p}")
                if K:
                    s[f"u_{p}"] = ((n, K), f"sigma_u_{p}")
            else:
                s[f"theta_{p}"] = ((n, K), f"sigma_theta_{p}")
        s["zeta"] = (((self.n_pairs,) if self.form == "nested" else (self.n_pairs, K)), "sigma_zeta")
        s["b"] = ((self.n_items,), "sigma_b")
        if self.form == "nested" and K:
            s["v"] = ((self.n_items, K), None)
        if self.form == "m2pl":
            s["a"] = ((self.n_items, K), None)
        return s

    # ------------------------------------------------------------------ model / guide
    def model(self, d):
        one, zero = d["b_n"].new_tensor(1.0), d["b_n"].new_tensor(0.0)
        lat = {}
        for name, (shape, sig_name) in self._shapes().items():
            sig = pyro.sample(sig_name, dist.HalfNormal(one)) if sig_name else one
            lat[name] = pyro.sample(name, dist.Normal(zero, sig).expand(shape).to_event(len(shape)))
        mu_b = self.logits(lat, d, d["b_pair"], d["b_item"])
        with pyro.plate("obs_binary", mu_b.size(0)):
            pyro.sample("y_binary", dist.Binomial(total_count=d["b_n"], logits=mu_b), obs=d["b_k"])
        if d["c_n"].numel():
            with pyro.plate("cont_benchmarks", self.n_cont_bench):
                sigma_m = pyro.sample("sigma_cont", dist.HalfNormal(one))
            pyro.factor("y_cont", self.cont_ll(self.logits(lat, d, d["c_pair"], d["c_item"]), sigma_m, d).sum())

    @staticmethod
    def cont_ll(mu_c, sigma_m, d):
        s, n = sigma_m[d["c_bench"]], d["c_n"]
        return (-n * (torch.log(s) + 0.5 * math.log(2 * math.pi))
                - (d["c_sz2"] - 2 * mu_c * d["c_sz"] + n * mu_c ** 2) / (2 * s ** 2))

    def guide(self, d):
        dev = d["b_n"].device
        for name, (shape, sig_name) in self._shapes().items():
            if sig_name:
                q = pyro.param(f"{sig_name}_q", torch.tensor(1.0, device=dev), constraint=constraints.positive)
                pyro.sample(sig_name, dist.Delta(q))
            loc = pyro.param(f"loc_{name}", torch.zeros(shape, device=dev))
            scale = pyro.param(f"scale_{name}", torch.ones(shape, device=dev), constraint=constraints.positive)
            pyro.sample(name, dist.Normal(loc, scale).to_event(len(shape)))
        if d["c_n"].numel():
            sm = pyro.param("sigma_cont_q", torch.ones(self.n_cont_bench, device=dev), constraint=constraints.positive)
            with pyro.plate("cont_benchmarks", self.n_cont_bench):
                pyro.sample("sigma_cont", dist.Delta(sm))

    def n_params(self) -> int:
        """Free latent values (point estimates), as counted by AP's compare_dims (abilities + item parameters)."""
        return int(sum(np.prod(shape) for shape, _ in self._shapes().values())) + self.n_cont_bench


# ---------------------------------------------------------------------- initialisation
def init_from_1d(m: MixedMIRT, meta: dict, d: dict, fit: str, args) -> None:
    """Warm start from the version-B 1D fit of the same fold, and SVD of its binary residuals for the loadings."""
    src = os.path.join(args.irt_dir, BASE_RUN, fit)
    ps = pyro.get_param_store()
    dev = ps["loc_b"].device                              # the params' device; d is the CPU copy
    b1 = pd.read_parquet(os.path.join(src, "items.parquet")).set_index("item").b.reindex(meta["items"]).fillna(0.0)
    z1 = pd.read_parquet(os.path.join(src, "zeta.parquet")).set_index("pair").zeta.reindex(meta["pairs"]).fillna(0.0)
    th1 = {}
    for p in m.n_parts:
        t = pd.read_parquet(os.path.join(src, f"theta_{p}.parquet")).set_index("key").theta
        th1[p] = t.reindex(meta["part_keys"][p]).fillna(0.0).to_numpy(np.float32)
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
    with torch.no_grad():
        ps["loc_b"].copy_(T(b1))
        if m.form == "nested":
            ps["loc_zeta"].copy_(T(z1))
            for p in m.n_parts:
                ps[f"loc_theta_{p}"].copy_(T(th1[p]))
        else:
            ps["loc_zeta"][:, 0] = T(z1)
            for p in m.n_parts:
                ps[f"loc_theta_{p}"][:, 0] = T(th1[p])
        # 1D logit of every binary cell, then residuals k/n - p in an items x pairs sparse matrix
        theta_pair = z1.to_numpy(np.float32).copy()
        for p in m.n_parts:
            idx = d["part_idx"][p].numpy()
            vals = np.r_[th1[p] - th1[p].mean(), 0.0].astype(np.float32)
            theta_pair += vals[idx]
        bp, bi = d["b_pair"].cpu().numpy(), d["b_item"].cpu().numpy()
        z = theta_pair[bp] - b1.to_numpy(np.float32)[bi]
        r = d["b_k"].cpu().numpy() / d["b_n"].cpu().numpy() - 1 / (1 + np.exp(-z))
        n_svd = m.K if m.form == "nested" else m.K - 1
        if n_svd > 0:
            import scipy.sparse as sp
            from scipy.sparse.linalg import svds
            R = sp.csr_matrix((r.astype(np.float64), (bi, bp)), shape=(m.n_items, m.n_pairs))
            U, S, _ = svds(R, k=n_svd, random_state=args.seed)
            U = U[:, np.argsort(-S)] * math.sqrt(m.n_items) * args.init_scale   # RMS init_scale per column
            if m.form == "nested":
                ps["loc_v"].copy_(T(U))
            else:
                ps["loc_a"][:, 1:] = T(U)
        if m.form == "m2pl":
            ps["loc_a"][:, 0] = 1.0
        for name, (shape, _) in m._shapes().items():   # new vector parameters start with small guide scales
            if len(shape) == 2:
                ps[f"scale_{name}"].fill_(args.init_guide_scale)


# ---------------------------------------------------------------------- fit
def loglik(m: MixedMIRT, d: dict) -> tuple[float, int]:
    """Training log-likelihood at the posterior means (binary: sum k log p + (n-k) log(1-p); continuous: logit-normal)."""
    ps = pyro.get_param_store()
    lat = {name: ps[f"loc_{name}"] for name in m._shapes()}
    with torch.no_grad():
        mu = m.logits(lat, d, d["b_pair"], d["b_item"])
        ll = float((d["b_k"] * torch.nn.functional.logsigmoid(mu)
                    + (d["b_n"] - d["b_k"]) * torch.nn.functional.logsigmoid(-mu)).sum())
        n = int(d["b_n"].sum())
        if d["c_n"].numel():
            ll += float(m.cont_ll(m.logits(lat, d, d["c_pair"], d["c_item"]), ps["sigma_cont_q"], d).sum())
            n += int(d["c_n"].sum())
    return ll, n


def fit_one(args, fold: str) -> None:
    fit = f"fold{fold}" if fold != "final" else "final"
    out = os.path.join(args.out_root or args.irt_dir, f"mirt_{args.form}_K{args.K}", fit)
    done, ck_losses = os.path.join(out, "DONE"), os.path.join(out, "losses.csv")
    if os.path.exists(done):
        n_done = len(pd.read_csv(ck_losses)) if os.path.exists(ck_losses) else 0
        if n_done >= args.epochs or not os.path.exists(os.path.join(out, "ckpt_params.pt")):
            print(f"[{args.form} K={args.K} {fit}] already done ({n_done} epochs) -> {out}")
            return
        os.remove(done)                                   # more epochs asked: continue from the kept checkpoint
        print(f"[{args.form} K={args.K} {fit}] extending from {n_done} to {args.epochs} epochs", flush=True)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    set_seed(args.seed)
    pyro.clear_param_store()
    data, meta = load_fit_data(args.irt_dir, fold, PARTS)
    dev = torch.device(args.device)
    d = {k: ({p: t.to(dev) for p, t in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in data.items()}
    used = [p for p in PARTS if len(meta["part_keys"][p])]
    m = MixedMIRT(args.form, args.K, {p: len(meta["part_keys"][p]) for p in used}, len(meta["pairs"]),
                  len(meta["items"]), len(meta["cont_benchmarks"]))
    print(f"[{args.form} K={args.K} {fit}] binary cells {meta['n_binary_cells']:,}, continuous cells "
          f"{meta['n_cont_cells']:,}, items {len(meta['items']):,}, pairs {len(meta['pairs']):,}, "
          f"parameters {m.n_params():,}, device {dev}", flush=True)

    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    optimizer = ClippedAdam({"lr": args.lr, "betas": (0.9, 0.999), "clip_norm": 5.0})
    svi = SVI(m.model, m.guide, optimizer, loss=Trace_ELBO())
    ck_params, ck_optim, ck_losses = (os.path.join(out, f) for f in ("ckpt_params.pt", "ckpt_optim.pt", "losses.csv"))
    losses, start = [], 1
    m.guide(d)                                            # creates the params, then the warm start fills them
    if os.path.exists(ck_params) and os.path.exists(ck_optim) and os.path.exists(ck_losses):
        svi.step(d)
        pyro.get_param_store().load(ck_params, map_location=dev)
        optimizer.load(ck_optim, map_location=dev)
        losses = pd.read_csv(ck_losses).loss.tolist()
        start = len(losses) + 1
        print(f"[{fit}] resumed at epoch {start}", flush=True)
    else:
        init_from_1d(m, meta, data, fit, args)            # data: the CPU copy (numpy work happens there)
        print(f"[{fit}] warm start from {BASE_RUN}/{fit} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    t_loop = time.time()
    for ep in range(start, args.epochs + 1):
        losses.append(float(svi.step(d)))
        if ep == 1 or ep % 200 == 0 or ep == args.epochs:
            print(f"[{fit}] epoch {ep:5d}  loss={losses[-1]:,.2f}  {(time.time() - t0) / 60:.1f} min "
                  f"({(time.time() - t_loop) / (ep - start + 1):.3f} s/epoch)", flush=True)
            pyro.get_param_store().save(ck_params)
            optimizer.save(ck_optim)
            pd.DataFrame({"loss": losses}).to_csv(ck_losses, index=False)

    ps = pyro.get_param_store()
    g = lambda k: ps[k].detach().cpu().numpy()
    K = args.K
    vec = "v" if args.form == "nested" else "a"
    items = pd.DataFrame({"item": meta["items"], "b": g("loc_b"), "b_scale": g("scale_b")})
    if K:
        L, S = g(f"loc_{vec}"), g(f"scale_{vec}")
        for k in range(K):
            items[f"{vec}{k + 1}"], items[f"{vec}{k + 1}_scale"] = L[:, k], S[:, k]
    items.to_parquet(os.path.join(out, "items.parquet"))
    for p in PARTS:
        th = pd.DataFrame({"key": meta["part_keys"][p] if p in used else pd.Index([], dtype=str)})
        if p in used:
            if args.form == "nested":
                loc = g(f"loc_theta_{p}")
                th["theta"] = loc - loc.mean()
                if K:
                    u = g(f"loc_u_{p}")
                    for k in range(K):
                        th[f"u{k + 1}"] = u[:, k] - u[:, k].mean()
            else:
                t = g(f"loc_theta_{p}")
                for k in range(K):
                    th[f"theta{k + 1}"] = t[:, k] - t[:, k].mean()
        th.to_parquet(os.path.join(out, f"theta_{p}.parquet"))
    z = pd.DataFrame({"pair": meta["pairs"]})
    zl = g("loc_zeta")
    if args.form == "nested":
        z["zeta"] = zl
    else:
        for k in range(K):
            z[f"zeta{k + 1}"] = zl[:, k]
    z.to_parquet(os.path.join(out, "zeta.parquet"))

    ll, n_resp = loglik(m, d)
    k_par = m.n_params()
    hyper = {k[:-2]: float(g(k)) for k in ps.keys() if k.startswith("sigma_") and k.endswith("_q") and k != "sigma_cont_q"}
    hyper["sigma_cont"] = dict(zip(meta["cont_benchmarks"], g("sigma_cont_q").tolist())) if "sigma_cont_q" in ps.keys() else {}
    hyper.update(form=args.form, K=K, loglik=ll, n_responses=n_resp, n_params=k_par,
                 aic=2 * k_par - 2 * ll, bic=k_par * math.log(n_resp) - 2 * ll,
                 config=vars(args) | {"fold": fold}, minutes=(time.time() - t0) / 60,
                 seconds_per_epoch=(time.time() - t_loop) / max(args.epochs - start + 1, 1),
                 **{k: v for k, v in meta.items() if k.startswith("n_")})
    with open(os.path.join(out, "hyper.json"), "w") as f:
        json.dump(hyper, f, indent=1, default=str)
    # the checkpoint (ckpt_*.pt) is kept, so the fit can be extended later with a larger --epochs
    open(done, "w").close()
    print(f"[{fit}] saved -> {out} ({hyper['minutes']:.1f} min, LL {ll:,.0f}, AIC {hyper['aic']:,.0f}, "
          f"BIC {hyper['bic']:,.0f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--irt_dir", default="data/irt")
    ap.add_argument("--out_root", default="", help="where mirt_<form>_K<K>/ goes (default: --irt_dir)")
    ap.add_argument("--form", required=True, choices=["nested", "m2pl"])
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--fold", default="all", help="0..4, final, or all")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_scale", type=float, default=0.1, help="RMS of the SVD-initialised loadings")
    ap.add_argument("--init_guide_scale", type=float, default=0.1, help="initial guide scale of the new vectors")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.form == "m2pl" and args.K < 1:
        raise SystemExit("m2pl needs K >= 1")
    pyro.enable_validation(False)
    for fold in (["0", "1", "2", "3", "4", "final"] if args.fold == "all" else [args.fold]):
        fit_one(args, fold)


if __name__ == "__main__":
    main()
