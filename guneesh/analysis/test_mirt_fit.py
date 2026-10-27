"""Checks of pipeline/mirt_fit.py on small simulated data with a known 2-dimensional structure (CPU, minutes).

Builds a tiny data/irt-like folder in a temporary directory: 60 named subjects with harness / provider / effort,
4 binary benchmarks + 1 continuous benchmark, every subject on every item, responses drawn from the NESTED model
with K = 2 (theta + zeta + u.v - b). Then:
  1. fits the 1D version B model (pipeline/irt_fit.py), which mirt_fit uses as its warm start;
  2. fits nested K = 0, 1, 2, 3 and m2pl K = 1, 2, 3 on the full data;
  3. checks: every fit writes its files with the right columns; nested K = 0 matches the 1D fit's logits closely
     (same model, same data); training log-likelihood rises with K; the true K = 2 nested logits are recovered
     (correlation of fitted vs true logits) better by nested K = 2 than by K = 0; AIC / BIC are finite.
Run from the project root:  python analysis/test_mirt_fit.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def simulate(irt_dir: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S, K = 60, 2
    names = [f"model{i}" for i in range(S)]
    harness = rng.choice(["h-a", "h-b", "h-c", ""], S)
    provider = rng.choice(["p1", "p2", ""], S)
    effort = rng.choice(["low", "high", ""], S)
    th_name, u_name = rng.normal(0, 0.8, S), rng.normal(0, 1.0, (S, K))
    th_h = {"h-a": 0.4, "h-b": -0.3, "h-c": 0.1, "": 0.0}
    benches = ["bin0", "bin1", "bin2", "bin3"]
    subj_rows, cells, truth = [], [], []
    for bench_i, bench in enumerate(benches + ["cont0"]):
        n_items = 100
        b = rng.normal(0, 1.5, n_items)
        v = rng.normal(0, 1.0, (n_items, K)) * (1.0 + bench_i % 2)          # benchmarks load differently
        for s in range(S):
            pair = f"{bench}::s{s}"
            subj_rows.append({"pair": pair, "benchmark": bench, "subject_id": f"s{s}", "name": names[s],
                              "harness_n": harness[s], "provider_n": provider[s], "effort_n": effort[s]})
            zeta = rng.normal(0, 0.5)
            mu = th_name[s] + th_h[harness[s]] + zeta + v @ u_name[s] - b
            for j in range(n_items):
                item = f"{bench}::i{j}"
                if bench.startswith("bin"):
                    n = 2
                    k = rng.binomial(n, 1 / (1 + np.exp(-mu[j])))
                    cells.append(("bin", pair, item, n, k, bench))
                    truth.append((pair, item, mu[j]))
                else:
                    y = 1 / (1 + np.exp(-(mu[j] + rng.normal(0, 0.5))))
                    z = np.log(np.clip(y, 0.01, 0.99) / (1 - np.clip(y, 0.01, 0.99)))
                    cells.append(("cont", pair, item, 1, (y, z), bench))
    pd.DataFrame(subj_rows).to_parquet(os.path.join(irt_dir, "subjects.parquet"))
    B = pd.DataFrame([c[1:] for c in cells if c[0] == "bin"], columns=["pair", "item", "n", "k", "benchmark"])
    B["k"] = B.k.astype(float)
    B.to_parquet(os.path.join(irt_dir, "cells_binary.parquet"))
    C = pd.DataFrame([(c[1], c[2], c[3], c[4][0], c[4][1], c[4][1] ** 2, c[5]) for c in cells if c[0] == "cont"],
                     columns=["pair", "item", "n", "sy", "sz", "sz2", "benchmark"])
    C.to_parquet(os.path.join(irt_dir, "cells_cont.parquet"))
    pd.DataFrame({"benchmark": benches, "fold": [0, 1, 2, 3]}).to_csv(os.path.join(irt_dir, "folds.csv"), index=False)
    return pd.DataFrame(truth, columns=["pair", "item", "mu"])


def fitted_logits(fit_dir: str, form: str, K: int, cells: pd.DataFrame) -> np.ndarray:
    items = pd.read_parquet(os.path.join(fit_dir, "items.parquet")).set_index("item")
    z = pd.read_parquet(os.path.join(fit_dir, "zeta.parquet")).set_index("pair")
    subj = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(fit_dir)), "subjects.parquet")).set_index("pair")
    col = {"name": "name", "harness": "harness_n", "provider": "provider_n", "effort": "effort_n"}
    th = {p: pd.read_parquet(os.path.join(fit_dir, f"theta_{p}.parquet")).set_index("key") for p in col}
    out = []
    for pair, item in zip(cells.pair, cells.item):
        s = subj.loc[pair]
        rows = [th[p].loc[s[c]] for p, c in col.items() if s[c] and s[c] in th[p].index]
        it = items.loc[item]
        if form == "nested":
            mu = sum(r.theta for r in rows) + z.loc[pair].zeta - it.b
            if K:
                u = sum(np.array([r[f"u{k + 1}"] for k in range(K)]) for r in rows)
                mu += float(np.dot(u, [it[f"v{k + 1}"] for k in range(K)]))
        else:
            t = sum(np.array([r[f"theta{k + 1}"] for k in range(K)]) for r in rows) + \
                np.array([z.loc[pair][f"zeta{k + 1}"] for k in range(K)])
            mu = float(np.dot(t, [it[f"a{k + 1}"] for k in range(K)])) - it.b
        out.append(mu)
    return np.array(out)


def run(args: list[str]) -> None:
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(args)}")


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="mirt_test_")
    truth = simulate(tmp)
    sample = truth.sample(3000, random_state=0)
    run(["pipeline/irt_fit.py", "--irt_dir", tmp, "--fold", "final", "--epochs", "1500", "--threads", "4"])
    fits = [("nested", K) for K in (0, 1, 2, 3)] + [("m2pl", K) for K in (1, 2, 3)]
    res = {}
    for form, K in fits:
        run(["pipeline/mirt_fit.py", "--irt_dir", tmp, "--form", form, "--K", str(K), "--fold", "final",
             "--epochs", "1500", "--device", "cpu"])
        fdir = os.path.join(tmp, f"mirt_{form}_K{K}", "final")
        h = json.load(open(os.path.join(fdir, "hyper.json")))
        mu = fitted_logits(fdir, form, K, sample)
        res[(form, K)] = (h["loglik"], h["aic"], h["bic"], np.corrcoef(mu, sample.mu)[0, 1])
        assert np.isfinite([h["loglik"], h["aic"], h["bic"]]).all()
        print(f"{form:6s} K={K}: LL {h['loglik']:,.0f}  AIC {h['aic']:,.0f}  BIC {h['bic']:,.0f}  "
              f"corr(fitted logit, true logit) {res[(form, K)][3]:.3f}", flush=True)
    # 1. nested K=0 is version B: compare with the 1D fit's own logits
    one_d = fitted_logits(os.path.join(tmp, "versionB_4parts_logitnormal", "final"), "nested", 0, sample)
    k0 = fitted_logits(os.path.join(tmp, "mirt_nested_K0", "final"), "nested", 0, sample)
    print(f"nested K=0 vs 1D version B: corr {np.corrcoef(one_d, k0)[0, 1]:.4f}, max |diff| {np.abs(one_d - k0).max():.3f}")
    assert np.corrcoef(one_d, k0)[0, 1] > 0.99
    # 2. more dimensions fit the training data better, and K=2 recovers the true structure better than K=0
    assert res[("nested", 2)][0] > res[("nested", 0)][0] and res[("nested", 1)][0] > res[("nested", 0)][0]
    assert res[("nested", 2)][3] > res[("nested", 0)][3] + 0.02
    assert res[("m2pl", 2)][3] > res[("m2pl", 1)][3]
    print(f"ALL CHECKS PASSED  (temporary data in {tmp})")


if __name__ == "__main__":
    main()
