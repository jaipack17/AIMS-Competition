"""Synthetic check of pipeline/lowrank_dir.py: 60 benchmarks whose difficulty directions lie in a 3-dim subspace.

Expect: basis row 0 = unit mean direction, rows orthonormal; with n labelled items the shared-basis ridge ("ours")
reaches its plateau by k = 4 (mean + 3 axes), beats from-scratch kernel ridge at small n, and the true subspace is
recovered (principal angles near 0).    Run from the project root:  python analysis/test_lowrank_dir.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import lowrank_dir as L  # noqa: E402
from avgdir_train import directions  # noqa: E402

rng = np.random.default_rng(0)
d, true_k, n_b = 256, 3, 60
U = np.linalg.qr(rng.normal(size=(d, true_k)))[0].T            # true shared axes (3 x d)
base = rng.normal(size=true_k)


def bench(n=400):
    a = base + 0.8 * rng.normal(size=true_k)                    # benchmark mix around a common mean
    Z = rng.normal(size=(n, d))
    y = Z @ (U.T @ a) + 0.5 * rng.normal(size=n)
    return Z, y


L.DRAWS = 5
dirs = {}
for m in range(n_b):
    Z, y = bench()
    dirs[m] = directions(Z, y)[L.ALPHAS.index(L.ALPHA)]
Q, pca = L.basis(dirs)
assert np.allclose(Q @ Q.T, np.eye(len(Q)), atol=1e-8), "basis not orthonormal"
mean_dir = np.mean(list(dirs.values()), 0)
assert np.allclose(Q[0], mean_dir / np.linalg.norm(mean_dir)), "row 0 is not the unit mean direction"
cos = np.linalg.svd(Q[:4] @ U.T, compute_uv=False)
print("principal cosines between first 4 basis rows and the true 3-dim subspace:", np.round(cos, 3))
assert cos.min() > 0.9, "true subspace not recovered"

rows = []
for m in range(10):
    Z, y = bench(800)
    rows += L.evaluate(Z, y, Q, seed=m)
df = pd.DataFrame(rows)
best = df.groupby(["method", "k", "n", "lam"], dropna=False).r.mean().groupby(["method", "k", "n"]).max().unstack("n")
print(best.round(3).to_string())
ours30 = best.loc["ours"]["30"]
assert ours30.loc[4] > ours30.loc[1] + 0.05, "k = 4 should beat the plain mean direction"
assert abs(ours30.loc[4] - ours30.max()) < 0.03, "plateau should be reached by k = 4"
assert ours30.loc[4] > best.loc[("scratch", 0), "30"] + 0.1, "shared basis should beat scratch at n = 30"
print("ALL CHECKS PASSED")
