"""How predictable is within-benchmark item difficulty from text embeddings, across benchmarks?"""
import sys; sys.path.insert(0, ".")
import os, numpy as np
from paiec import fit
T = fit.load_table(os.environ.get("TRAIN", "data/train.npz")); nb = len(T["benches"])
s1 = fit.fit_stage1(T, np.arange(nb), epochs=250)
beta, g, X = s1["beta"], T["item_bench"], T["X"]
prec = fit.item_precision(T, s1["item_ok"]); ok = prec > 0
nsub = np.bincount(T["sub_bench"], minlength=nb)
mu_b = np.array([beta[(g == b) & ok].mean() for b in range(nb)]); eps = beta - mu_b[g]
print("benchmarks", nb, "| var(beta)", beta[ok].var().round(3), "| between", mu_b.var().round(3), "| within", eps[ok].var().round(3))
rng = np.random.default_rng(0); perm = rng.permutation(nb); fold = {b: i % 5 for i, b in enumerate(perm)}; f = np.array([fold[b] for b in g])
for lam in (10, 100, 1000, 10000):
    per = []
    for i in range(5):
        tr, te = (f != i) & ok, (f == i) & ok
        xm = X[tr].mean(0); A = (X[tr] - xm).T @ (X[tr] - xm) + lam * np.eye(X.shape[1]); c = np.linalg.solve(A, (X[tr] - xm).T @ eps[tr])
        pred = (X[te] - xm) @ c
        for b in np.unique(g[te]):
            m = g[te] == b
            if m.sum() < 50 or nsub[b] < 5: continue          # skip benchmarks whose difficulty targets are noisy
            pe, ee = pred[m] - pred[m].mean(), eps[te][m] - eps[te][m].mean()
            per.append((ee @ pe / (np.linalg.norm(ee) * np.linalg.norm(pe) + 1e-9), 1 - ((ee - pe) ** 2).sum() / (ee ** 2).sum()))
    per = np.array(per)
    print(f"lam {lam:6d}: within-benchmark corr median {np.median(per[:,0]):.3f} mean {per[:,0].mean():.3f} | share corr>0.1: {(per[:,0]>0.1).mean():.0%} | R2 median {np.median(per[:,1]):.3f} | n={len(per)} benchmarks (>=5 subjects)")
