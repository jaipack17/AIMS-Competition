"""Our final model vs baselines on IDENTICAL pairs and folds (leave-benchmark-out), cluster-bootstrap over benchmarks."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from paiec import fit, harness
from paiec.model import IRTPredictor
from baselines import simple

cv = json.load(open("data/final_cv.json")); A, C = cv["a"], cv["c"]; TAU, TAU_B = 0.5, 1.0
T = fit.load_table("data/train_qwen.npz"); nb = len(T["benches"]); names = [str(b) for b in T["benches"]]
perm = np.random.default_rng(0).permutation(nb); zero = lambda *a, **k: np.zeros(384, np.float32)
methods = {"ours": [], "smoothed_mean": [], "pooled_mean": [], "constant_half": []}; bench_of = []
for i in range(5):
    test = perm[i::5]; train = np.setdiff1d(np.arange(nb), test)
    s1 = fit.fit_stage1(T, train, lam=dict(r=1000, name=0.03, prov=0.03, text=1e9), epochs=300); s1["w"] = 0 * s1["w"]
    ours = IRTPredictor(s1, dict(coef=np.zeros(T["X"].shape[1], np.float32), b0=0.0), zero, a=A, c=C, tau=TAU, tau_b=TAU_B)
    byb = {}
    for b in test:
        for p in harness.make_pairs(names[b], 8, 300): byb.setdefault(p["bench"], []).append(p)
    for bn, g in byb.items():
        for k, fn in (("ours", ours), ("smoothed_mean", simple.smoothed_mean), ("pooled_mean", simple.pooled_mean), ("constant_half", simple.constant_half)):
            methods[k] += harness.run_group(g, fn)
        bench_of += [bn] * len(g)
    print("fold", i, flush=True)
M = {k: np.array(v) @ harness.WEIGHTS for k, v in methods.items()}; bench_of = np.array(bench_of); ub = np.unique(bench_of)
idx = {b: np.where(bench_of == b)[0] for b in ub}; rng = np.random.default_rng(0); B = 2000
boots = []
for _ in range(B):
    s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]); boots.append(s)
print(f"\n{len(bench_of)} pairs, {len(ub)} benchmarks")
for k, v in M.items():
    bs = np.array([v[s].mean() for s in boots]); print(f"{k:14s} ALC {v.mean():.4f}  95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
for k in ("smoothed_mean", "pooled_mean", "constant_half"):
    d = M[k] - M["ours"]; bs = np.array([d[s].mean() for s in boots])
    pb = np.array([ (M[k][idx[b]].mean() - M["ours"][idx[b]].mean()) for b in ub])
    print(f"{k} - ours: {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}] | ours better on {np.mean(pb>0):.0%} of benchmarks")
by = {k: np.array(v).mean(0).round(4).tolist() for k, v in methods.items()}
for k, v in by.items(): print(f"{k:14s} by budget {v}")
