"""Is ability uni- or multidimensional? Held-out tests within and across benchmarks."""
import glob, numpy as np, pandas as pd, torch
torch.manual_seed(0); rng = np.random.default_rng(0)
B = pd.concat([pd.read_parquet(f) for f in glob.glob("data/mdb/*/formatted_tables/benchmarks.parquet")]).set_index("benchmark_id")
S = pd.read_parquet("data/subjects_all.parquet")[["subject_id", "bench", "normalized_name"]]
print("response_type counts:", B.response_type.value_counts().to_dict())

def load(b):
    r = pd.read_parquet(f"data/mdb/{b}/formatted_tables/responses.parquet", columns=["subject_id", "item_id", "response"])
    r = r.dropna(subset=["response"])
    return r[r.response.isin([0.0, 1.0])]

# ---------- A. within-benchmark: logistic factor model, rank 0 (Rasch) vs 1 vs 2 vs 3 ----------
def fit(si, ii, y, ns, ni, k, wd, epochs=300):
    a = torch.zeros(ns, requires_grad=True); d = torch.zeros(ni, requires_grad=True)
    U = (0.1 * torch.randn(ns, k)).requires_grad_(); V = (0.1 * torch.randn(ni, k)).requires_grad_()
    ps = [a, d] + ([U, V] if k else [])
    opt = torch.optim.Adam(ps, lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        z = a[si] - d[ii] + ((U[si] * V[ii]).sum(1) if k else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + wd * (a.pow(2).mean() + d.pow(2).mean() + (U.pow(2).mean() + V.pow(2).mean() if k else 0))
        loss.backward(); opt.step()
    return lambda s, i: (a[s] - d[i] + ((U[s] * V[i]).sum(1) if k else 0)).detach()

rows = []
for b in B.index:
    r = load(b)
    ns, ni = r.subject_id.nunique(), r.item_id.nunique()
    if ns < 8 or ni < 60 or len(r) > 400_000 or r.response.mean() in (0.0, 1.0): continue
    si = pd.factorize(r.subject_id)[0]; ii = pd.factorize(r.item_id)[0]; y = r.response.values
    m = rng.random(len(r)) < 0.8
    t = lambda x: torch.tensor(x)
    out = {"bench": b, "ns": ns, "ni": ni}
    for k in (0, 1, 2, 3):
        best = None
        for wd in (0.01, 0.1, 1.0):
            f = fit(t(si[m]), t(ii[m]), t(y[m]).float(), ns, ni, k, wd)
            p = torch.sigmoid(f(t(si[~m]), t(ii[~m]))).numpy().clip(1e-4, 1 - 1e-4)
            bs = float(np.mean((p - y[~m]) ** 2))
            best = bs if best is None else min(best, bs)
        out[f"brier_k{k}"] = best
    rows.append(out)
    if len(rows) >= 60: break
A = pd.DataFrame(rows)
print(f"\n[A] within-benchmark held-out Brier, {len(A)} benchmarks (rank 0 = Rasch/1D ability)")
print(A[[f"brier_k{k}" for k in range(4)]].mean().round(4).to_string())
for k in (1, 2, 3):
    print(f"  rank {k} beats rank 0 in {(A[f'brier_k{k}'] < A.brier_k0 - 1e-4).mean():.0%} of benchmarks; rank {k} vs rank {k-1} wins {(A[f'brier_k{k}'] < A[f'brier_k{k-1}'] - 1e-4).mean():.0%}")
A.to_csv("data/dim_within.csv", index=False)

# ---------- B. across benchmarks: named subject x benchmark mean accuracy ----------
acc = []
for b in B.index[B.response_type == "binary"]:
    r = load(b)
    if r.empty: continue
    g = r.groupby("subject_id").response.mean().rename("acc").reset_index()
    g["bench"] = b; acc.append(g)
acc = pd.concat(acc).merge(S, on=["subject_id", "bench"])
acc = acc[acc.normalized_name != ""].groupby(["normalized_name", "bench"]).acc.mean().reset_index()
cnt_s = acc.groupby("normalized_name").bench.nunique(); cnt_b = acc.groupby("bench").normalized_name.nunique()
acc = acc[acc.normalized_name.map(cnt_s) >= 5]; acc = acc[acc.bench.map(cnt_b) >= 5]
logit = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
acc["z"] = logit(acc.acc.values)
si, bi = pd.factorize(acc.normalized_name)[0], pd.factorize(acc.bench)[0]
ns, nb = si.max() + 1, bi.max() + 1
print(f"\n[B] subject x benchmark matrix: {ns} named subjects x {nb} benchmarks, {len(acc)} cells ({len(acc)/ns/nb:.1%} filled)")
z = acc.z.values; res = []
for rep in range(5):
    m = rng.random(len(acc)) < 0.85
    row = {}
    for k in (0, 1, 2, 3, 4):
        best = None
        for wd in (0.003, 0.03, 0.3):
            torch.manual_seed(rep)
            a = torch.zeros(ns, requires_grad=True); c = torch.zeros(nb, requires_grad=True)
            U = (0.1 * torch.randn(ns, k)).requires_grad_(); V = (0.1 * torch.randn(nb, k)).requires_grad_()
            opt = torch.optim.Adam([a, c, U, V], lr=0.03); Si, Bi, Z = map(torch.tensor, (si, bi, z))
            Z = Z.float(); mt = torch.tensor(m)
            for _ in range(800):
                opt.zero_grad()
                pred = a[Si] + c[Bi] + (U[Si] * V[Bi]).sum(1)
                loss = ((pred - Z)[mt] ** 2).mean() + wd * (U.pow(2).sum() + V.pow(2).sum() + a.pow(2).sum() + c.pow(2).sum()) / mt.sum()
                loss.backward(); opt.step()
            pred = (a[Si] + c[Bi] + (U[Si] * V[Bi]).sum(1)).detach().numpy()
            pa = 1 / (1 + np.exp(-pred)); mse = float(np.mean((pa[~m] - acc.acc.values[~m]) ** 2))
            best = mse if best is None else min(best, mse)
        row[k] = best
    res.append(row)
R = pd.DataFrame(res).mean()
print("held-out MSE of accuracy (15% of cells), additive subject+benchmark effects + rank-k interaction:")
print(R.round(5).to_string()); print("relative to rank 0:", (R / R[0]).round(3).to_dict())
# singular value spectrum of the double-centered logit matrix on the densest block
M = acc.pivot(index="normalized_name", columns="bench", values="z")
dense = M.loc[M.notna().sum(1) >= M.shape[1] * 0.25, M.notna().sum() >= 8]
print(f"\nDensest block: {dense.shape}; filling missing with row/col additive fit, then SVD of residual")
Mz = dense.values; mask = ~np.isnan(Mz); fill = np.where(mask, Mz, np.nanmean(Mz))
for _ in range(50):
    rm, cm = fill.mean(1, keepdims=True), fill.mean(0, keepdims=True); R0 = fill - rm - cm + fill.mean()
    u, s, vt = np.linalg.svd(R0, full_matrices=False); L = (u[:, :2] * s[:2]) @ vt[:2]
    fill = np.where(mask, Mz, rm + cm - fill.mean() + L)
Xc = np.where(mask, Mz, np.nan); 
full = fill - fill.mean(0, keepdims=True) ; s_all = np.linalg.svd(fill, compute_uv=False)
print("variance share of top singular values of the filled logit matrix (incl. mean effects):", (s_all[:5] ** 2 / (s_all ** 2).sum()).round(3))
print("share after removing subject & benchmark mean effects:", (np.linalg.svd(fill - fill.mean(1, keepdims=True) - fill.mean(0, keepdims=True) + fill.mean(), compute_uv=False)[:5] ** 2 / (np.linalg.svd(fill - fill.mean(1, keepdims=True) - fill.mean(0, keepdims=True) + fill.mean(), compute_uv=False) ** 2).sum()).round(3))
