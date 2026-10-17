"""Two-stage fit.

Stage 1: logistic IRT on binomial cells, P = sigmoid(theta_subject - beta_item), with
  theta_subject = name_row + provider_row + year_effect + subject_text . w_text + r_s   (r_s: free residual, shrunk)
  beta_item     = mu_bench + eps_item                                                    (eps shrunk to mu_bench)
Stage 2: weighted ridge from item embedding to beta (leave-benchmark-out CV for lambda).
"""
from __future__ import annotations
import numpy as np, torch

LAM = dict(r=1.0, eps=1.0, name=0.3, prov=0.3, text=1.0, year=0.1)


def load_table(path="data/train.npz"):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fit_stage1(T, train_benches, lam=None, epochs=300, lr=0.05, seed=0, verbose=False):
    lam = {**LAM, **(lam or {})}
    torch.manual_seed(seed)
    tb = np.zeros(len(T["benches"]), bool); tb[list(train_benches)] = True
    sub_ok = tb[T["sub_bench"]]; item_ok = tb[T["item_bench"]]
    cell_ok = sub_ok[T["sub"]] & item_ok[T["item"]]
    sub, item, n, k = (torch.tensor(T[x][cell_ok]) for x in ("sub", "item", "n", "k"))
    nS, nI, nB = len(T["sub_bench"]), len(T["item_bench"]), len(T["benches"])
    names = sorted({x for x, ok in zip(T["sub_name"], sub_ok) if ok and x})
    provs = sorted({x for x, ok in zip(T["sub_provider"], sub_ok) if ok and x})
    nid = {x: i + 1 for i, x in enumerate(names)}; pid = {x: i + 1 for i, x in enumerate(provs)}   # 0 = unknown/unseen
    sname = torch.tensor([nid.get(x, 0) if ok else 0 for x, ok in zip(T["sub_name"], sub_ok)])
    sprov = torch.tensor([pid.get(x, 0) if ok else 0 for x, ok in zip(T["sub_provider"], sub_ok)])
    year = np.nan_to_num(T["sub_year"] - 2024.0, nan=0.0); unk = np.isnan(T["sub_year"]).astype(np.float32)
    year_t, unk_t = torch.tensor(year, dtype=torch.float32), torch.tensor(unk)
    S_t = torch.tensor(T["S"]); ibench = torch.tensor(T["item_bench"])
    p = dict(r=torch.zeros(nS), mu=torch.zeros(nB), eps=torch.zeros(nI), name=torch.zeros(len(names) + 1),
             prov=torch.zeros(len(provs) + 1), w=torch.zeros(S_t.shape[1]), yr=torch.zeros(3))
    for v in p.values(): v.requires_grad_()
    opt = torch.optim.Adam(p.values(), lr=lr)
    N = float(n.sum())
    used_s = torch.tensor(sub_ok); used_i = torch.tensor(item_ok)
    def theta():
        # index 0 rows are pinned to zero so unseen names/providers fall back cleanly
        nm = torch.cat([torch.zeros(1), p["name"][1:]]); pv = torch.cat([torch.zeros(1), p["prov"][1:]])
        return nm[sname] + pv[sprov] + p["yr"][0] * year_t + p["yr"][1] * year_t ** 2 / 10 + p["yr"][2] * unk_t + S_t @ p["w"] + p["r"]
    for ep in range(epochs):
        opt.zero_grad()
        z = theta()[sub] - (p["mu"][ibench] + p["eps"])[item]
        nll = -(k * torch.nn.functional.logsigmoid(z) + (n - k) * torch.nn.functional.logsigmoid(-z)).sum() / N
        pen = (lam["r"] * (p["r"][used_s] ** 2).sum() + lam["eps"] * (p["eps"][used_i] ** 2).sum()
               + lam["name"] * (p["name"] ** 2).sum() + lam["prov"] * (p["prov"] ** 2).sum()
               + lam["text"] * (p["w"] ** 2).sum() + lam["year"] * (p["yr"] ** 2).sum()) / N
        (nll + pen).backward(); opt.step()
        if verbose and ep % 50 == 0: print(f"  ep {ep} nll {nll.item():.4f}", flush=True)
    d = {k_: v.detach().numpy() for k_, v in p.items()}
    return dict(names=names, provs=provs, name_row=d["name"][1:], prov_row=d["prov"][1:], w=d["w"], yr=d["yr"],
                beta=(d["mu"][T["item_bench"]] + d["eps"]), mu=d["mu"], item_ok=item_ok, nll=float(nll.detach()))


def item_precision(T, item_ok):
    """Number of observed responses per item, used to weight the difficulty regression."""
    m = item_ok[T["item"]]
    return np.bincount(T["item"][m], weights=T["n"][m], minlength=len(T["item_bench"]))


def fit_stage2(T, s1, lams=(1, 10, 100, 1000), folds=5, seed=0):
    ok = s1["item_ok"] & (item_precision(T, s1["item_ok"]) > 0)
    X, y, g = T["X"][ok], s1["beta"][ok], T["item_bench"][ok]
    prec = item_precision(T, s1["item_ok"])[ok]
    # cap each benchmark's total weight so large benchmarks do not dominate
    w = np.minimum(prec, 20.0); tot = np.bincount(g, weights=w); w = w / np.sqrt(tot[g]) 
    ub = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(ub); fold = {b: i % folds for i, b in enumerate(ub)}
    f = np.array([fold[b] for b in g])
    def ridge(Xa, ya, wa, lam):
        xm = (wa[:, None] * Xa).sum(0) / wa.sum(); ym = (wa * ya).sum() / wa.sum()
        Xc = Xa - xm; A = Xc.T @ (wa[:, None] * Xc) + lam * np.eye(Xa.shape[1])
        coef = np.linalg.solve(A, Xc.T @ (wa * (ya - ym)))
        return coef, ym - xm @ coef
    cv = {}
    for lam in lams:
        err = 0.0
        for i in range(folds):
            c, b0 = ridge(X[f != i], y[f != i], w[f != i], lam)
            err += (w[f == i] * (X[f == i] @ c + b0 - y[f == i]) ** 2).sum()
        cv[lam] = err / w.sum()
    best = min(cv, key=cv.get); coef, b0 = ridge(X, y, w, best)
    base = (w * (y - (w * y).sum() / w.sum()) ** 2).sum() / w.sum()
    return dict(coef=coef, b0=float(b0), lam=best, cv=cv, cv_baseline=float(base))
