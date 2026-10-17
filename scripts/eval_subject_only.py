"""Step 1: subject-ability + constant difficulty + delta + calibration, leave-benchmark-out (5 folds)."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from paiec import fit, harness
from paiec.model import IRTPredictor

T = fit.load_table(); nb = len(T["benches"])
bench_names = [str(b) for b in T["benches"]]
rng = np.random.default_rng(0); perm = rng.permutation(nb); K = 5
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
_cache = {}
def embed(text, item=False):
    if text not in _cache: _cache[text] = enc.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    return _cache[text]

def variants(s1, mu_const):
    s2 = dict(coef=np.zeros(T["X"].shape[1], np.float32), b0=mu_const)
    nosubj = {**s1, "name_row": 0 * s1["name_row"], "prov_row": 0 * s1["prov_row"], "w": 0 * s1["w"], "yr": 0 * s1["yr"]}
    nodate = {**s1, "yr": 0 * s1["yr"]}
    return {"full": (s1, s2), "no_subject": (nosubj, s2), "no_date": (nodate, s2)}

folds = []
t0 = time.time()
for i in range(K):
    test = perm[i::K]; train = np.setdiff1d(np.arange(nb), test)
    s1 = fit.fit_stage1(T, train, lam=dict(r=1000, name=0.03, prov=0.03), epochs=300)
    mu_const = 0.0   # constant difficulty is absorbed into the calibrated intercept c
    pairs = []
    for b in test:
        pairs += harness.make_pairs(bench_names[b], max_pairs=8, max_eval_rows=300)
    folds.append((variants(s1, mu_const), pairs))
    print(f"fold {i}: train {len(train)} test {len(test)} benchmarks, {len(pairs)} pairs, nll {s1['nll']:.4f}, {time.time()-t0:.0f}s", flush=True)

def platt(name):
    """Slope a from WITHIN-benchmark variation (benchmark fixed effects); c0 = global intercept given a."""
    rows = []
    for v, pairs in folds:
        m = IRTPredictor(*v[name], embed)
        for p in pairs:
            ys = np.array([y for _, y in p["eval"]], float)
            rows.append((p["bench"], m.theta(p["subject"]), ys.mean()))
    bs = sorted({r[0] for r in rows}); bi = {b: i for i, b in enumerate(bs)}
    z = np.array([r[1] for r in rows]); yb = np.array([r[2] for r in rows]); b_ix = np.array([bi[r[0]] for r in rows])
    if z.std() < 1e-9: a = 1.0   # no subject information: slope unidentifiable
    else:
        D = np.zeros((len(rows), len(bs))); D[np.arange(len(rows)), b_ix] = 1
        X = np.vstack([np.hstack([z[:, None], D]), np.hstack([z[:, None], D])]); Y = np.r_[np.ones(len(rows)), np.zeros(len(rows))]
        W = np.r_[yb, 1 - yb]
        a = float(LogisticRegression(C=1e4, fit_intercept=False).fit(X, Y, sample_weight=W).coef_[0, 0])
    c = 0.0
    for _ in range(25):   # intercept only, slope fixed: each pair counts equally
        p_ = 1 / (1 + np.exp(-(a * z + c))); c += (yb - p_).sum() / max((p_ * (1 - p_)).sum(), 1e-9)
    return a, float(c)

def score(name, a, c, tau, tau_b=2.0, lam_note=''):
    rows = []
    for v, pairs in folds:
        m = IRTPredictor(*v[name], embed, a=a, c=c, tau=tau, tau_b=tau_b)
        byb = {}
        for p in pairs: byb.setdefault(p["bench"], []).append(p)
        for grp in byb.values(): rows += harness.run_group(grp, m)
    arr = np.array(rows)
    return arr.mean(0), float((arr @ harness.WEIGHTS).mean()), len(rows)

results = {}
for name in ("full", "no_subject", "no_date"):
    a, c = platt(name)
    print(f"\n[{name}] Platt a={a:.3f} c={c:.3f}", flush=True)
    for tau, tau_b in (((0.5, 2.0), (1.0, 2.0), (1.0, 4.0), (2.0, 4.0)) if name != "no_date" else (best_tau,)):
        by, alc, n = score(name, a, c, tau, tau_b)
        results[(name, (tau, tau_b))] = alc
        print(f"  tau,tau_b {tau},{tau_b}: ALC {alc:.4f} | by budget {np.round(by, 4).tolist()} | pairs {n}", flush=True)
    if name == "full": best_tau = min((t for (n_, t) in results if n_ == "full"), key=lambda t: results[("full", t)])
    if name == "full": print(f"  best tau for full: {best_tau}")
# unit-scale reference: the same pairs scored with the smoothed-mean baseline
from baselines import simple
rows = [harness.run_pair(p, simple.smoothed_mean) for _, pairs in folds for p in pairs]  # own-pair labels only (it ignores other subjects)
arr = np.array(rows); print(f"\n[smoothed mean, same pairs] ALC {(arr @ harness.WEIGHTS).mean():.4f} | by budget {np.round(arr.mean(0), 4).tolist()}")
