"""Final submission model (name+provider+year ability, no text term, constant difficulty): refit Platt a,c and grid tau, tau_b.
Leave-benchmark-out 5 folds on the Qwen-table benchmark set. Writes data/final_cv.json."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from sklearn.linear_model import LogisticRegression
from paiec import fit, harness
from paiec.model import IRTPredictor

T = fit.load_table("data/train_qwen.npz"); nb = len(T["benches"]); names = [str(b) for b in T["benches"]]
rng = np.random.default_rng(0); perm = rng.permutation(nb); K = 5; folds = []; t0 = time.time()
zero = lambda *a, **k: np.zeros(384, np.float32)
for i in range(K):
    test = perm[i::K]; train = np.setdiff1d(np.arange(nb), test)
    s1 = fit.fit_stage1(T, train, lam=dict(r=1000, name=0.03, prov=0.03, text=1e9), epochs=300); s1["w"] = 0 * s1["w"]
    pairs = []
    for b in test: pairs += harness.make_pairs(names[b], max_pairs=8, max_eval_rows=300)
    folds.append((s1, dict(coef=np.zeros(T["X"].shape[1], np.float32), b0=0.0), pairs)); print(f"fold {i} {len(pairs)} pairs {time.time()-t0:.0f}s", flush=True)

rows = []
for s1, s2, pairs in folds:
    m = IRTPredictor(s1, s2, zero)
    for p in pairs: rows.append((p["bench"], m.theta(p["subject"]), np.mean([y for _, y in p["eval"]])))
bs = sorted({r[0] for r in rows}); bi = {b: i for i, b in enumerate(bs)}
z = np.array([r[1] for r in rows]); yb = np.array([r[2] for r in rows]); b_ix = np.array([bi[r[0]] for r in rows])
D = np.zeros((len(rows), len(bs))); D[np.arange(len(rows)), b_ix] = 1
X = np.vstack([np.hstack([z[:, None], D])] * 2); Y = np.r_[np.ones(len(rows)), np.zeros(len(rows))]; W = np.r_[yb, 1 - yb]
a = float(LogisticRegression(C=1e4, fit_intercept=False).fit(X, Y, sample_weight=W).coef_[0, 0]); c = 0.0
for _ in range(25):
    p_ = 1 / (1 + np.exp(-(a * z + c))); c += (yb - p_).sum() / max((p_ * (1 - p_)).sum(), 1e-9)
print(f"Platt a={a:.3f} c={c:.3f} theta sd {z.std():.3f}", flush=True)

def score(a_, tau, tau_b):
    out = []
    for s1, s2, pairs in folds:
        m = IRTPredictor(s1, s2, zero, a=a_, c=c, tau=tau, tau_b=tau_b); byb = {}
        for p in pairs: byb.setdefault(p["bench"], []).append(p)
        for g in byb.values(): out += harness.run_group(g, m)
    arr = np.array(out); return arr.mean(0), float((arr @ harness.WEIGHTS).mean())
res = []
for a_ in (a,):
    for tau, tau_b in ((0.5, 1.0), (0.25, 2.0), (0.25, 1.0), (0.5, 0.5), (0.35, 1.5)):
        by, alc = score(a_, tau, tau_b); res.append(dict(a=a_, tau=tau, tau_b=tau_b, alc=alc, by=by.round(4).tolist()))
        print(f"a={a_:.3f} tau {tau} tau_b {tau_b}: ALC {alc:.4f} {np.round(by,4).tolist()}", flush=True)
json.dump(dict(a=a, c=c, grid=res), open("data/final_cv2.json", "w"))
