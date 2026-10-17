"""Option 1: does a global ridge difficulty head on Qwen3-4B item embeddings help Brier ALC? (leave-benchmark-out)"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from sentence_transformers import SentenceTransformer
from paiec import fit, harness, data
from paiec.model import IRTPredictor

QDIR = "data/kaggle_out_qwen/emb_qwen"
T = fit.load_table("data/train_qwen.npz"); nb = len(T["benches"]); names = [str(b) for b in T["benches"]]
_items = {}
def fake_items(b):
    """Items keyed by id; item_content is a lookup token because item texts are not stored locally."""
    if b not in _items:
        z = np.load(f"{QDIR}/{b}.npz"); ids = z["item_id"]
        for i, v in zip(ids, z["emb"]): _vec[(b, str(i))] = v.astype(np.float32)
        _items[b] = {str(i): {"item_content": f"@@{b}|{i}", "item_features": "", "interactors": "", "benchmark_id": b} for i in ids}
    return _items[b]
_vec = {}; data.load_items = fake_items
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); _c = {}
state = {"mean": None}
def embed(text, item=False):
    if item:
        if text.startswith("@@"):
            b, i = text[2:].split("|", 1); return _vec.get((b, i), state["mean"])
        return state["mean"]
    if text not in _c: _c[text] = enc.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    return _c[text]

rng = np.random.default_rng(0); perm = rng.permutation(nb); K = 5; folds = []; t0 = time.time()
for i in range(K):
    test = perm[i::K]; train = np.setdiff1d(np.arange(nb), test)
    s1 = fit.fit_stage1(T, train, lam=dict(r=1000, name=0.03, prov=0.03), epochs=300)
    s1e = {**s1, "beta": s1["beta"] - s1["mu"][T["item_bench"]]}          # within-benchmark difficulty only
    s2 = fit.fit_stage2(T, s1e, lams=(100, 1000, 10000, 100000))
    xm = T["X"][s1["item_ok"]].mean(0); s2["b0"] = float(xm @ s2["coef"]) * 0 + 0.0  # centred: unseen item -> 0
    pairs = []
    for b in test: pairs += harness.make_pairs(names[b], max_pairs=8, max_eval_rows=300)
    folds.append((s1, s2, xm, pairs)); print(f"fold {i}: lam {s2['lam']} cv {s2['cv'][s2['lam']]:.3f} vs base {s2['cv_baseline']:.3f} | {len(pairs)} pairs | {time.time()-t0:.0f}s", flush=True)

def centered(s2, xm):  # intercept so that the head predicts 0 for the mean embedding; absent items embed to xm
    return {**s2, "b0": 0.0}
def score(use_head, a, c, tau, tau_b, g):
    rows = []
    for s1, s2, xm, pairs in folds:
        state["mean"] = xm
        s2u = centered(s2, xm) if use_head else dict(coef=np.zeros_like(s2["coef"]), b0=0.0)
        m = IRTPredictor(s1, s2u, embed, a=a, c=c, tau=tau, tau_b=tau_b, g=g)
        byb = {}
        for p in pairs: byb.setdefault(p["bench"], []).append(p)
        for grp in byb.values(): rows += harness.run_group(grp, m)
    arr = np.array(rows); return arr.mean(0), float((arr @ harness.WEIGHTS).mean())
A, C = 0.03, 0.011   # ability slope / intercept fitted earlier for MiniLM table; refit below
state["mean"] = folds[0][2]
for lbl, head, g in [("no head", False, 0.0)] + [(f"head g={g}", True, g) for g in (0.5, 1.0, 2.0, 4.0)]:
    by, alc = score(head, A, C, 1.0, 2.0, g)
    print(f"{lbl:12s} ALC {alc:.4f} | by budget {np.round(by, 4).tolist()}", flush=True)
