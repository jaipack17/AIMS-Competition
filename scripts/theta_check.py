"""Does the transferable part of theta (name/provider/year/text) rank subjects within held-out benchmarks?"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from paiec import fit, harness
from paiec.model import IRTPredictor

T = fit.load_table(); nb = len(T["benches"]); names = [str(b) for b in T["benches"]]
rng = np.random.default_rng(0); perm = rng.permutation(nb); K = 5
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"); _c = {}
def embed(t, item=False):
    if t not in _c: _c[t] = enc.encode([t], normalize_embeddings=True)[0].astype(np.float32)
    return _c[t]
pairs_by_fold = [[p for b in perm[i::K] for p in harness.make_pairs(names[b], 8, 300)] for i in range(K)]
dummy = dict(coef=np.zeros(T["X"].shape[1], np.float32), b0=0.0)
for lam_r, lam_name in [(1, 0.3), (10, 0.3), (100, 0.3), (100, 0.03), (1000, 0.03)]:
    t0 = time.time(); per_bench = {}
    named_share = []
    for i in range(K):
        train = np.setdiff1d(np.arange(nb), perm[i::K])
        s1 = fit.fit_stage1(T, train, lam=dict(r=lam_r, name=lam_name, prov=lam_name), epochs=300)
        m = IRTPredictor(s1, dummy, embed)
        for p in pairs_by_fold[i]:
            acc = np.mean([y for _, y in p["eval"]])
            per_bench.setdefault(p["bench"], []).append((m.theta(p["subject"]), acc, bool(p["subject"]["normalized_name"] and p["subject"]["normalized_name"] in m.name_ix)))
    rs, rs_named, n_b = [], [], 0
    for b, rows in per_bench.items():
        th, ac, seen = (np.array(x) for x in zip(*rows))
        if len(rows) >= 4 and th.std() > 1e-9 and ac.std() > 1e-9:
            rs.append(spearmanr(th, ac)[0]); n_b += 1
        if seen.sum() >= 4 and th[seen].std() > 1e-9 and ac[seen].std() > 1e-9:
            rs_named.append(spearmanr(th[seen], ac[seen])[0])
    print(f"lam_r {lam_r:5} lam_name {lam_name:5}: within-benchmark Spearman(theta, accuracy) mean {np.nanmean(rs):.3f} median {np.nanmedian(rs):.3f} over {n_b} benchmarks | seen-name subjects only: mean {np.nanmean(rs_named):.3f} ({len(rs_named)} bm) | {time.time()-t0:.0f}s", flush=True)
