"""Build the stage-1 training table from embedded items + responses.

Outputs data/train.npz:
  cell arrays  : sub (subject-in-benchmark index), item (global item index), n (trials), k (successes)
  item arrays  : item_bench (benchmark index), X (item embeddings, float32)
  subject arrays: sub_bench, sub_name, sub_provider, sub_year (nan if unknown), S (subject text embeddings)
"""
import glob, os, sys
EMB_DIR = os.environ.get("EMB_DIR", "data/emb"); OUT = os.environ.get("TRAIN_OUT", "data/train.npz")
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
sys.path.insert(0, ".")
from paiec import data

def main():
    benches = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{EMB_DIR}/*.npz"))
    X, item_bench, cells, subs = [], [], [], []
    item_off = 0
    for bi, b in enumerate(benches):
        z = np.load(f"{EMB_DIR}/{b}.npz")
        ids, E = z["item_id"], z["emb"].astype(np.float32)
        empty = z["empty"].astype(bool) if "empty" in z.files else np.zeros(len(ids), bool)
        if len(ids) == 0: continue
        keep = ~empty                                   # items with no text carry no signal for the head
        ids, E = ids[keep], E[keep]
        if len(ids) < 20: continue
        r = data.load_responses(b)
        r = r[r.item_id.isin(set(ids))]
        if r.empty: continue
        g = r.groupby(["subject_id", "item_id"]).response.agg(n="size", k="sum").reset_index()
        item_ix = pd.Series(np.arange(len(ids)) + item_off, index=ids)
        subj = data.load_subjects(b)
        sid_ix = {s: len(subs) + j for j, s in enumerate(sorted(g.subject_id.unique()))}
        for s, j in sid_ix.items():
            d = subj.get(s, {})
            yr = pd.to_numeric(str(d.get("release_date", ""))[:4], errors="coerce")
            subs.append((bi, d.get("normalized_name", ""), d.get("provider", ""), float(yr) if pd.notna(yr) else np.nan))
        cells.append(np.column_stack([g.subject_id.map(sid_ix).to_numpy(), g.item_id.map(item_ix).to_numpy(),
                                      g.n.to_numpy(), g.k.to_numpy()]).astype(np.int64))
        X.append(E); item_bench.append(np.full(len(ids), bi)); item_off += len(ids)
    cells = np.concatenate(cells)
    text = [f"{n} | {p}" if (n or p) else "" for _, n, p, _ in subs]
    uniq = sorted(set(text)); enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                                       device="mps" if torch.backends.mps.is_available() else "cpu")
    U = enc.encode(uniq, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    pos = {t: i for i, t in enumerate(uniq)}
    S = np.stack([U[pos[t]] if t else np.zeros(U.shape[1], np.float32) for t in text]).astype(np.float32)
    np.savez_compressed(OUT, sub=cells[:, 0], item=cells[:, 1], n=cells[:, 2], k=cells[:, 3],
                        X=np.concatenate(X), item_bench=np.concatenate(item_bench), benches=np.array(benches),
                        sub_bench=np.array([s[0] for s in subs]), sub_name=np.array([s[1] for s in subs]),
                        sub_provider=np.array([s[2] for s in subs]), sub_year=np.array([s[3] for s in subs]), S=S)
    print(f"{len(benches)} benchmarks | {len(subs)} subject-in-benchmark | {item_off} items | {len(cells)} cells | named {np.mean([bool(s[1]) for s in subs]):.1%}")

if __name__ == "__main__":
    main()
