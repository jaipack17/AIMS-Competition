"""Stream items per binary benchmark, embed a capped random sample, keep only the vectors."""
import os, sys, zlib
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
sys.path.insert(0, ".")
from paiec import data

CAP, MODEL = 2000, "sentence-transformers/all-MiniLM-L6-v2"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
enc = SentenceTransformer(MODEL, device=dev); enc.max_seq_length = 256
print("device", dev, flush=True)
for b in data.binary_benchmark_ids():
    out = f"data/emb/{b}.npz"
    if os.path.exists(out): continue
    try:
        resp_items = data.load_responses(b).item_id.unique()
        path = hf_hub_download("aims-foundations/measurement-db", f"{b}/formatted_tables/items.parquet", repo_type="dataset", local_dir="data/mdb")
        it = pd.read_parquet(path, columns=["item_id", "content", "item_features"])
        it = it[it.item_id.isin(set(resp_items))]
        if len(it) > CAP:
            it = it.sample(CAP, random_state=zlib.crc32(b.encode()))
        text = [(str(c) if pd.notna(c) else "")[:2000] for c in it.content]
        E = enc.encode(text, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.savez_compressed(out, item_id=it.item_id.to_numpy(dtype=str), emb=E.astype(np.float16),
                            empty=np.array([t == "" for t in text]))
        os.remove(path)
        print(b, len(it), flush=True)
    except Exception as e:
        print("FAIL", b, type(e).__name__, str(e)[:120], flush=True)
print("DONE", flush=True)
