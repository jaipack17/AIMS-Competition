"""Embed the sampled measurement-db items with Qwen3 on a Kaggle GPU (resumable, one file per benchmark).

Kaggle setup: GPU = T4 x2, Internet on, Add-ons > Secrets: HF_TOKEN; attach the dataset holding sample_manifest.parquet.
    !pip install -q -U "transformers>=4.51" accelerate huggingface_hub pandas pyarrow
Output: /kaggle/working/emb_qwen/<benchmark>.npz  (item_id, emb float16) -- same item ids as data/emb on the laptop.

MODEL choices:
  Qwen/Qwen3-Embedding-8B  dedicated embedding model (last-token pooling), 4096-d.   [default]
  Qwen/Qwen3-Embedding-4B  same family, 2560-d, fits one T4 in fp16 (~8 GB)  -> run two shards, one per GPU.
  Qwen/Qwen3-8B            chat LLM; mean-pooled last hidden state (no embedding training), 4096-d.

Two T4s with a 4B model (run as two background processes in the notebook):
  CUDA_VISIBLE_DEVICES=0 QWEN_MODEL=Qwen/Qwen3-Embedding-4B SHARD=0/2 python kaggle_embed_qwen3.py &
  CUDA_VISIBLE_DEVICES=1 QWEN_MODEL=Qwen/Qwen3-Embedding-4B SHARD=1/2 python kaggle_embed_qwen3.py &
"""
import glob, os, sys, time
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-Embedding-8B")
POOL = "last" if "Embedding" in MODEL else "mean"
MAX_LEN = int(os.environ.get("MAX_LEN", 256))
PER_BENCH = int(os.environ.get("PER_BENCH", 2000))       # lower (e.g. 500) to fit the Kaggle weekly GPU quota
MAX_HOURS = float(os.environ.get("MAX_HOURS", 9))        # stop cleanly before the Kaggle session limit so output is saved
TOK_BUDGET = int(os.environ.get("TOK_BUDGET", 6000))      # tokens per batch, tune to GPU memory
SHARD = os.environ.get("SHARD", "0/1"); SH_I, SH_N = (int(x) for x in SHARD.split("/"))
SMOKE = os.environ.get("SMOKE", "0") == "1"   # 3 smallest benchmarks x 50 items: checks token, GPU, speed, NaNs
OUT = "/kaggle/working/emb_qwen_smoke" if SMOKE else "/kaggle/working/emb_qwen"; os.makedirs(OUT, exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN missing: add it under Add-ons > Secrets and attach it to this notebook")
man = pd.read_parquet(glob.glob("/kaggle/input/**/sample_manifest.parquet", recursive=True)[0])
if SMOKE:
    small_b = man.groupby("benchmark").size().sort_values().index[:3]
    man, PER_BENCH = man[man.benchmark.isin(small_b)], 50

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
small = any(t in MODEL for t in ("0.6B", "4B"))
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0} if small else "auto").eval()   # 8B shards across both T4s
dev = next(model.parameters()).device
print(MODEL, "pool", POOL, "| GPUs", torch.cuda.device_count(), flush=True)

@torch.no_grad()
def embed(texts):
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * min(MAX_LEN, 1 + len(texts[order[j]]) // 3) <= TOK_BUDGET: j += 1
        idx = order[i:max(j, i + 1)]
        enc = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state
        if POOL == "last":
            v = h[:, -1]                                           # left padding: last position is the last real token
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype); v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        out[idx] = v.cpu().numpy().astype(np.float16); i = max(j, i + 1)
    return out

t0 = time.time()
for bi, (b, g) in enumerate(man.groupby("benchmark")):
    if bi % SH_N != SH_I: continue
    if time.time() - t0 > MAX_HOURS * 3600:
        print("TIME BUDGET REACHED, stopping; rerun for the rest", flush=True); break
    path = f"{OUT}/{b}.npz"
    if os.path.exists(path): continue
    ids = g.item_id.to_numpy()[:PER_BENCH]
    try:
        f = hf_hub_download("aims-foundations/measurement-db", f"{b}/formatted_tables/items.parquet", repo_type="dataset", local_dir="/tmp/mdb")
        it = pd.read_parquet(f, columns=["item_id", "content"]).set_index("item_id")
        texts = [str(it.content.get(i, "") or "")[:2000] for i in ids]; os.remove(f)
        E = embed(texts)
        if not np.isfinite(E.astype(np.float32)).all():
            raise SystemExit(f"NaN/inf embeddings in {b}: fp16 overflow; switch to 8-bit or bf16")
        np.savez_compressed(path, item_id=ids.astype(str), emb=E)
        print(f"{b} {len(ids)} items | {time.time()-t0:.0f}s", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print("FAIL", b, type(e).__name__, str(e)[:100], flush=True)
        if SMOKE: raise
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
