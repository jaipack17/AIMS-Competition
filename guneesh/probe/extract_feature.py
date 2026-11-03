"""Extract the probe feature (default: rate prompt, layer 20, last token) for EVERY embedded item (GPU).

The output is a drop-in replacement for data/emb/qwen3vl8b/ (one <benchmark>.npz per benchmark with X and task_ids
in the same order), so pipeline/ridge_train.py and pipeline/stretch_factor.py can train on it with --emb_dir.
The vectors are computed by submission_src/common/probe_feature.py, the same code the submission runs at test time.

  python probe/extract_feature.py                                   # rate, layer 20 -> data/emb/qwen3vl8b_rate_L20
  python probe/extract_feature.py --layer 22 --out_dir data/emb/qwen3vl8b_rate_L22
  python probe/extract_feature.py --benchmarks a,b --max_hours 2    # subset / time limit (rerun to resume)

Items: the task_ids of data/emb/qwen3vl8b/<benchmark>.npz; texts from data/measurement-db/<benchmark>/
formatted_tables/items.parquet (downloaded if missing, as extract_activations.py). Identical texts are run once.
Batching: texts sorted by token count, batches up to --token_budget tokens (padded) and --max_batch texts, right
padding (the last real token does not see the padding; differences vs one-at-a-time are bf16 rounding).
On CUDA out-of-memory a batch is split in two and retried.

Check: if --check_acts (default probe/acts/full_l8) holds the same prompt and layer for some items, the new vectors
are compared with them (relative L2 difference; bf16 rounding gives ~1e-2 or less, a wrong text/layer gives ~1).

Output (out_dir):
  spec.json         prompt, layer, position, max_length, backbone: bundle it with build_submission.py --embed_spec
  <benchmark>.npz   X (n, hidden) float32, task_ids, n_tokens; written atomically, finished files are skipped
  _progress.csv     one row per benchmark (items, unique texts, tokens, seconds, parity)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # guneesh/
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "submission_src" / "common"))
sys.path.insert(0, str(ROOT / "probe"))

import probe_feature as pf  # noqa: E402  (shared with the submission)


def batches(lengths: np.ndarray, token_budget: int, max_batch: int) -> list[np.ndarray]:
    """Indices, longest first, cut so that len(batch) * longest <= token_budget and len(batch) <= max_batch."""
    order = np.argsort(-lengths, kind="stable")
    out, cur = [], []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * longest > token_budget):
            out.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def run_batch(state, texts: list[str], spec: dict) -> np.ndarray:
    """embed_texts with OOM fallback: split the batch in two until it fits (a single text that does not fit raises)."""
    torch = state["torch"]
    try:
        return pf.embed_texts(state["tokenizer"], state["text_model"], state["embed_device"], texts, spec).numpy()
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        h = len(texts) // 2
        return np.concatenate([run_batch(state, texts[:h], spec), run_batch(state, texts[h:], spec)])


def parity(check_dir: Path, bench: str, key: str, layer: int, task_ids: np.ndarray, X: np.ndarray):
    """Median and max relative L2 difference vs the activations saved by extract_activations.py (None if absent)."""
    p = check_dir / f"{bench}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        if key not in z.files or layer not in list(z["layers"]):
            return None
        li = list(z["layers"]).index(layer)
        ref = (z[key][:, li].astype(np.uint32) << 16).view(np.float32)
        ids = np.array([s.split("::", 1)[1] for s in z["item"].astype(str)])
    pos = pd.Series(np.arange(len(task_ids)), index=task_ids)
    ok = np.isin(ids, task_ids)
    if not ok.any():
        return None
    mine = X[pos[ids[ok]].to_numpy()]
    rel = np.linalg.norm(mine - ref[ok], axis=1) / np.maximum(np.linalg.norm(ref[ok], axis=1), 1e-12)
    return float(np.median(rel)), float(rel.max()), int(ok.sum())


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--prompt", default="rate", choices=["rate", "ap"])
    pa.add_argument("--layer", type=int, default=20)
    pa.add_argument("--max_length", type=int, default=8192)
    pa.add_argument("--out_dir", default="", help="default data/emb/qwen3vl8b_<prompt>_L<layer>")
    pa.add_argument("--data_dir", default=str(ROOT / "data"))
    pa.add_argument("--ref_emb", default="", help="task_ids source (default <data_dir>/emb/qwen3vl8b)")
    pa.add_argument("--check_acts", default=str(ROOT / "probe" / "acts" / "full_l8"))
    pa.add_argument("--benchmarks", default="", help="comma-separated subset")
    pa.add_argument("--token_budget", type=int, default=16384, help="max padded tokens per batch")
    pa.add_argument("--max_batch", type=int, default=32)
    pa.add_argument("--backbone", default="Qwen/Qwen3-VL-8B-Instruct")
    pa.add_argument("--device_map", default="auto")
    pa.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    pa.add_argument("--attn_implementation", default="auto")
    pa.add_argument("--max_hours", type=float, default=0.0, help="stop starting new benchmarks after this many hours")
    pa.add_argument("--no_fetch", action="store_true")
    pa.add_argument("--allow_cpu", action="store_true", help="smoke tests only")
    args = pa.parse_args()

    spec = pf.check_spec({"prompt": args.prompt, "layer": args.layer, "max_length": args.max_length})
    data_dir = Path(args.data_dir)
    ref_dir = Path(args.ref_emb) if args.ref_emb else data_dir / "emb" / "qwen3vl8b"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "emb" / f"qwen3vl8b_{args.prompt}_L{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = sorted(p.stem for p in ref_dir.glob("*.npz"))
    if args.benchmarks:
        want = args.benchmarks.split(",")
        missing = sorted(set(want) - set(benches))
        if missing:
            raise SystemExit(f"not in {ref_dir}: {missing}")
        benches = [b for b in benches if b in want]
    if not benches:
        raise SystemExit(f"no embedding files in {ref_dir}")

    spec_out = {**spec, "backbone": args.backbone, "torch_dtype": args.torch_dtype}
    spec_path = out_dir / "spec.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        diff = [k for k in spec_out if old.get(k) != spec_out[k]]
        if diff:
            raise SystemExit(f"{spec_path} differs in {diff}; use another --out_dir")
    spec_path.write_text(json.dumps(spec_out, indent=1))
    for stale in out_dir.glob("*.tmp.npz"):
        stale.unlink()
    todo = [b for b in benches if not (out_dir / f"{b}.npz").exists()]
    print(f"{len(benches) - len(todo)} of {len(benches)} benchmarks already done; spec {spec_out}", flush=True)
    if not todo:
        print("done", flush=True)
        return

    import torch
    from embed_manifest import load_backbone
    from extract_activations import fetch_items_table
    import ap_embeddings as apx
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no CUDA GPU visible; --allow_cpu for smoke tests")
    t0 = time.time()
    apx.seed_everything(0, deterministic=True)
    state = load_backbone(backbone=args.backbone, trust_remote_code=False, device_map=args.device_map,
                          torch_dtype=args.torch_dtype, attn_implementation=args.attn_implementation)
    state["torch"] = torch
    tok = state["tokenizer"]
    n_dec = len(pf.decoder_layers(state["text_model"]))
    print(f"loaded {args.backbone} in {time.time() - t0:.0f}s on {state['embed_device']}; {n_dec} decoder layers; "
          f"stopping after layer {spec['layer']}", flush=True)

    # texts and token counts of every benchmark to do (for the ETA)
    work = {}
    for bench in todo:
        with np.load(ref_dir / f"{bench}.npz", allow_pickle=True) as z:
            task_ids = z["task_ids"].astype(str)
        it = pd.read_parquet(fetch_items_table(data_dir, bench, not args.no_fetch), columns=["item_id", "content"])
        it["item_id"] = it.item_id.astype(str)
        content = it.drop_duplicates("item_id").set_index("item_id").content
        lost = [t for t in task_ids if t not in content.index]
        if lost:
            raise SystemExit(f"{bench}: {len(lost)} embedded items have no text, e.g. {lost[:3]}")
        texts = [pf.build_text(c, spec["prompt"]) for c in content.reindex(task_ids).tolist()]
        uniq, inv = np.unique(np.array(texts, dtype=object), return_inverse=True)
        lengths = np.array([len(x) for x in tok(list(uniq), truncation=True, max_length=spec["max_length"])["input_ids"]])
        work[bench] = (task_ids, list(uniq), inv, lengths)
    total_tok = sum(int(w[3].sum()) for w in work.values())
    print(f"{sum(len(w[0]) for w in work.values()):,} items ({sum(len(w[1]) for w in work.values()):,} unique texts), "
          f"{total_tok / 1e6:.1f}M tokens to run [{time.time() - t0:.0f}s]", flush=True)

    progress = out_dir / "_progress.csv"
    done_tok, t_start = 0, time.time()
    key = f"{spec['prompt']}__last"
    for bench in todo:
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"time limit {args.max_hours} h reached; rerun the same command to continue", flush=True)
            break
        task_ids, uniq, inv, lengths = work[bench]
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        U = None
        for idx in batches(lengths, args.token_budget, args.max_batch):
            v = run_batch(state, [uniq[i] for i in idx], spec)
            if U is None:
                U = np.empty((len(uniq), v.shape[1]), np.float32)
            U[idx] = v
        X = U[inv]
        if not np.isfinite(X).all():
            raise SystemExit(f"{bench}: non-finite values in the features")
        path, tmp = out_dir / f"{bench}.npz", out_dir / f"{bench}.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, X=X, task_ids=task_ids, n_tokens=lengths[inv].astype(np.int32))
        os.replace(tmp, path)

        dt = time.time() - t1
        done_tok += int(lengths.sum())
        eta = (time.time() - t_start) / done_tok * (total_tok - done_tok) / 3600
        par = parity(Path(args.check_acts), bench, key, spec["layer"], task_ids, X) if args.check_acts else None
        row = {"benchmark": bench, "items": len(task_ids), "unique_texts": len(uniq), "tokens": int(lengths.sum()),
               "seconds": round(dt, 1), "parity_rel_median": par[0] if par else np.nan,
               "parity_rel_max": par[1] if par else np.nan, "parity_n": par[2] if par else 0,
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0}
        pd.DataFrame([row]).to_csv(progress, mode="a", header=not progress.exists(), index=False)
        ptxt = f"parity rel diff median {par[0]:.1e} max {par[1]:.1e} (n {par[2]})" if par else "no parity reference"
        print(f"{bench}: {len(task_ids):,} items, {lengths.sum() / 1e6:.2f}M tokens in {dt:.0f}s | {ptxt} | "
              f"peak {row['peak_gpu_gb']} GB | {(time.time() - t0) / 3600:.2f} h, ETA {eta:.2f} h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
