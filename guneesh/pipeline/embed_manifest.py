"""Embed every item of an embedding manifest with the Agent Psychometrics recipe, one .npz per benchmark.

The model is loaded ONCE; each benchmark is then embedded and saved separately, so a crashed or
time-limited run resumes where it stopped (finished benchmarks are skipped).

Faithfulness to agent-psychometrics utils/embeddings.py (copied verbatim into ap_embeddings.py):
  * text: format_qs_instruction(question_statement=item content, instruction=DIFFICULTY_INSTRUCTION)
    (already applied in the manifest by build_embed_manifest.py);
  * `load_backbone` is lines 413-458 of their embed_items() and `embed_batch` is its inner flush()
    (lines 471-524), split in two only so the model is not reloaded for every benchmark;
  * defaults equal their CLI defaults: max_length 8192 (left truncation), batch_size 1, bfloat16,
    device_map auto, last layer, seed 0 with deterministic algorithms;
  * output files use their save_embeddings_cache() format (task_ids, X float32, metadata).
`--parity N` embeds N texts with both this path and their unmodified embed_items() and checks the vectors match.

Usage on the GPU machine:
  python pipeline/embed_manifest.py --manifest data/embed_manifest/binary_cap5000.parquet \
      --backbone Qwen/Qwen3-VL-8B-Instruct --out_dir data/emb/qwen3vl8b --max_hours 11.5
  python pipeline/embed_manifest.py ... --pilot 300      # timing run on 300 texts, writes nothing to out_dir
"""
from __future__ import annotations

import argparse
import inspect
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import ap_embeddings as ap


def load_backbone(*, backbone: str, trust_remote_code: bool, device_map: str, torch_dtype: str,
                  attn_implementation: str) -> Dict[str, Any]:
    """Model and tokenizer setup, as in ap.embed_items() lines 413-458."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ap._load_tokenizer(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    dtype_arg = ap._torch_dtype_arg(torch_dtype)
    fp_params = inspect.signature(ap.AutoModel.from_pretrained).parameters
    model_kwargs: Dict[str, Any] = {"dtype" if "dtype" in fp_params else "torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        cfg = ap.AutoConfig.from_pretrained(backbone, trust_remote_code=trust_remote_code)
        quantization_config = getattr(cfg, "quantization_config", None)
        if isinstance(quantization_config, dict) and str(quantization_config.get("quant_method", "")).lower() == "fp8":
            try:
                from transformers import FineGrainedFP8Config

                model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            except Exception:
                pass
    except Exception:
        pass

    model = ap._try_load_model_class(backbone, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)
    model.eval()
    if device_map in ("", "none", None):
        model.to(device)

    text_model = ap._select_text_submodel(model)
    for module in (model, text_model):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = False
            except Exception:
                pass

    try:
        embed_device = text_model.get_input_embeddings().weight.device
    except Exception:
        embed_device = device
    return {"tokenizer": tokenizer, "model": model, "text_model": text_model, "embed_device": embed_device}


def embed_batch(state: Dict[str, Any], texts: List[str], *, max_length: int, embedding_layer: int) -> np.ndarray:
    """One forward pass over `texts`, as in the flush() of ap.embed_items() (lines 471-524). Returns float32 rows."""
    tokenizer, text_model, embed_device = state["tokenizer"], state["text_model"], state["embed_device"]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(embed_device)
    attention_mask = enc["attention_mask"].to(embed_device)
    want_hidden_states = int(embedding_layer) != -1

    with torch.inference_mode():
        fwd_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": bool(want_hidden_states),
        }
        try:
            sig = inspect.signature(text_model.forward)
            if "use_cache" in sig.parameters:
                fwd_kwargs["use_cache"] = False
        except Exception:
            fwd_kwargs["use_cache"] = False

        outputs = text_model(**fwd_kwargs)
        try:
            hidden = ap._extract_hidden_state(outputs, embedding_layer=int(embedding_layer))
        except RuntimeError:
            if want_hidden_states:
                raise
            fwd_kwargs["output_hidden_states"] = True
            outputs = text_model(**fwd_kwargs)
            hidden = ap._extract_hidden_state(outputs, embedding_layer=int(embedding_layer))

        pooled = ap.last_token_pool(hidden, attention_mask).detach().float().cpu().numpy()
    return pooled.astype(np.float32, copy=False)


def embed_texts(state, texts: List[str], *, batch_size: int, max_length: int, embedding_layer: int,
                n_tokens: List[int]) -> np.ndarray:
    """Embed in manifest order with batches of `batch_size` (default 1, as in the paper's runs)."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        try:
            out.append(embed_batch(state, chunk, max_length=max_length, embedding_layer=embedding_layer))
        except torch.cuda.OutOfMemoryError:
            # Fall back to one text at a time for this chunk; a single text that still does not fit is a hard error.
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise RuntimeError(f"Out of GPU memory on one text of {n_tokens[i]} tokens; lower --max_length")
            out.extend(embed_batch(state, [t], max_length=max_length, embedding_layer=embedding_layer) for t in chunk)
    return np.concatenate(out, axis=0)


def embed_benchmark(state, g: pd.DataFrame, args) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int], int]:
    """Embed each distinct text once and give every item its text's vector."""
    uniq = g.drop_duplicates("text")
    X = embed_texts(state, uniq.text.tolist(), batch_size=args.batch_size, max_length=args.max_length,
                    embedding_layer=args.embedding_layer, n_tokens=uniq.n_tokens.tolist())
    vec = dict(zip(uniq.text, X))
    ids = sorted(g.item_id.astype(str))
    by_id = dict(zip(g.item_id.astype(str), g.text))
    return ids, {i: vec[by_id[i]] for i in ids}, {i: len(by_id[i]) for i in ids}, int(X.shape[1])


def parity_check(args, M: pd.DataFrame, state) -> None:
    """Compare this file's load/flush split with the unmodified ap.embed_items() on the first N texts."""
    sub = M.drop_duplicates("text").head(args.parity)
    ours = embed_texts(state, sub.text.tolist(), batch_size=args.batch_size, max_length=args.max_length,
                       embedding_layer=args.embedding_layer, n_tokens=sub.n_tokens.tolist())
    # embed_items() applies format_qs_instruction itself, so feed it the raw statement recovered from our text
    head, tail = "Task statement:\n", "\n\n" + ap.DIFFICULTY_INSTRUCTION.strip()
    recs = [ap.ItemRecord(item_id=f"{k:06d}", question_statement=t[len(head):-len(tail)], solution="")
            for k, t in enumerate(sub.text)]
    for r, t in zip(recs, sub.text):
        assert ap.format_qs_instruction(question_statement=r.question_statement, instruction=ap.DIFFICULTY_INSTRUCTION) == t
    ids, per_id, _, _ = ap.embed_items(items=recs, backbone=args.backbone, trust_remote_code=args.trust_remote_code,
                                       max_length=args.max_length, batch_size=args.batch_size, device_map=args.device_map,
                                       torch_dtype=args.torch_dtype, attn_implementation=args.attn_implementation,
                                       instruction=ap.DIFFICULTY_INSTRUCTION, embedding_layer=args.embedding_layer,
                                       include_solution=False)
    theirs = np.stack([per_id[f"{k:06d}"] for k in range(len(recs))])
    diff = float(np.abs(ours - theirs).max())
    print(f"parity on {len(recs)} texts: max |ours - theirs| = {diff:.3g} (vector norm ~{np.linalg.norm(theirs, axis=1).mean():.1f})")
    if diff > 1e-3 * float(np.abs(theirs).max()):
        raise SystemExit("PARITY FAILED")
    print("PARITY OK")


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--manifest", required=True)
    ap_.add_argument("--out_dir", default="")
    ap_.add_argument("--backbone", default="Qwen/Qwen3-VL-8B-Instruct")
    ap_.add_argument("--trust_remote_code", action="store_true")
    ap_.add_argument("--max_length", type=int, default=8192)
    ap_.add_argument("--batch_size", type=int, default=1)
    ap_.add_argument("--device_map", default="auto")
    ap_.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap_.add_argument("--attn_implementation", default="auto")
    ap_.add_argument("--embedding_layer", type=int, default=-1)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--max_hours", type=float, default=0.0, help="stop starting new benchmarks after this many hours")
    ap_.add_argument("--benchmarks", default="", help="comma-separated subset (default: all in manifest)")
    ap_.add_argument("--pilot", type=int, default=0, help="time N texts sampled across benchmarks, then exit")
    ap_.add_argument("--parity", type=int, default=0, help="check N texts against unmodified embed_items, then exit")
    args = ap_.parse_args()

    ap.seed_everything(int(args.seed), deterministic=True)
    M = pd.read_parquet(args.manifest)
    if args.benchmarks:
        M = M[M.benchmark.isin(args.benchmarks.split(","))]
    t0 = time.time()
    state = load_backbone(backbone=args.backbone, trust_remote_code=args.trust_remote_code, device_map=args.device_map,
                          torch_dtype=args.torch_dtype, attn_implementation=args.attn_implementation)
    print(f"loaded {args.backbone} in {time.time() - t0:.0f}s; text model {type(state['text_model']).__name__} "
          f"on {state['embed_device']}", flush=True)

    if args.parity:
        parity_check(args, M, state)
        return
    if args.pilot:
        u = M.drop_duplicates("text").sample(min(args.pilot, M.text.nunique()), random_state=0)
        t1 = time.time()
        embed_texts(state, u.text.tolist(), batch_size=args.batch_size, max_length=args.max_length,
                    embedding_layer=args.embedding_layer, n_tokens=u.n_tokens.tolist())
        dt = time.time() - t1
        tok_pilot = u.n_tokens.clip(upper=args.max_length).sum()
        tok_all = M.drop_duplicates(["benchmark", "text"]).n_tokens.clip(upper=args.max_length).sum()
        print(f"pilot: {len(u)} texts, {tok_pilot:,} tokens in {dt:.0f}s = {tok_pilot / dt:,.0f} tokens/s; "
              f"projected full run (token-proportional, ignores the extra cost of long texts): {tok_all / (tok_pilot / dt) / 3600:.1f} h")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "_progress.csv")
    for b, g in M.groupby("benchmark", sort=True):
        path = os.path.join(args.out_dir, f"{b}.npz")
        if os.path.exists(path):
            continue
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"time limit {args.max_hours} h reached; rerun the same command to continue", flush=True)
            break
        t1 = time.time()
        ids, per_id, counts, dim = embed_benchmark(state, g, args)
        tmp = path[:-4] + ".tmp.npz"
        ap.save_embeddings_cache(path=tmp, task_ids=ids, embeddings_by_id=per_id, counts_by_id=counts,
                                 embedding_dim=dim, dataset_sources=f"measurement-db:{b}", split="sampled",
                                 dataset_path=args.manifest, instruction=ap.DIFFICULTY_INSTRUCTION,
                                 backbone=args.backbone, max_length=args.max_length,
                                 embedding_layer=args.embedding_layer, include_solution=False)
        os.replace(tmp, path)
        dt = time.time() - t1
        row = pd.DataFrame([{"benchmark": b, "items": len(ids), "unique_texts": g.text.nunique(),
                             "tokens": int(g.drop_duplicates("text").n_tokens.clip(upper=args.max_length).sum()),
                             "seconds": round(dt, 1), "dim": dim}])
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        print(f"{b}: {len(ids)} items in {dt:.0f}s | total {(time.time() - t0) / 3600:.2f} h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
