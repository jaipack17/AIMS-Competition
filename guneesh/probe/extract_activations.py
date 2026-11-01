"""Extract Qwen3-VL-8B activations for the difficulty-probe experiments (see probe/README.md).

For a sample of items of each binary training benchmark, the text model is run once per prompt and this is saved:

  hidden states   all 37 layer outputs (index 0 = token embeddings, 36 = last layer after the final RMSNorm,
                  i.e. exactly the vector of the existing embeddings in data/emb/qwen3vl8b/), at the positions
                    last       last token of the whole text                       (every prompt)
                    item_last  last token of the item text, before any instruction (prompt "ap" only)
                    item_mean  mean over the item's tokens                         (prompt "ap" only)
                  The model is causal and every prompt starts with the same "Task statement:\\n<item>", so the two
                  item positions do not see the instruction: they are the "no prompt" state and are saved once.
  logit lens      at the "last" position of every layer, lm_head(final_norm(h_l)):
                    probabilities of the digit tokens 0-9, their total mass, entropy of the full next-token
                    distribution; plus the last layer's top-5 next tokens (what the model would actually write).
  parity          max |ap layer-36 last-token state - stored embedding| per item (should be ~0).

Prompts (all start with "Task statement:\\n<item>\\n\\n"):
  ap        the Agent Psychometrics difficulty prompt, character for character the text of the existing embeddings
  rate      "How difficult is the above task for an AI model? Answer with a single digit from 0 (very easy) to
             9 (very hard).\\nDifficulty: "   -> next token is a digit: the model's own 0-9 rating
  rate_rev  the same with the scale reversed (0 = very hard); optional robustness check, not run by default
  solve     "Solve the above task. Reply with only the final answer.\\nAnswer:"  -> state just before answering

Sample: for each benchmark, the items that have an IRT difficulty b (final fit) and a stored embedding (if the
embedding file exists) and a text, in a fixed random order (seeded by benchmark name and --seed); the first N
are taken. The pilot items are therefore a subset of the full-run items.
  --mode pilot   6 benchmarks per fold (30), 100 items each      (default)
  --mode full    all 115 benchmarks, 300 items each

Output (out_dir = probe/acts/<mode>/ by default):
  run_config.json          prompts, model, layer convention, digit token ids, versions
  sample.parquet           benchmark, item, item_id, b, fold of every sampled item
  <benchmark>.npz          one file per benchmark, written atomically (resumable: finished files are skipped)
  _progress.csv            one row per finished benchmark (time, tokens, digit mass, parity, GPU memory)
  Hidden states are stored as the raw bfloat16 bits in uint16 arrays (lossless); decode with probe/acts_io.py.

Usage (from the project root guneesh/; see probe/README.md for the SSH setup):
  python probe/extract_activations.py --dry_run            # tokenizer only: prints prompts and token positions
  python probe/extract_activations.py --mode pilot
  python probe/extract_activations.py --mode full --max_hours 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # guneesh/
sys.path.insert(0, str(ROOT / "pipeline"))

import ap_embeddings as ap  # noqa: E402  (verbatim Agent Psychometrics code: prompt format, seeding)

HEAD = "Task statement:\n"
PROMPTS = {
    "ap": None,  # built with ap.format_qs_instruction(..., ap.DIFFICULTY_INSTRUCTION), as the existing embeddings
    "rate": ("How difficult is the above task for an AI model? Answer with a single digit from "
             "0 (very easy) to 9 (very hard).\nDifficulty: "),
    "rate_rev": ("How difficult is the above task for an AI model? Answer with a single digit from "
                 "0 (very hard) to 9 (very easy).\nDifficulty: "),
    "solve": "Solve the above task. Reply with only the final answer.\nAnswer:",
}
ITEM_POSITIONS = ("item_last", "item_mean")  # saved for the first prompt only
MODES = {"pilot": (6, 100), "full": (None, 300)}  # (benchmarks per fold, items per benchmark)
MEASUREMENT_DB = "aims-foundations/measurement-db"


# ----------------------------------------------------------------------------------------------------- texts
def build_text(prompt: str, content: str) -> tuple[str, tuple[int, int]]:
    """Full input text and the character span of the item inside it."""
    qs = ap._sanitize_text(str(content or "")).strip()
    if prompt == "ap":
        text = ap.format_qs_instruction(question_statement=content, instruction=ap.DIFFICULTY_INSTRUCTION)
    else:
        text = f"{HEAD}{qs}\n\n{PROMPTS[prompt]}"
    span = (len(HEAD), len(HEAD) + len(qs))
    assert text[span[0]:span[1]] == qs, "item span does not match the text"
    return text, span


def tokenize(tok, text: str, span: tuple[int, int], max_length: int):
    """Token ids (left truncation, as the embedding pipeline) and the indices of the item's tokens."""
    enc = tok(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    ids = enc["input_ids"]
    off = np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
    # a token belongs to the item if it starts inside the item span (a token like "?\n\n" that starts on the
    # item's last character still only sees item text)
    is_item = (off[:, 0] >= span[0]) & (off[:, 0] < span[1]) & (off[:, 1] > off[:, 0])
    item_idx = np.flatnonzero(is_item)
    truncated = len(tok(text, truncation=False)["input_ids"]) > len(ids)
    return ids, item_idx, truncated


# ----------------------------------------------------------------------------------------------------- sample
def fetch_items_table(data_dir: Path, bench: str, allow_fetch: bool) -> Path:
    p = data_dir / "measurement-db" / bench / "formatted_tables" / "items.parquet"
    if not p.exists():
        if not allow_fetch:
            raise SystemExit(f"missing {p}; rerun without --no_fetch or run analysis/fetch_items_binary.py")
        from huggingface_hub import hf_hub_download
        hf_hub_download(MEASUREMENT_DB, f"{bench}/formatted_tables/items.parquet", repo_type="dataset",
                        local_dir=str(data_dir / "measurement-db"))
    return p


def build_sample(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    folds = pd.read_csv(data_dir / "irt" / "folds.csv")
    per_fold, n_items = MODES[args.mode]
    if args.n_items:
        n_items = args.n_items
    if args.benchmarks:
        benches = args.benchmarks.split(",")
    elif per_fold is None:
        benches = sorted(folds.benchmark)
    else:
        rng = np.random.default_rng(args.seed)
        benches = []
        for f in sorted(folds.fold.unique()):
            fb = sorted(folds.loc[folds.fold == f, "benchmark"])
            benches += sorted(rng.choice(fb, min(per_fold, len(fb)), replace=False).tolist())

    irt = pd.read_parquet(data_dir / "irt" / args.irt_run / "final" / "items.parquet", columns=["item", "b"])
    irt["benchmark"] = irt["item"].str.split("::", n=1).str[0]
    irt["item_id"] = irt["item"].str.split("::", n=1).str[1]
    fold_of = dict(zip(folds.benchmark, folds.fold))

    parts = []
    for bench in benches:
        pool = irt[irt.benchmark == bench]
        emb = data_dir / "emb" / "qwen3vl8b" / f"{bench}.npz"
        if emb.exists():
            ids = set(np.load(emb, allow_pickle=True)["task_ids"].astype(str))
            pool = pool[pool.item_id.isin(ids)]
        content = pd.read_parquet(fetch_items_table(data_dir, bench, not args.no_fetch), columns=["item_id", "content"])
        content["item_id"] = content.item_id.astype(str)
        content = content.drop_duplicates("item_id")
        content = content[content.content.fillna("").astype(str).str.strip() != ""]
        pool = pool.merge(content, on="item_id").sort_values("item_id").reset_index(drop=True)
        if pool.empty:
            print(f"WARNING {bench}: no usable items, skipped", flush=True)
            continue
        rng_b = np.random.default_rng(zlib.crc32(f"{bench}:{args.seed}".encode()))
        take = pool.iloc[rng_b.permutation(len(pool))[:n_items]].copy()
        take["fold"] = fold_of.get(bench, -1)
        parts.append(take[["benchmark", "item", "item_id", "b", "fold", "content"]])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------------------------------- model
def load(args):
    import torch
    from embed_manifest import load_backbone  # the existing pipeline's loader (same settings as the embeddings)

    state = load_backbone(backbone=args.backbone, trust_remote_code=False, device_map=args.device_map,
                          torch_dtype=args.torch_dtype, attn_implementation=args.attn_implementation)
    model, text_model = state["model"], state["text_model"]
    norm = getattr(text_model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = model.get_output_embeddings()
    if norm is None or lm_head is None:
        raise SystemExit(f"cannot find the final norm / lm_head on {type(model).__name__}")
    state.update(norm=norm, lm_head=lm_head, torch=torch)
    return state


def digit_token_ids(tok) -> list[int]:
    out = []
    for d in "0123456789":
        ids = tok(d, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"digit {d!r} is not a single token: {ids}")
        out.append(ids[0])
    return out


def run_one(state, ids: list[int], item_idx: np.ndarray, want_item: bool, digit_ids):
    """One forward pass. Returns hidden states at the positions (bf16 bits) and the logit-lens readout."""
    torch = state["torch"]
    text_model, dev = state["text_model"], state["embed_device"]
    x = torch.tensor([ids], device=dev)
    with torch.inference_mode():
        out = text_model(input_ids=x, attention_mask=torch.ones_like(x), return_dict=True,
                         output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of n_layers + 1 tensors (1, T, H)
        if state.get("last_is_normed") is None:  # checked once: is the last entry the final-normed state?
            state["last_is_normed"] = bool(torch.equal(hs[-1], out.last_hidden_state))
            print(f"hidden_states[-1] == last_hidden_state (already normed): {state['last_is_normed']}", flush=True)
        res = {"last": torch.stack([h[0, -1] for h in hs])}  # (L, H) bf16
        if want_item:
            j = torch.as_tensor(item_idx, device=hs[0].device)
            res["item_last"] = torch.stack([h[0, int(item_idx[-1])] for h in hs])
            res["item_mean"] = torch.stack([h[0].index_select(0, j.to(h.device)).float().mean(0) for h in hs]).to(hs[0].dtype)
        del out, hs

        # logit lens at the last position: final norm on every layer except the last one if it is already normed
        h = res["last"].to(state["lm_head"].weight.device)
        k = h.shape[0] - 1 if state["last_is_normed"] else h.shape[0]
        normed = state["norm"](h[:k].to(state["norm"].weight.device)).to(h.device)
        z = torch.cat([normed, h[k:]], 0)
        logp = torch.log_softmax(state["lm_head"](z).float(), dim=-1)  # (L, V)
        p = logp.exp()
        dp = p[:, digit_ids]
        top = torch.topk(p[-1], 5)
        lens = {"digit_probs": dp.cpu().numpy().astype(np.float32),
                "digit_mass": dp.sum(-1).cpu().numpy().astype(np.float32),
                "entropy": (-(p * logp).sum(-1)).cpu().numpy().astype(np.float32),
                "top5_ids": top.indices.cpu().numpy().astype(np.int64),
                "top5_probs": top.values.cpu().numpy().astype(np.float32)}
    acts = {k: v.contiguous().view(torch.int16).cpu().numpy().view(np.uint16) for k, v in res.items()}
    return acts, lens


# ----------------------------------------------------------------------------------------------------- main
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def dry_run(args, S: pd.DataFrame, prompts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    tok.truncation_side = "left"
    print("digit token ids:", digit_token_ids(tok))
    for _, r in S.groupby("benchmark").head(1).head(2).iterrows():
        print(f"\n===== {r['item']}  b={r['b']:.2f}")
        for p in prompts:
            text, span = build_text(p, r["content"])
            ids, item_idx, trunc = tokenize(tok, text, span, args.max_length)
            print(f"--- prompt {p}: {len(ids)} tokens{' (TRUNCATED)' if trunc else ''}, item tokens "
                  f"{item_idx[0]}..{item_idx[-1]}\n    text ends: {text[-110:]!r}\n"
                  f"    item_last token: {tok.decode([ids[item_idx[-1]]])!r}   last token: {tok.decode([ids[-1]])!r}")


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--mode", choices=list(MODES), default="pilot")
    pa.add_argument("--prompts", default="ap,rate,solve", help=f"comma-separated, first one also saves the item "
                    f"positions; available: {','.join(PROMPTS)}")
    pa.add_argument("--n_items", type=int, default=0, help="items per benchmark (default: 100 pilot, 300 full)")
    pa.add_argument("--benchmarks", default="", help="comma-separated subset (overrides the mode's choice)")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--data_dir", default=str(ROOT / "data"))
    pa.add_argument("--irt_run", default="versionB_4parts_logitnormal")
    pa.add_argument("--out_dir", default="", help="default: probe/acts/<mode>")
    pa.add_argument("--backbone", default="Qwen/Qwen3-VL-8B-Instruct")
    pa.add_argument("--max_length", type=int, default=8192)
    pa.add_argument("--device_map", default="auto")
    pa.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    pa.add_argument("--attn_implementation", default="auto")
    pa.add_argument("--max_hours", type=float, default=0.0, help="stop starting new benchmarks after this many hours")
    pa.add_argument("--no_fetch", action="store_true", help="do not download missing measurement-db item tables")
    pa.add_argument("--allow_cpu", action="store_true", help="allow running without CUDA (smoke tests only)")
    pa.add_argument("--dry_run", action="store_true", help="tokenizer only: show prompts and positions, then exit")
    args = pa.parse_args()

    prompts = args.prompts.split(",")
    unknown = [p for p in prompts if p not in PROMPTS]
    if unknown:
        raise SystemExit(f"unknown prompts {unknown}")
    out_dir = Path(args.out_dir or ROOT / "probe" / "acts" / args.mode)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # sample (reused on restart so a resumed run uses exactly the same items)
    sample_path = out_dir / "sample.parquet"
    if sample_path.exists() and not args.dry_run:
        S = pd.read_parquet(sample_path)
        content = build_sample(args).set_index("item").content
        S["content"] = S["item"].map(content)
        if S.content.isna().any():
            raise SystemExit(f"{sample_path} does not match the current arguments; use another --out_dir")
    else:
        S = build_sample(args)
        if not args.dry_run:
            S.drop(columns="content").to_parquet(sample_path, index=False)
    n_keys = len(prompts) + len(ITEM_POSITIONS)
    print(f"sample: {len(S):,} items, {S.benchmark.nunique()} benchmarks; prompts {prompts}", flush=True)
    if args.dry_run:
        dry_run(args, S, prompts)
        return

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no CUDA GPU visible (check nvidia-smi / CUDA_VISIBLE_DEVICES); --allow_cpu for smoke tests")
    ap.seed_everything(int(args.seed), deterministic=True)
    state = load(args)
    tok = state["tokenizer"]
    digit_ids = digit_token_ids(tok)
    n_layers = int(getattr(state["text_model"].config, "num_hidden_layers", 0)) + 1
    hidden = int(getattr(state["text_model"].config, "hidden_size", 0))
    gb = len(S) * n_keys * n_layers * hidden * 2 / 1e9
    print(f"loaded {args.backbone} in {time.time() - t0:.0f}s on {state['embed_device']}; {n_layers} layer outputs x "
          f"{hidden}; expected output size {gb:.1f} GB", flush=True)

    cfg = {"mode": args.mode, "prompts": {p: (ap.DIFFICULTY_INSTRUCTION if p == "ap" else PROMPTS[p]) for p in prompts},
           "item_positions_from": prompts[0], "positions": ["last", *ITEM_POSITIONS],
           "backbone": args.backbone, "torch_dtype": args.torch_dtype, "max_length": args.max_length,
           "n_layer_outputs": n_layers, "hidden_size": hidden, "digit_token_ids": digit_ids,
           "layer_convention": "index 0 = token embeddings, i = output of decoder layer i, last = after final norm",
           "storage": "hidden states as uint16 = raw bfloat16 bits; decode with probe/acts_io.py",
           "seed": args.seed, "irt_run": args.irt_run, "git_commit": git_commit(),
           "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    cfg_path = out_dir / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for k in ("prompts", "backbone", "max_length", "torch_dtype"):
            if old.get(k) != cfg[k]:
                raise SystemExit(f"{cfg_path} has a different {k}; use another --out_dir to avoid mixing runs")
        cfg["last_is_normed"] = old.get("last_is_normed")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    for stale in out_dir.glob("*.tmp.npz"):
        stale.unlink()

    progress = out_dir / "_progress.csv"
    benches = list(dict.fromkeys(S.benchmark))
    todo = [b for b in benches if not (out_dir / f"{b}.npz").exists()]
    n_todo = int(S.benchmark.isin(todo).sum())
    print(f"{len(benches) - len(todo)} benchmarks already done, {len(todo)} to do ({n_todo:,} items)", flush=True)
    done_items, t_start = 0, time.time()

    for bench in todo:
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"time limit {args.max_hours} h reached; rerun the same command to continue", flush=True)
            break
        g = S[S.benchmark == bench].reset_index(drop=True)
        emb_path = Path(args.data_dir) / "emb" / "qwen3vl8b" / f"{bench}.npz"
        ref = None
        if emb_path.exists():
            z = np.load(emb_path, allow_pickle=True)
            ref = dict(zip(z["task_ids"].astype(str), z["X"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t1, n_tok = time.time(), 0
        rows, skipped = [], []
        for _, r in g.iterrows():
            rec = {"item": r["item"], "b": r["b"]}
            try:
                for k, p in enumerate(prompts):
                    text, span = build_text(p, r["content"])
                    ids, item_idx, trunc = tokenize(tok, text, span, args.max_length)
                    if k == 0 and len(item_idx) == 0:
                        raise ValueError("no item tokens left after truncation")
                    acts, lens = run_one(state, ids, item_idx, k == 0, digit_ids)
                    n_tok += len(ids)
                    rec[f"{p}__n_tokens"], rec[f"{p}__truncated"] = len(ids), trunc
                    rec[f"{p}__last"] = acts["last"]
                    if k == 0:
                        for pos in ITEM_POSITIONS:
                            rec[f"{p}__{pos}"] = acts[pos]
                        if ref is not None and r["item_id"] in ref and p == "ap":
                            mine = (acts["last"][-1].astype(np.uint32) << 16).view(np.float32)
                            rec["parity_maxabs"] = float(np.abs(mine - ref[r["item_id"]]).max())
                    for name, v in lens.items():
                        rec[f"{p}__{name}"] = v
            except (torch.cuda.OutOfMemoryError, ValueError) as e:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                skipped.append(r["item"])
                print(f"  skip {r['item']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            rows.append(rec)
        if not rows:
            print(f"WARNING {bench}: every item failed, nothing written", flush=True)
            continue

        arrays = {"item": np.array([x["item"] for x in rows]), "b": np.array([x["b"] for x in rows], np.float32),
                  "parity_maxabs": np.array([x.get("parity_maxabs", np.nan) for x in rows], np.float32),
                  "skipped": np.array(skipped, dtype=str)}
        for key in rows[0]:
            if "__" in key:
                arrays[key] = np.stack([np.asarray(x[key]) for x in rows])
        path = out_dir / f"{bench}.npz"
        tmp = out_dir / f"{bench}.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)

        dt = time.time() - t1
        done_items += len(g)
        eta_h = (time.time() - t_start) / done_items * (n_todo - done_items) / 3600
        row = {"benchmark": bench, "items": len(rows), "skipped": len(skipped), "seconds": round(dt, 1),
               "tokens": n_tok, "parity_median": float(np.nanmedian(arrays["parity_maxabs"]))
               if np.isfinite(arrays["parity_maxabs"]).any() else np.nan,
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0}
        for p in prompts:
            row[f"digit_mass_{p}"] = round(float(arrays[f"{p}__digit_mass"][:, -1].mean()), 3)
        pd.DataFrame([row]).to_csv(progress, mode="a", header=not progress.exists(), index=False)
        print(f"{bench}: {len(rows)} items ({len(skipped)} skipped) in {dt:.0f}s | parity {row['parity_median']:.2g} | "
              f"digit mass {' '.join(f'{p}={row[f'digit_mass_{p}']}' for p in prompts)} | "
              f"{(time.time() - t0) / 3600:.2f} h, ETA {eta_h:.2f} h", flush=True)

    if state.get("last_is_normed") is not None:  # a resumed run with nothing left keeps the earlier value
        cfg["last_is_normed"] = state["last_is_normed"]
    elif cfg_path.exists():
        cfg["last_is_normed"] = json.loads(cfg_path.read_text()).get("last_is_normed")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
