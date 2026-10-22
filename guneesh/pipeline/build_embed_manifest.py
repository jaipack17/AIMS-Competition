"""Build the list of items to embed: binary training benchmarks, up to CAP items each.

Per benchmark (analysis/train_benchmarks.csv, response_type == binary):
  * candidate items = items with at least one non-null response;
  * if more than CAP, draw CAP of them without replacement, RNG seeded with crc32(benchmark name);
  * text = Agent Psychometrics no-solution template applied to the item's `content`
    (the same string predict() receives as item_content).
Output: data/embed_manifest/binary_cap{CAP}.parquet with columns
  benchmark, item_id, text, n_tokens (tokenizer of the embedding backbone, special tokens included).

Usage: python pipeline/build_embed_manifest.py --cap 5000 --tokenizer Qwen/Qwen3-VL-8B-Instruct
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

from ap_embeddings import DIFFICULTY_INSTRUCTION, format_qs_instruction

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "measurement-db")


def sample_items(bench: str, cap: int) -> pd.DataFrame:
    r = pd.read_parquet(os.path.join(DATA, bench, "formatted_tables", "responses.parquet"), columns=["item_id", "response"])
    answered = set(r.dropna(subset=["response"]).item_id.unique())
    it = pd.read_parquet(os.path.join(DATA, bench, "formatted_tables", "items.parquet"), columns=["item_id", "content"])
    it = it[it.item_id.isin(answered)].drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    if len(it) > cap:
        rng = np.random.default_rng(zlib.crc32(bench.encode()))
        it = it.iloc[np.sort(rng.choice(len(it), cap, replace=False))].reset_index(drop=True)
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    T = pd.read_csv(os.path.join(ROOT, "analysis", "train_benchmarks.csv"))
    parts = []
    for b in T.loc[T.response_type == "binary", "benchmark"]:
        it = sample_items(b, args.cap)
        text = [format_qs_instruction(question_statement=c, instruction=DIFFICULTY_INSTRUCTION) for c in it.content]
        n_tok = [len(x) for x in tok(text, add_special_tokens=True)["input_ids"]]
        parts.append(pd.DataFrame({"benchmark": b, "item_id": it.item_id.astype(str).values, "text": text, "n_tokens": n_tok}))
    M = pd.concat(parts, ignore_index=True)
    out = os.path.join(ROOT, "data", "embed_manifest", f"binary_cap{args.cap}.parquet")
    M.to_parquet(out, index=False)
    t = M.n_tokens.clip(upper=8192)
    print(f"{out}\n{len(M):,} items, {M.benchmark.nunique()} benchmarks, {M.text.nunique():,} unique texts, "
          f"{t.sum() / 1e6:.1f}M tokens after the 8192 cap, median {int(M.n_tokens.median())}, "
          f"over 8192: {(M.n_tokens > 8192).mean():.2%}")


if __name__ == "__main__":
    main()
