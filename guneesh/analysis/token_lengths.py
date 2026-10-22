"""Token lengths of the items we would embed (binary training benchmarks).

Sample: per benchmark, items with >= 1 non-null response, up to CAP items, drawn with a crc32-seeded RNG.
Text: the Agent Psychometrics no-solution template (utils/embeddings.py: format_qs_instruction with
DIFFICULTY_INSTRUCTION), tokenized exactly as their embed_items() does (default special tokens, no truncation here).
Writes analysis/token_lengths.parquet (benchmark, item_id, n_tokens).
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "measurement-db")
CAP = 2000
TOKENIZER = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-8B-Instruct"

# --- copied verbatim from agent-psychometrics utils/embeddings.py ---
DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one "
    "floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)


def _sanitize_text(s: str) -> str:
    return "".join((" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch) for ch in (s or ""))


def format_qs_instruction(*, question_statement: str, instruction: str) -> str:
    qs = _sanitize_text(str(question_statement or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\n{instr}".strip()
# --- end copy ---


def sample_items(bench):
    r = pd.read_parquet(os.path.join(DATA, bench, "formatted_tables", "responses.parquet"), columns=["item_id", "response"])
    answered = r.dropna(subset=["response"]).item_id.unique()
    it = pd.read_parquet(os.path.join(DATA, bench, "formatted_tables", "items.parquet"), columns=["item_id", "content"])
    it = it[it.item_id.isin(set(answered))].sort_values("item_id").reset_index(drop=True)
    if len(it) > CAP:
        rng = np.random.default_rng(zlib.crc32(bench.encode()))
        it = it.iloc[np.sort(rng.choice(len(it), CAP, replace=False))]
    return it


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    T = pd.read_csv(os.path.join(HERE, "train_benchmarks.csv"))
    rows = []
    for b in T.loc[T.response_type == "binary", "benchmark"]:
        it = sample_items(b)
        texts = [format_qs_instruction(question_statement=c, instruction=DIFFICULTY_INSTRUCTION) for c in it.content]
        n = [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
        rows.append(pd.DataFrame({"benchmark": b, "item_id": it.item_id.values, "n_tokens": n,
                                  "is_empty": it.content.fillna("").str.strip().eq("").values}))
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(os.path.join(HERE, "token_lengths.parquet"))
    print(f"{len(D):,} items from {D.benchmark.nunique()} benchmarks; empty content: {int(D.is_empty.sum()):,}")
