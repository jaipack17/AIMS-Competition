"""Readability features of every embedded item, for STAN (Huang et al., ICDM 2021; github.com/bigdata-ustc/STAN).

STAN's 12 features (paper Table I; repo dataloader.QuestionLoader) come from StanfordCoreNLP POS tags and parse trees of
Chinese text.  Our items are JSON / code / many languages, so we keep the features that need no parser, computed with
the repo's own formulas (dataloader.py lines "type_len ... fk_level"):
    type_len, word_len, ttr, punctuation share (the repo's PU share), ques_len (sum of word lengths),
    average word length (= the repo's per_syl), Gunning-Fog = 0.4 (per_word + 100 per_col),
    FK level = 0.39 per_word + 11.8 per_syl - 15.59,
with per_word = words per sentence, per_col = share of words longer than 2 characters.  Dropped (need a parser): noun /
verb / adverb shares, syntax-tree height, dependency distance.
Tokens: words and single punctuation marks (regex), sentences split on . ! ? and newlines (the repo splits on '.').
Text: the item content exactly as embedded (manifest text without the template head and instruction tail).
Output: data/stan/readability.parquet (benchmark, item, 8 features).   Run:  python pipeline/stan_readability.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

import ap_embeddings as ap

OUT = "data/stan/readability.parquet"
FEATURES = ["type_len", "word_len", "ttr", "pu_share", "ques_len", "avg_word_len", "gunning_fog", "fk_level"]
TOKEN = re.compile(r"\w+|[^\w\s]")
SENT = re.compile(r"[.!?\n]+")
HEAD, TAIL = "Task statement:\n", "\n\n" + ap.DIFFICULTY_INSTRUCTION.strip()


def strip_template(text: str) -> str:
    assert text.startswith(HEAD) and text.endswith(TAIL), "manifest text does not use the expected template"
    return text[len(HEAD):-len(TAIL)]


def features(content: str) -> list[float]:
    words = TOKEN.findall(content)
    if not words:
        return [0.0] * len(FEATURES)
    sentences = [s for s in SENT.split(content) if TOKEN.search(s)] or [content]
    type_len, word_len = len(set(words)), len(words)
    pu = sum(1 for w in words if not w[0].isalnum() and w[0] != "_")
    ques_len = sum(len(w) for w in words)
    per_word = word_len / len(sentences)
    per_col = sum(1 for w in words if len(w) > 2) / word_len
    per_syl = ques_len / word_len
    return [type_len, word_len, type_len / word_len, pu / word_len, ques_len, per_syl,
            0.4 * (per_word + 100 * per_col), 0.39 * per_word + 11.8 * per_syl - 15.59]


def main() -> None:
    man = pd.read_parquet("data/embed_manifest/binary_cap5000.parquet", columns=["benchmark", "item_id", "text"])
    feats = np.array([features(strip_template(t)) for t in man.text], dtype=np.float64)
    out = pd.DataFrame(feats, columns=FEATURES)
    out.insert(0, "item", man.benchmark + "::" + man.item_id)
    out.insert(0, "benchmark", man.benchmark)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(out.describe().T.round(2).to_string())
    print(f"saved {OUT}: {len(out)} items")


if __name__ == "__main__":
    main()
