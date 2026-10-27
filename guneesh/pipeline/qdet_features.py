"""Question-difficulty features of Benedetto (AIED 2023, arXiv 2305.10236) for every embedded item.

Source code: github.com/lucabenedetto/qdet-comparison, which takes its features from
  text2props  (github.com/lucabenedetto/text2props, text2props/modules/feature_engineering/components/_linguistic.py,
               _readability.py)  and
  r2de        (github.com/lucabenedetto/r2de-nlp-to-estimating-irt-parameters, r2de/utils/text_processing.py).
All three are MIT License, Copyright (c) 2023 Luca Benedetto. Library versions as pinned there: textstat 0.7.3, nltk 3.7.

Linguistic features (text2props LinguisticFeaturesComponent, version 2).  Our items have no answer key, so the
"correct choices" / "wrong choices" pair becomes one group, "options" (all answer options of the item):
  ling_code  (6)  what the published code really feeds the model: the "version 2" features are added to new_row
                  after new_row was copied into the output table, so they are silently dropped. Columns:
                  lexicon_count_question, sentence_count_question, avg_word_len_question,
                  lexicon_count_options, sentence_count_options, ratio_len_question_options
  ling_paper (14) ling_code + the 8 "version 2" features as the paper describes them:
                  character_count_question, avg_sentence_length, avg_nouns_per_sentence, avg_verbs_per_sentence,
                  num_long_words, num_short_words, avg_num_long_words_per_sentence, avg_num_short_words_per_sentence
                  FIXED: the code loops over the characters of a sentence (`for w in s`) for the last two, so it
                  counts letters; we count the words of the sentence (same word split as num_long_words).
  Dropped: the 3 correct-answer features (no answer key in our data).
Readability features (text2props ReadabilityFeaturesComponent(use_smog=False, version=2)), 7 textstat indices.
TF-IDF text: r2de's text_preprocessor (lower case, English stop words, numbers, punctuation, Porter stemming) of the
question (QO), question + options (QA) and the full text; the vectorizer itself is fitted later, on training items only.

Text of an item (one generic rule, so it also applies to unseen benchmarks):
  full      all string values of the item JSON (minus metadata keys such as content_type, location, role), or the
            whole content if it is not JSON;
  question  chat items (role/content messages): the last user message;  JSON objects: the values of question-like
            keys (QUESTION_KEYS), or else every non-option value;  plain text: the whole content.
  options   values of option-like keys (OPTION_KEY; ignored when they are bare letters like " A"); otherwise the last
            block of lines like "A. ...", "B) ...", "(C) ..." in the question (it must contain an A and a B line;
            earlier blocks are few-shot examples), which is then removed from the question.
  Both texts are cut to CAP characters (features of 600k-character items would take minutes and say little more).

Output: data/qdet/features.parquet (numeric features) and data/qdet/tfidf_text.parquet (preprocessed texts).
Run:  python pipeline/qdet_features.py [--workers 7] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import nltk  # noqa: E402
import textstat  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from stan_readability import strip_template  # noqa: E402

OUT_DIR = "data/qdet"
MANIFEST = "data/embed_manifest/binary_cap5000.parquet"
CAP = 20000
PART = 10000
META_KEYS = {"content_type", "location", "role", "kind", "input_format", "request_protocol", "language", "database_id",
             "case_id", "user_id", "id", "url", "urls", "port", "proto", "duration"}
QUESTION_KEYS = {"question", "question_sentence", "prompt", "instruction", "instructions", "input", "target_input",
                 "task", "user", "text", "description", "title", "statement", "query", "problem"}
OPTION_KEY = re.compile(r"^(options|choices|candidate_continuations|option_?[A-Za-z0-9])$")
OPTION_LINE = re.compile(r"^\s*(?:\(?[A-J]\)|[A-J][.:：])\s*(?=\S)")  # matches the "A. " prefix only

LING_CODE = ["lexicon_count_question", "sentence_count_question", "avg_word_len_question",
             "lexicon_count_options", "sentence_count_options", "ratio_len_question_options"]
LING_V2 = ["character_count_question", "avg_sentence_length", "avg_nouns_per_sentence", "avg_verbs_per_sentence",
           "num_long_words", "num_short_words", "avg_num_long_words_per_sentence", "avg_num_short_words_per_sentence"]
LING_PAPER = LING_CODE + LING_V2
READ = ["flesch_reading_ease", "flesch_kincaid_grade_level", "automated_readability_index", "gunning_fog_index",
        "coleman_liau", "linsear_write_formula", "dale_chall"]
TEXTS = ("q", "full")
NOUNS = {"NN", "NNS"}
VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


# ---------------------------------------------------------------- text of an item
def _leaves(obj, key=""):
    """(key, string) for every string value, skipping metadata keys; list items inherit their parent's key."""
    if isinstance(obj, str):
        if key not in META_KEYS and obj.strip():
            yield key, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v, key)


def _chat(obj):
    """The messages list if obj is a chat (a list of role/content dicts, or {"messages": [...]}), else None."""
    msgs = obj.get("messages") if isinstance(obj, dict) else obj
    if isinstance(msgs, list) and msgs and all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
        return msgs
    return None


def _split_option_lines(question: str) -> tuple[str, list[str]]:
    """Take the LAST block of consecutive option lines (earlier blocks are few-shot examples) if it has an A and a B
    line (so "Q: ... / A: ..." answer lines are not options); remove that block from the question."""
    lines = question.split("\n")
    blocks, start = [], None
    for i, ln in enumerate(lines + [""]):
        if OPTION_LINE.match(ln):
            start = i if start is None else start
        elif start is not None:
            blocks.append((start, i))
            start = None
    for s, e in reversed(blocks):
        letters = {re.sub(r"[^A-J]", "", OPTION_LINE.match(ln).group(0)) for ln in lines[s:e]}
        if {"A", "B"} <= letters:
            return "\n".join(lines[:s] + lines[e:]), [OPTION_LINE.sub("", ln, count=1) for ln in lines[s:e]]
    return question, []


def item_texts(content: str) -> dict:
    """{"q": question text, "full": full text, "options": [option texts]} of one item (see the module docstring)."""
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        obj = None
    if not isinstance(obj, (dict, list)):
        question, full, options = content, content, []
    else:
        leaves = list(_leaves(obj))
        full = "\n".join(v for _, v in leaves)
        options = [v for k, v in leaves if OPTION_KEY.match(k)]
        if all(re.fullmatch(r"[A-J]", o.strip()) for o in options):  # bare letter labels: the text is inline
            options = []
        chat = _chat(obj)
        if chat is not None:
            users = [m["content"] for m in chat if m["role"] == "user" and isinstance(m["content"], str)]
            question = users[-1] if users else full
        else:
            q = [v for k, v in leaves if k in QUESTION_KEYS]
            question = "\n".join(q) if q else "\n".join(v for k, v in leaves if not OPTION_KEY.match(k))
    if not options:
        question, options = _split_option_lines(question)
    return {"q": question[:CAP], "full": full[:CAP], "options": [o[:CAP] for o in options]}


# ---------------------------------------------------------------- features (text2props formulas)
def _lexicon(text: str) -> float:
    return textstat.lexicon_count(text)


def _sentences(text: str) -> float:
    return textstat.sentence_count(text)


def linguistic(q_text: str, options: list[str], tokenizer) -> dict:
    """text2props LinguisticFeaturesComponent(version=2) with the 'options' group (see module docstring)."""
    q_no_punct = q_text.translate(str.maketrans("", "", string.punctuation))
    list_words = q_no_punct.split(" ")
    lex_q = _lexicon(q_text)
    # text2props falls back to 1 when there is no answer text (textstat raises TypeError on a missing value)
    lex_o = float(np.mean([_lexicon(o) for o in options])) if options else 1
    sen_o = float(np.mean([_sentences(o) for o in options])) if options else 1
    row = {
        "lexicon_count_question": lex_q,
        "sentence_count_question": _sentences(q_text),
        "avg_word_len_question": float(np.mean([len(x) for x in list_words])),
        "lexicon_count_options": lex_o,
        "sentence_count_options": sen_o,
        "ratio_len_question_options": (1 + lex_q) / (1 + lex_o),
    }
    sents = tokenizer.tokenize(q_text) or [q_text]
    tags = [nltk.pos_tag(nltk.word_tokenize(s)) for s in sents]
    sent_words = [s.translate(str.maketrans("", "", string.punctuation)).split(" ") for s in sents]
    row.update({
        "character_count_question": len(q_text),
        "avg_sentence_length": float(np.mean([len(s) for s in sents])),
        "avg_nouns_per_sentence": float(np.mean([sum(t in NOUNS for _, t in tg) for tg in tags])),
        "avg_verbs_per_sentence": float(np.mean([sum(t in VERBS for _, t in tg) for tg in tags])),
        "num_long_words": sum(len(x) >= 7 for x in list_words),
        "num_short_words": sum(len(x) < 7 for x in list_words),
        "avg_num_long_words_per_sentence": float(np.mean([sum(len(w) >= 7 for w in ws) for ws in sent_words])),
        "avg_num_short_words_per_sentence": float(np.mean([sum(len(w) < 7 for w in ws) for ws in sent_words])),
    })
    return row


READ_FUNCS = {
    "flesch_reading_ease": textstat.flesch_reading_ease,
    "flesch_kincaid_grade_level": textstat.flesch_kincaid_grade,
    "automated_readability_index": textstat.automated_readability_index,
    "gunning_fog_index": textstat.gunning_fog,
    "coleman_liau": textstat.coleman_liau_index,
    "linsear_write_formula": textstat.linsear_write_formula,
    "dale_chall": textstat.dale_chall_readability_score,
}


def readability(text: str) -> tuple[dict, int]:
    """The 7 textstat indices; a formula that fails on degenerate text (e.g. no words) gives 0 and is counted."""
    row, failed = {}, 0
    for name, f in READ_FUNCS.items():
        try:
            row[name] = float(f(text))
        except (ZeroDivisionError, ValueError, TypeError, IndexError):
            row[name], failed = 0.0, failed + 1
    return row, failed


_STEM = PorterStemmer()
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def r2de_preprocess(text: str) -> str:
    """r2de text_preprocessor with its defaults (html tags kept, as there)."""
    text = text.lower()
    text = " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)
    text = " ".join(x for x in text.split(" ") if not x.isdigit())
    text = text.translate(_PUNCT)
    return " ".join(_STEM.stem(w) for w in text.split())


# ---------------------------------------------------------------- driver
_TOKENIZER = None


def _init_worker():
    global _TOKENIZER
    warnings.filterwarnings("ignore")
    _TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")


def featurize(content: str) -> tuple[dict, dict]:
    t = item_texts(content)
    feats = {"n_options": len(t["options"])}
    for v in TEXTS:
        for k, x in linguistic(t[v], t["options"], _TOKENIZER).items():
            feats[f"{v}__{k}"] = x
        r, failed = readability(t[v])
        feats.update({f"{v}__{k}": x for k, x in r.items()})
        feats[f"{v}__read_failed"] = failed
    texts = {"tfidf_qo": r2de_preprocess(t["q"]),
             "tfidf_qa": r2de_preprocess(" ".join([t["q"]] + t["options"])),
             "tfidf_full": r2de_preprocess(t["full"])}
    return feats, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (timing runs)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    from mlp_b_train import keep_awake  # here, not at the top: workers re-import this module and need no torch
    keep_awake()
    man = pd.read_parquet(MANIFEST, columns=["benchmark", "item_id", "text"])
    if args.limit:
        man = man.sample(args.limit, random_state=0)
    keys = pd.DataFrame({"benchmark": man.benchmark.values, "item": (man.benchmark + "::" + man.item_id).values})
    parts_dir = os.path.join(args.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:  # saved in parts of PART items; a rerun skips done parts
        for start in range(0, len(man), PART):
            path = os.path.join(parts_dir, f"part_{start // PART:03d}.parquet")
            if os.path.exists(path):
                continue
            contents = [strip_template(t) for t in man.text.iloc[start:start + PART]]
            res = pool.map(featurize, contents, chunksize=32)
            k = keys.iloc[start:start + PART].reset_index(drop=True)
            part = pd.concat([k, pd.DataFrame([f for f, _ in res]), pd.DataFrame([t for _, t in res])], axis=1)
            part.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print(f"  items {start + len(part)}/{len(man)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    full = pd.concat([pd.read_parquet(os.path.join(parts_dir, f)) for f in sorted(os.listdir(parts_dir))
                      if f.endswith(".parquet")], ignore_index=True)
    assert len(full) == len(man) and (full["item"].values == keys["item"].values).all(), "parts do not match the manifest"
    text_cols = ["tfidf_qo", "tfidf_qa", "tfidf_full"]
    feats, texts = full.drop(columns=text_cols), full[["benchmark", "item"] + text_cols]
    feats.to_parquet(os.path.join(args.out, "features.parquet"), index=False)
    texts.to_parquet(os.path.join(args.out, "tfidf_text.parquet"), index=False)
    print(f"saved {args.out}: {feats.shape[1] - 2} feature columns; options found for "
          f"{(feats.n_options > 0).mean():.0%} of items; readability failures {(feats.q__read_failed > 0).mean():.2%}")


if __name__ == "__main__":
    main()
