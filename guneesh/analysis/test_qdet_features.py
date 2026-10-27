"""Checks of pipeline/qdet_features.py against the original text2props / r2de code and hand-worked examples.

1. Text extraction: JSON question/options/context, chat (last user message), inline "A. ..." options, multimedia
   metadata skipped, optionA..optionE keys.
2. The 6 ling_code features equal text2props' own loop (copied below verbatim, version-1 part) when the options are
   passed as its correct and wrong answers.
3. The published code really drops its 8 version-2 features (its concat-then-assign pattern returns 9 columns).
4. The fixed per-sentence long/short word counts on a hand-worked sentence.
5. Readability equals textstat called directly;  r2de_preprocess equals r2de's text_preprocessor (copied verbatim).
Run from the project root:  python analysis/test_qdet_features.py
"""
import json
import os
import re
import string
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import nltk  # noqa: E402
import textstat  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

import qdet_features as Q  # noqa: E402

TOK = nltk.data.load("tokenizers/punkt/english.pickle")

# 1. text extraction
t = Q.item_texts(json.dumps({"context": "Two people met.", "options": ["Ann", "Bob"], "question": "Who spoke?"}))
assert t["q"] == "Who spoke?" and t["options"] == ["Ann", "Bob"] and "Two people met." in t["full"], t
chat = [{"role": "system", "content": "Answer with a letter."}, {"role": "user", "content": "Q1?\nA: x\nB: y"},
        {"role": "assistant", "content": "A"}, {"role": "user", "content": "What is 2+2?\nA: 3\nB: 4"}]
t = Q.item_texts(json.dumps(chat))
assert t["q"] == "What is 2+2?" and t["options"] == ["3", "4"], t
t = Q.item_texts("Pick the planet.\nA. Mars\nB) Moon\n(C) Sun\nAnswer:")
assert t["q"] == "Pick the planet.\nAnswer:" and t["options"] == ["Mars", "Moon", "Sun"], t
t = Q.item_texts(json.dumps({"multimedia_elements": [{"content_type": "image/png", "location": "images/a.png"},
                                                     {"content_type": "text/plain", "text": "Count the dots."}]}))
assert t["q"] == "Count the dots." and t["full"] == "Count the dots." and t["options"] == [], t
t = Q.item_texts(json.dumps({"narration": "C walks.", "question": "Why?", "optionA": "a", "optionB": "b"}))
assert t["q"] == "Why?" and t["options"] == ["a", "b"], t
t = Q.item_texts(json.dumps({"caption0": "Busy road.", "caption1": "Quiet road."}))  # no question key: all values
assert t["q"] == "Busy road.\nQuiet road.", t
t = Q.item_texts("Q1?\nA. a1\nB. b1\nAnswer: A\n\nQ2?\nA. a2\nB. b2\nC. c2\nAnswer:")  # few-shot: last block only
assert t["options"] == ["a2", "b2", "c2"] and t["q"] == "Q1?\nA. a1\nB. b1\nAnswer: A\n\nQ2?\nAnswer:", t
t = Q.item_texts("Q: What is X?\nA: It is Y.\n\nQ: What is Z?\nA: It is W.")  # Q/A format: no options
assert t["options"] == [], t
t = Q.item_texts(json.dumps({"prompt": "Which?\nA. one\nB. two\nAnswer:", "candidate_continuations": [" A", " B"]}))
assert t["options"] == ["one", "two"] and t["q"] == "Which?\nAnswer:", t  # letter labels -> inline options
t = Q.item_texts("x" * 30000)
assert len(t["q"]) == Q.CAP and len(t["full"]) == Q.CAP
print("1. text extraction OK")


# 2. text2props LinguisticFeaturesComponent.transform, version-1 part, verbatim (dicts as in the original)
def text2props_v1(q_text, answers_text, distractors_text):
    q_text_no_punctuation = q_text.translate(str.maketrans('', '', string.punctuation))
    list_words = q_text_no_punctuation.split(' ')
    try: lexicon_count_question = textstat.lexicon_count(q_text)
    except TypeError: lexicon_count_question = 1
    try: lexicon_count_answers = np.mean([textstat.lexicon_count(x) for x in answers_text])
    except TypeError: lexicon_count_answers = 1
    try: lexicon_count_distractors = np.mean([textstat.lexicon_count(x) for x in distractors_text])
    except TypeError: lexicon_count_distractors = 1
    try: sentence_count_question = textstat.sentence_count(q_text)
    except TypeError: sentence_count_question = 1
    try: sentence_count_answer = np.mean([textstat.sentence_count(x) for x in answers_text])
    except TypeError: sentence_count_answer = 1
    return {
        "lexicon_count_question": lexicon_count_question,
        "sentence_count_question": sentence_count_question,
        "avg_word_len_question": np.mean([len(x) for x in list_words]),
        "lexicon_count_options": lexicon_count_answers,
        "sentence_count_options": sentence_count_answer,
        "ratio_len_question_options": (1 + lexicon_count_question) / (1 + lexicon_count_answers),
    }


cases = [("Which of these is a mammal? Think carefully, then answer.", ["A whale.", "A shark swims.", "Trout"]),
         ("Solve: 3x + 2 = 11.  What is x?", ["3", "4", "five"]),
         ("Translate 'bonjour' into English.", [])]
for q, opts in cases:
    ours = Q.linguistic(q, opts, TOK)
    ref = text2props_v1(q, opts if opts else None, opts if opts else None)
    for k in Q.LING_CODE:
        assert np.isclose(ours[k], ref[k]), (k, ours[k], ref[k])
print("2. ling_code equals text2props' own code")

# 3. the published concat-then-assign pattern loses the version-2 columns
df = pd.DataFrame()
for _ in range(2):
    new_row = pd.DataFrame([{f"v1_feature_{i}": 1.0 for i in range(9)}])
    df = pd.concat([df, new_row], ignore_index=True)
    new_row["character_count_question"] = 5  # what text2props does after the concat
assert "character_count_question" not in df.columns
print("3. published code drops the version-2 features (confirmed)")

# 4. fixed per-sentence word counts: "Extraordinary things happen. Cats run." -> long 1 and 0, short 2 and 2
f = Q.linguistic("Extraordinary things happen. Cats run.", [], TOK)
assert f["avg_num_long_words_per_sentence"] == 0.5 and f["avg_num_short_words_per_sentence"] == 2.0, f
assert f["num_long_words"] == 1 and f["num_short_words"] == 4, f
assert f["avg_nouns_per_sentence"] == 1.0 and f["avg_verbs_per_sentence"] == 1.0, f  # things/Cats, happen/run
print("4. fixed per-sentence word counts OK")

# 5. readability and r2de preprocessing
text = "The quick brown fox jumps over the lazy dog. Photosynthesis converts light energy into chemical energy."
r, failed = Q.readability(text)
assert failed == 0 and np.isclose(r["dale_chall"], textstat.dale_chall_readability_score(text))
assert np.isclose(r["linsear_write_formula"], textstat.linsear_write_formula(text))
_, failed = Q.readability("")
print(f"5a. readability equals textstat (empty text: {failed} of 7 formulas fail and are set to 0)")


def r2de_text_preprocessor(text):  # r2de/utils/text_processing.py with its defaults, verbatim
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = ' '.join([x for x in text.split(' ') if not x.isdigit()])
    text = ''.join([char if char not in string.punctuation else ' ' for char in text])
    text = ' '.join([PorterStemmer().stem(word) for word in text.split()])
    return text


for s in [text, "In 1999, 42 students (and 3 teachers) were RUNNING tests!", "<b>HTML</b> stays; numbers 7 go"]:
    assert Q.r2de_preprocess(s) == r2de_text_preprocessor(s), (Q.r2de_preprocess(s), r2de_text_preprocessor(s))
print("5b. r2de preprocessing equals r2de's code")
print("ALL CHECKS PASSED")
