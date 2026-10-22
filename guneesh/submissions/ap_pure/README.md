# Submission `ap_pure`

**In one sentence:** the plain two-stage method from the Agent Psychometrics (AP) paper. It predicts every answer from the subject's ability and the item's predicted difficulty, and it ignores the revealed labels.

| | |
|---|---|
| Zip | `ap_pure.zip` (the same files are unpacked in `files/`) |
| Uses labels? | No |
| Has `labeling.py`? | No (the platform then picks labels at random, but we ignore them anyway) |
| Codabench result | Brier ALC about **0.217** on 9 subject–benchmark pairs (uploaded, ran without errors) |
| Local 5-fold result | Brier ALC **0.2588** (0.2594 at every budget) |
| Status | Baseline. Superseded by the label versions. |

---

## 1. The idea in simple words

Every AI system (a "subject") has an **ability**, and every question (an "item") has a **difficulty**. If the ability is higher than the difficulty, the subject will probably answer correctly. If it is lower, the subject will probably fail. This is called Item Response Theory (IRT).

The problem is that the test benchmarks are **new**. We have never seen their items, so we do not know their difficulty. AP solves this in two stages:

1. **Stage 1 (training, done once on our computer):** fit an IRT model on the training data. This gives an ability for every known subject and a difficulty for every known item.
2. **Stage 2 (training, done once):** turn the text of every training item into a vector of numbers (an "embedding") with a language model, and train a linear regression (a "ridge" regression) that predicts the item's difficulty from that vector.

At test time, for a new item, we embed its text with the same language model, apply the ridge, and get a predicted difficulty `b_hat`. The subject's ability `theta` is looked up from its fields.

## 2. The formula

```
P(correct) = sigmoid(theta(subject) - b_hat(item))
sigmoid(x) = 1 / (1 + exp(-x))
```

* `theta(subject) = theta_name + theta_harness + theta_provider + theta_effort`. Each part is looked up in `theta.json`. A value never seen in training adds 0.
* `b_hat(item) = ((x - mean) / scale) . coef + intercept`, where `x` is the 4096-number embedding of the item text. The numbers come from `ridge.npz`.
* The embedding is the last-layer, last-token hidden state of `Qwen/Qwen3-VL-8B-Instruct` for the text:

  ```
  Task statement:
  {item_content}

  How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:
  ```

## 3. Files in this submission

| File | What it is |
|---|---|
| `model.py` | The `predict()` function. Very short: it calls `core.subject_theta` and `core.item_b_hat`. |
| `core.py` | Shared code: computes theta from the subject fields, computes b_hat from the item text, loads the language model, caches b_hat on disk. |
| `ap_embed_core.py` | AP's embedding functions, copied word for word (MIT licence), so the test-time vectors equal the training vectors. |
| `models.txt` | One line: `Qwen/Qwen3-VL-8B-Instruct`. The platform downloads this model before our code runs. |
| `theta.json` | The four theta tables from the final IRT fit: 626 names, 157 harnesses, 82 providers, 7 effort levels. |
| `ridge.npz` | The ridge regression: `mean`, `scale`, `coef` (4096 numbers each), `intercept`, `alpha` = 1e7. |
| `shift_priors.json` | Not used by `ap_pure`. It is written by the builder for every submission (sigma_delta = 0.456, sigma_zeta = 1.110). |

## 4. Important details

* **When the model loads.** The model is loaded only in prediction worker processes (where the environment variable `PAIEC_PREDICTION_WORKER_ID` is set). The parent process never needs it, because there is no `labeling.py`.
* **No silent fallback.** If there is no GPU, `core.load_model()` raises an error. We prefer a failed run to a run with wrong predictions.
* **Smoke-test mode.** The starting kit tools set `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1`. In that mode, no model is loaded and `b_hat` is the ridge intercept (the average training difficulty). This is only for local checks.
* **Cache.** Workers restart at every label budget, so `b_hat` is saved to `bhat_cache/<sha256 of text>.txt` in the submission folder. Each item is embedded only once. Files are written under a temporary name and then renamed, so a half-written file is never read.
* **Literal repo ID.** The platform only accepts `from_pretrained("Qwen/Qwen3-VL-8B-Instruct", ...)` with the literal string, not a variable. Our first upload failed with error `PAIEC-HF-REMOTE-CODE` because of this. It is fixed in `core.py`.
* Because labels are ignored, the prediction is the same at all six budgets.

## 5. How it was built

From the project root (`guneesh/`):

```
python pipeline/build_submission.py --variant ap_pure
```

## 6. How it was tested

* `python ../starting_kit/tools/check_submission_zip.py submissions/ap_pure.zip` passed.
* `python ../starting_kit/tools/run_smoke_test.py submissions/ap_pure` passed.
* GPU check on Modal (`modal run pipeline/modal_check_submission.py --variant ap_pure`): b_hat for real training items matched the stored training predictions, and the probabilities matched the formula.
* Uploaded to Codabench: it ran without errors.

## 7. Scores

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| Local 5-fold simulation | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |

Codabench: ALC about 0.217 on 9 pairs. The local and Codabench numbers are not on the same draws, so compare versions only within one of them.

## 8. What we learned

* It works end to end on the real platform, so the whole pipeline (IRT, embeddings, ridge, loading the model offline) is correct.
* Its weakness is that a wrong guess for a benchmark's overall difficulty can never be corrected, because it ignores labels. Using labels (`label_update`) improves the local score by about 0.034.
