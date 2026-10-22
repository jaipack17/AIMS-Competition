# Submission `ap_pure_stretch2`

**In one sentence:** the same as `ap_pure`, but the predicted difficulties are spread out by multiplying their distance from the average by a learned factor `k = 4.05`.

| | |
|---|---|
| Zip | `ap_pure_stretch2.zip` (the same files are unpacked in `files/`) |
| Uses labels? | No |
| Code | Byte-identical to `ap_pure`. It adds the file `bhat_stretch.json`, which `core.py` applies whenever the file is present. |
| Codabench result | Uploaded. The result was still pending when this README was written (Codabench was down). |
| Local 5-fold result | Brier ALC **0.2639** (worse than `ap_pure`, 0.2588) |
| Status | Experiment. Worse. Do not use. |

---

## 1. The problem it tries to fix

As in stretch 1: the original ridge predicts difficulties that are far too close together (spread 0.10, against about 1.6 for the true values).

## 2. What stretch 2 changes

Stretch 2 keeps the original ridge but stretches its output:

```
b_stretched = c + k * (b_hat - c)
```

* `c` = **0.0146**: the ridge intercept, which is the mean training difficulty.
* `k` = **4.05**: the slope of a regression of the true difficulty on the predicted difficulty, **within benchmarks**, using only held-out benchmarks of the 5 folds (so `k` is not fitted on the same items the ridge was trained on).
* Computed by `pipeline/stretch_factor.py` and saved in `data/ridge/versionB_4parts_logitnormal/stretch.json`.
* When `k` is estimated on 4 folds and tested on the fifth, the within-benchmark error improves by about 1% in 4 of the 5 folds.

Line in `core.py` that applies it:

```python
b = STRETCH_CENTER + STRETCH_K * (b - STRETCH_CENTER)
```

## 3. The formula

```
P(correct) = sigmoid(theta(subject) - b_stretched(item))
```

## 4. Files

Same as `ap_pure`, plus `bhat_stretch.json` = `{"center": 0.01462, "k": 4.0502}`. `shift_priors.json` has sigma_delta = 0.570 (measured for the stretched predictions; not used by `ap_pure`).

## 5. How it was built

```
python pipeline/stretch_factor.py
python pipeline/build_submission.py --variant ap_pure --name ap_pure_stretch2 --stretch data/ridge/versionB_4parts_logitnormal/stretch.json
```

## 6. Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `ap_pure` | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| **`ap_pure_stretch2`** | 0.2647 | 0.2647 | 0.2647 | 0.2647 | 0.2647 | 0.2647 | **0.2639** |

## 7. What we learned

* Multiplying by 4 makes the ranking a little better on average, but it also makes the predictions much more confident. When the ranking is wrong, a confident prediction costs a lot in Brier score. Overall the score gets worse.
* With the label improvements it only ties the original ridge (0.2214). Stretch 1 is the better way to spread the predictions.
