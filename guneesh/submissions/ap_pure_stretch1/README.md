# Submission `ap_pure_stretch1`

**In one sentence:** the same as `ap_pure`, but with a different ridge regression that is chosen for ranking items **inside** a benchmark, so its difficulty predictions are less squeezed together.

| | |
|---|---|
| Zip | `ap_pure_stretch1.zip` (the same files are unpacked in `files/`) |
| Uses labels? | No |
| Code | Byte-identical to `ap_pure` (`model.py`, `core.py`, `ap_embed_core.py`). Only `ridge.npz` and `shift_priors.json` differ. |
| Codabench result | Uploaded. The result was still pending when this README was written (Codabench was down). |
| Local 5-fold result | Brier ALC **0.2584** (vs 0.2588 for `ap_pure`) |
| Status | Experiment. Not better on its own. The ridge is useful together with the label improvements (see `label_update_3improvements_tuned` README, "stretch 1" row). |

---

## 1. The problem it tries to fix

The original ridge (in `ap_pure`) predicts difficulties that are far too close together. The true difficulties of items have a spread (standard deviation) of about 1.6, but the ridge predictions have a spread of only about 0.10. So the ridge says "every item is about average", which wastes the information it has about which items are harder.

The reason is how the penalty strength `lambda` (called `alpha` in sklearn) was chosen. The original ridge picks `lambda` by the R² over all items together. That R² mostly rewards getting the **average difficulty of each benchmark** right, and a very strong penalty (1e7) is safest for that. A strong penalty squeezes all predictions toward the mean.

## 2. What stretch 1 changes

Stretch 1 chooses `lambda` by a different score: the R² **within each benchmark**, after removing each benchmark's mean from both the true and predicted difficulties. This score only cares whether the items of one benchmark are put in the right order. We call it `r2_within`. It is implemented in `pipeline/fast_ridge.py`.

* Chosen `lambda`: **1e5** for the final fit (1e6 for fold 3), instead of 1e7. A smaller penalty means less squeezing.
* Within-benchmark mean squared error on held-out folds: 2.791, against 2.815 for the original ridge (a small gain).
* Everything else (embeddings, IRT targets, cross-validation grouped by benchmark) is the same.

## 3. The formula

Same as `ap_pure`:

```
P(correct) = sigmoid(theta(subject) - b_hat(item))
```

Only the ridge numbers inside `b_hat` are different.

## 4. Files

Same list as `ap_pure`. `ridge.npz` here has `alpha` = 1e5. `shift_priors.json` has sigma_delta = 0.587 (measured for this ridge), but `ap_pure` does not use it.

## 5. How it was built

```
python pipeline/ridge_train.py --score r2_within --out_dir data/ridge_within      # trains the ridge (6 fits)
python pipeline/build_submission.py --variant ap_pure --name ap_pure_stretch1 --ridge_dir data/ridge_within
```

## 6. How it was tested

The zip check and the smoke test passed. The expected b_hat values for the Modal GPU check of this ridge are in `data/submission_check_items_ap_pure_stretch1.parquet`. The code is byte-identical to `ap_pure`, which passed the GPU check, and only the ridge numbers differ.

## 7. Scores (local 5-fold simulation, same draws as all other submissions)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `ap_pure` | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| **`ap_pure_stretch1`** | 0.2591 | 0.2591 | 0.2591 | 0.2591 | 0.2591 | 0.2591 | **0.2584** |

## 8. What we learned

* Alone, the gain is tiny (0.0004).
* But combined with the label update, the averaging and the item shifts (and tuned settings), the stretch 1 ridge gives our **best local score so far, 0.2196**, compared with 0.2214 for the same method on the original ridge. The more spread-out difficulties help once the model also knows how uncertain they are.
