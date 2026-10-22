# Submission `label_update_3improvements`

**In one sentence:** `label_update` plus three improvements: (1) the release-date rule for unseen model names, (2) averaging the prediction over our remaining uncertainty, and (3) a difficulty shift for each item learned from **other subjects'** labels on the same item.

| | |
|---|---|
| Zip | `label_update_3improvements.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | The **current** code: `submission_src/label_update/model.py` (v2) and `submission_src/common/core.py`. |
| Settings | sigma_delta = 0.456, sigma_zeta = 1.110, **sigma_item = 0.975** (0.6 × measured 1.624), item shifts on, averaging on, date rule on |
| Codabench result | Not known at the time of writing. |
| Local 5-fold result | Brier ALC **0.2227** (vs 0.2246 for `label_update`) |
| Status | Replaced by `label_update_3improvements_tuned` (same code, better sigma_delta). |

---

## 1. The full formula

```
P(correct) = sigmoid( (theta(subject) + zeta) - (b_hat(item) + delta + epsilon) )
```

| Symbol | Meaning | Prior | Learned from |
|---|---|---|---|
| `theta` | the subject's ability (4 parts from `theta.json`; date rule for unseen names) | fixed | training |
| `b_hat` | the item's predicted difficulty (Qwen3-VL-8B embedding → ridge) | fixed | training |
| `delta` | shift of the whole benchmark's difficulty | Normal(0, 0.456²) | labels of all subjects on this benchmark |
| `zeta` | this subject's ability shift on this benchmark | Normal(0, 1.110² + extra for unseen names) | this subject's labels on this benchmark |
| `epsilon` | this item's own difficulty shift | Normal(0, 0.975²) | labels of **any** subject on this exact item |

All shifts of one benchmark are estimated together by one MAP fit (Newton's method), as in `label_update`.

## 2. Improvement 1: release-date rule

For a model name never seen in training:

```
theta_name = -0.154 + 0.217 * years since 1 January 2024   (years clipped to [-1, 2])
```

Known names are not changed. A blank or unreadable date gives 0. Full details are in `label_update_date/README.md`.

**New in this version:** for an unseen name, the prior variance of `zeta` is increased by the error of the line, `0.485²` (or `0.516²` when there is no date). The reason is that the subject's theta is less certain, so the labels should be allowed to move it more. This is `core.subject_theta_var()`.

## 3. Improvement 2: averaging over uncertainty

**The problem.** The old formula uses `sigmoid(m)`, where `m` is our best guess of the logit. But `theta`, `b_hat`, `delta`, `zeta` and `epsilon` are not exact. For the Brier score, the best prediction is the **average** of `sigmoid(z)` over our uncertainty about `z`, and that average is always closer to 0.5 than `sigmoid(m)` when we are unsure.

**The formula.** If `z` has mean `m` and variance `v`, the standard probit approximation gives:

```
P(correct) = sigmoid( m / sqrt(1 + pi * v / 8) )
```

**What goes into `v`:**

* The uncertainty of the fitted shifts. This comes from the **Laplace approximation**: the covariance of the shifts is the inverse of the Hessian of the objective at the best fit. For the prediction, `v = a' Cov a`, where `a` picks out the shifts that appear in this subject's and item's logit.
* The full prior variance of every shift that has **no labels yet**: `sigma_zeta²` (+ unseen-name extra) if this subject has no labels on the benchmark; `sigma_item²` if this item has no labels. (`delta` is always in the fit, so its uncertainty is already in `Cov`.)

**Behaviour.** The prediction is pulled toward 0.5 but never crosses 0.5. As labels arrive, the shifts become more certain, `v` shrinks, and the pull becomes weaker.

## 4. Improvement 3: item shifts from other subjects' labels

**The idea.** If one subject answers an item wrongly, that item is probably harder for other subjects too. A test pair has about 100–150 items and buys up to 31 labels, and a test benchmark often has 2 or 3 subjects. So many of a subject's evaluation items get a label from **another** subject.

**How it works.**

* One `epsilon` per labelled item, estimated in the same Newton fit as `delta` and `zeta`.
* An item is matched by all 4 of its fields (`item_content`, `item_features`, `interactors`, `benchmark_id`).
* Because of the prior, one label only moves `epsilon` part of the way.
* Items with no labels keep `epsilon = 0`.

## 5. Choosing sigma_item (tuning)

`sigma_item` should be "how far an item's true difficulty usually is from the ridge prediction, within its benchmark". Measured on the held-out items of the 5 folds, it is **1.624**. In the local simulation this was too large: the item shifts reacted too strongly to single labels, and the averaging pulled too hard at high budgets. We tried scales of the measured value:

| Scale | sigma_item | Brier ALC |
|---|---|---|
| 0.25 | 0.41 | 0.2232 |
| 0.5 | 0.81 | 0.2228 |
| **0.6** | **0.97** | **0.2227** |
| 0.75 | 1.22 | 0.2230 |
| 1.0 | 1.62 | 0.2235 |
| 1.25 | 2.03 | 0.2246 |

When the scale was chosen on 4 folds and tested on the fifth, **0.6 was chosen every time**, and it beat `label_update` on all 5 folds.

## 6. Switches

All three improvements can be turned off. `model.py` reads them from `shift_priors.json`:

| Key | Meaning |
|---|---|
| `item_shifts` | true/false: improvement 3 |
| `average_over_uncertainty` | true/false: improvement 2 |
| `sigma_item` | prior scale of `epsilon` (also used by the averaging) |
| (file `name_date_prior.json` present or not) | improvement 1 |

With all three off, the predictions are **exactly** those of `label_update` (tested).

## 7. Files

| File | What it is |
|---|---|
| `model.py` | v2: general `fit_shifts(base, y, A, prec)` (Newton + Laplace covariance), `_benchmark_fit`, `predict` with switches |
| `labeling.py` | Fisher-information rule (unchanged) |
| `core.py` | Current version with `subject_theta_var()` and the date rule |
| `name_date_prior.json` | a, b, clip range, leave-one-out errors |
| `shift_priors.json` | all sigmas and switches (see the table at the top) |
| `ap_embed_core.py`, `models.txt`, `theta.json`, `ridge.npz` | as in `ap_pure` (original ridge, `lambda` = 1e7) |

## 8. How it was built and tested

```
python pipeline/build_submission.py --variant label_update --name label_update_3improvements \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
    --item_shifts --average --sigma_item_scale 0.6
python analysis/test_label_update.py label_update_3improvements
python analysis/test_name_date_prior.py label_update_3improvements
python ../starting_kit/tools/check_submission_zip.py submissions/label_update_3improvements.zip
python ../starting_kit/tools/run_smoke_test.py submissions/label_update_3improvements
modal run pipeline/modal_check_submission.py --variant label_update_3improvements
python analysis/simulate_alc.py --sub label_update_3improvements --draws 30 --configs all3 --tag _final
```

* `test_label_update.py`: the Newton fit agrees with scipy BFGS (difference below 1e-4; in practice about 1e-7), the covariance is the exact inverse Hessian, item shifts recover the truth with many labels, switches off = `ap_pure` with no labels, another benchmark's labels change nothing, another subject's wrong answer lowers only that item, averaging pulls toward 0.5 and fades with labels, and the hook behaves correctly.
* Modal A100 GPU check passed: b_hat difference 8e-17, probability difference 1e-16, and labels move p the right way (0.753 → 0.803 after correct labels, → 0.546 after wrong labels).
* The simulation on this exact zip reproduced 0.2227.

## 9. Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| **`label_update_3improvements`** | 0.2525 | 0.2403 | 0.2271 | 0.2151 | 0.2059 | 0.1996 | **0.2227** |

What each change does (all 8 on/off combinations were tried, with the untuned sigma_item = 1.62):

* **Averaging** helps most at budgets 0 to 3 (budget 0 improves by about 0.009).
* **Item shifts** help most at budgets 15 and 31 (budget 31 improves by about 0.005).
* **The date rule** has no measurable effect locally (held-out subjects mostly have known names).
