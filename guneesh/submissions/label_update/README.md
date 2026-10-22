# Submission `label_update`

**In one sentence:** `ap_pure` plus a correction learned from the revealed labels: one shift for the whole benchmark (`delta`) and one shift for each subject on this benchmark (`zeta`). It also has its own label-choosing rule (`labeling.py`).

| | |
|---|---|
| Zip | `label_update.zip` (the same files are unpacked in `files/`) |
| Uses labels? | **Yes** |
| Has `labeling.py`? | Yes: the Fisher-information rule |
| Codabench result | Brier ALC about **0.178** on 8 pairs (uploaded, ran without errors) |
| Local 5-fold result | Brier ALC **0.2246** (vs 0.2588 for `ap_pure`) |
| Status | Uploaded. The base of all later label versions. |

---

## 1. The idea in simple words

At test time the platform can show us the true answer (a "label") for some items. If a subject answers several items correctly that we thought were hard, then probably:

* the whole benchmark is **easier** than our ridge predicted (this affects all subjects on the benchmark), or
* this subject is **stronger on this benchmark** than its usual ability says.

We add two numbers that capture exactly these two things and learn them from the labels.

## 2. The formula

```
P(correct) = sigmoid( (theta(subject) + zeta) - (b_hat(item) + delta) )
```

* `delta` = shift of the benchmark's overall difficulty. It is **shared by all subjects** on that benchmark, so every subject's labels on the benchmark help to estimate it.
* `zeta` = this subject's own ability shift **on this benchmark**. Only that subject's labels on that benchmark inform it. (The same idea is in our IRT model, where every subject–benchmark pair has a `zeta`; it comes from the LEGO-MB model.)
* Labels of **other benchmarks** have no effect.
* With no labels, `delta = zeta = 0` and the prediction is exactly `ap_pure`.

## 3. How `delta` and `zeta` are learned

Both start at 0 and have Normal (Gaussian) priors. The prior says "before any labels, we expect the shift to be about this big":

```
delta ~ Normal(0, sigma_delta^2)      sigma_delta = 0.456
zeta  ~ Normal(0, sigma_zeta^2)       sigma_zeta  = 1.110
```

* `sigma_zeta = 1.110` comes straight from the IRT fit (the learned spread of the per-benchmark deviations).
* `sigma_delta = 0.456` is measured: for each of the 115 held-out benchmarks of the 5 folds, take (true mean difficulty minus mean predicted difficulty), and compute the root mean square. It says how far the ridge's guess of a benchmark's overall level usually is off.

Given all labels of one benchmark (from all subjects, because `labeled` contains the evidence of all sampled pairs), the code finds the **most probable** values of `delta` and all `zeta`s together. This is called the MAP estimate. The objective is:

```
sum over labels of  log-likelihood of the label under the formula     (logistic)
+ the two Gaussian prior terms
```

This objective has exactly one best point (it is strictly concave). The code finds it with **Newton's method** with backtracking (each step is halved until it improves the objective). It is fast: a benchmark has at most a few hundred labels and a handful of parameters.

The result is stored in memory (memoised) by the exact set of labels, so all predictions at one checkpoint reuse it.

## 4. How labels are chosen (`labeling.py`)

The platform shows acquisition candidates **one at a time**, and we must say "buy" or "skip" at once. We prefer labels that teach us the most.

In this model, a label on an item with predicted probability `p` carries **Fisher information** `p(1 - p)` about the shifts. It is largest (0.25) at `p = 0.5` (we are unsure, so the answer is informative), and small when we are already nearly sure.

```
r   = labels_remaining / items_remaining
buy if r >= 1                                   (we can afford all remaining candidates)
otherwise buy if p(1 - p) >= (1 - r^2) / 4      (same as |p - 0.5| <= r / 2)
never buy when labels_remaining = 0
```

If predictions were spread evenly between 0 and 1, this rule buys about a share `r` of the candidates, so the budget gets used while the most informative items are preferred. When called without `context` (old evaluators), it returns the numeric priority `p(1 - p)`.

## 5. Important details

* **The model loads in every process**, including the parent. The parent runs `labeling.py`, which needs a prediction for every candidate, and a prediction needs `b_hat`, which needs the language model. So the parent and the worker each hold a copy of Qwen3-VL-8B (about 16 GB each). The Codabench run worked, so the GPU has enough memory.
* A subject is matched by all 8 of its fields, a benchmark by `benchmark_id`.
* `model.py` here is the **first** version of the label code (only `delta` and `zeta`). The later version in `label_update_3improvements` has switches for the new features, and with the switches off it gives exactly the same predictions as this one (checked by `analysis/test_label_update.py`).

## 6. Files

| File | What it is |
|---|---|
| `model.py` | `predict()` with the shift fit (`fit_shifts`, Newton's method) |
| `labeling.py` | The Fisher-information acquisition rule |
| `core.py`, `ap_embed_core.py`, `models.txt`, `theta.json`, `ridge.npz` | Same as `ap_pure` |
| `shift_priors.json` | `sigma_delta` = 0.4559, `sigma_zeta` = 1.1100 |

## 7. How it was built and tested

```
python pipeline/build_submission.py --variant label_update
python analysis/test_label_update.py label_update
python ../starting_kit/tools/check_submission_zip.py submissions/label_update.zip
python ../starting_kit/tools/run_smoke_test.py submissions/label_update
modal run pipeline/modal_check_submission.py --variant label_update
```

The Modal check also confirmed that correct labels raise the prediction and wrong labels lower it.

## 8. Scores

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `ap_pure` (local) | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| **`label_update` (local)** | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | **0.2246** |

Codabench: about 0.178 on 8 pairs (ap_pure: about 0.217 on 9 pairs; different draws, so this comparison is rough). Some pairs with a big error at budget 0 improved a lot with labels, for example from 0.640 to 0.056, and from 0.241 to 0.004.

## 9. What we learned

* Using labels is the biggest single gain we have found (about 0.034 locally).
* Budget 0 is unchanged, because there are no labels yet. The later "averaging" improvement fixes budget 0.
