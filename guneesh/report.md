# PAIEC Submissions: Technical Report

This report explains every submission we have built for the Stanford AIMS PAIEC competition. It describes what each submission does, which models and formulas it uses, what was trained, and how it was tested. The last sections describe how a submission runs on Codabench, how we test locally, and what is in the codebase.

All code is in the project folder `AIMS_project`. Built submissions are in `submissions/`.

---

## 1. The task in short

For every test case, the platform calls our function

```python
predict([subject, item], labeled) -> probability
```

and we must return the probability that the **subject** (an AI system, such as a language model with a harness) answers the **item** (one task of a benchmark) correctly. The true outcome is 0 or 1.

* A **subject** is described by 8 text fields: `normalized_name`, `provider`, `release_date`, `access_date`, `harness`, `reasoning_effort`, `harness_version` and `subject_features_extra`.
* An **item** is described by 4 fields: `item_content` (the task text), `item_features`, `interactors` and `benchmark_id` (an anonymous benchmark alias).
* The test benchmarks are **new**. We never see them during training. This is called a benchmark-level cold start.

**Labels.** For each subject and benchmark pair, half of the items form an acquisition pool and the other half form an evaluation pool. Our optional function `acquisition_function` in `labeling.py` may choose to reveal the true outcome (the label) of some acquisition items, up to 31 labels per pair. The revealed labels are passed to `predict()` in the list `labeled`. This list holds the labels of **all** sampled pairs, not only the current pair.

**Score.** The platform scores our predictions on the evaluation pool after 0, 1, 3, 7, 15 and 31 labels per pair. At each of these budgets it computes the **Brier score**, which is the mean of (prediction minus outcome) squared. Lower is better. A constant prediction of 0.5 scores 0.25. The final score, **Brier ALC**, is a weighted average of the six Brier scores with weights 0.1, 0.2, 0.2, 0.2, 0.2 and 0.1. Every subject and benchmark pair counts equally.

---

## 2. Shared foundation used by every submission

All submissions use the same two-stage method, which is taken from the Agent Psychometrics (AP) paper and code. Stage 1 fits an Item Response Theory (IRT) model on training data. Stage 2 trains a regression that predicts item difficulty from the item text, so that we can handle items we have never seen.

### 2.1 Training data

* We use 152 training benchmarks from the competition's measurement database (listed in `analysis/train_benchmarks.csv`): 115 with binary outcomes (0 or 1) and 37 with non-binary outcomes. The non-binary ones are 14 bounded continuous, 8 continuous, 5 five-point Likert, 5 ordinal, 4 fraction and 1 ten-point Likert benchmarks.
* A **cell** is one subject on one item. When a subject answered the same item several times, all attempts are kept in the same cell.

| | Binary benchmarks | Non-binary benchmarks | Total |
|---|---|---|---|
| Benchmarks | 115 | 37 | 152 |
| Items | 1,939,246 | 424,058 | 2,363,304 |
| Subject and benchmark pairs | 9,893 | 3,655 | 13,548 |
| Cells | 24,998,450 | 3,553,143 | 28,551,593 |
| Responses | 58,188,963 | 4,397,434 | 62,586,397 |

* In the subject table there are 13,658 subject and benchmark rows in total, including rows without responses, which the table above leaves out. The subjects carry 638 different public model names, 160 harnesses, 84 providers and 8 reasoning effort values (blank included). 9,974 rows have a blank name and receive a private name (see section 2.2).
* 8,426 of the binary pairs have at least 80 distinct items, which is the size the competition requires for a test pair.
* All of this data is used to fit the IRT model (section 2.2). Only the binary benchmarks are used for the ridge regression (section 2.4), and only for the items that were embedded (section 2.3).
* The 115 binary benchmarks are split at random (seed 0) into **5 folds** of 23 benchmarks each. The folds are used to measure how well our models work on benchmarks they have not seen. The non-binary benchmarks are never held out.

### 2.2 Stage 1: the IRT model (version B)

**Code:** `pipeline/irt_build_data.py` prepares the data. `pipeline/irt_fit.py` fits the model.

IRT describes the chance of a correct answer with two numbers: the **ability** of the subject and the **difficulty** of the item. We use the one-parameter logistic model (1PL, also called the Rasch model):

```
P(correct) = sigmoid(ability - b)
sigmoid(x) = 1 / (1 + exp(-x))
```

Here `b` is the difficulty of the item. When ability equals difficulty, the probability is 0.5.

**Ability is built from four parts.** The test gives us only the 8 subject fields, so the ability must be computed from those fields. We split it into four learned parts:

```
ability(subject) = theta_name + theta_harness + theta_provider + theta_effort
```

* `theta_name` belongs to the model name (`normalized_name`).
* `theta_harness` belongs to the harness, after cleaning: lower case, spaces and underscores replaced by hyphens, version numbers removed, and the Codex variants merged.
* `theta_provider` belongs to the provider (lower case; "unknown" counts as blank).
* `theta_effort` belongs to the reasoning effort (lower case).
* A blank field adds 0. A training subject with a blank name gets a private name that never matches any test subject.

**Per-benchmark deviation.** A subject can be stronger or weaker on one particular benchmark than its overall ability suggests. The model therefore also has a term `zeta` for every subject and benchmark pair, as in the LEGO-MB model:

```
mu = theta_name + theta_harness + theta_provider + theta_effort + zeta(pair) - b(item)
```

**Likelihood.**

* Binary items: `outcome ~ Bernoulli(sigmoid(mu))`. Repeated attempts of the same cell are grouped as a Binomial, which gives exactly the same likelihood.
* Non-binary items: scores are scaled to [0, 1], clipped to [0.01, 0.99], and modelled as `logit(y) ~ Normal(mu, sigma_m^2)`, with one `sigma_m` per benchmark. This is the "logit-normal" option. The non-binary benchmarks help to estimate the subject abilities, but they are never held out.

**Priors.** Every part is drawn from a Normal distribution with mean 0 and a learned spread. The spreads themselves have HalfNormal(1) priors. Abilities are mean-centred.

```
theta_part ~ Normal(0, sigma_part^2)      zeta ~ Normal(0, sigma_zeta^2)      b ~ Normal(0, sigma_b^2)
```

**Fitting.** The model is fitted with stochastic variational inference (SVI) in Pyro, exactly as in AP's code: Trace_ELBO loss, ClippedAdam optimiser with learning rate 0.01, full batch, 2000 epochs. We run 6 fits: one for each fold (fold *f* leaves out the 23 binary benchmarks of fold *f*) and one "final" fit on all data. The final fit took 23 minutes.

**Learned spreads (final fit):**

| Spread | Value |
|---|---|
| sigma_name | 0.377 |
| sigma_harness | 0.647 |
| sigma_provider | 0.455 |
| sigma_effort | 0.528 |
| sigma_zeta (per-benchmark deviation) | 1.110 |
| sigma_b (item difficulty) | 1.811 |

**Output** (in `data/irt/versionB_4parts_logitnormal/<fit>/`): the item difficulties `b`, the four theta tables and the zeta values. The submissions use the final fit. It contains 626 public names, 157 harnesses, 82 providers and 7 effort levels.

### 2.3 Item embeddings

**Code:** `pipeline/build_embed_manifest.py`, `pipeline/embed_manifest.py`, `pipeline/modal_embed.py`.

To predict the difficulty of a new item, we first turn its text into a vector (an embedding) with a language model.

* **Model:** `Qwen/Qwen3-VL-8B-Instruct`, loaded in bfloat16.
* **Input text:** AP's "no solution" template:

  ```
  Task statement:
  {item_content}

  How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:
  ```

* **Vector:** the hidden state of the last layer at the last token. It has 4096 numbers.
* **Settings:** at most 8192 tokens, with truncation from the left, batch size 1 and a fixed seed of 0 with deterministic algorithms. These are AP's defaults.
* **Items embedded:** up to 5000 items per binary benchmark, drawn at random from the items that have at least one response. In total, 255,286 items from the 115 binary benchmarks. 33 benchmarks have more than 5000 items and were capped; the other 82 were embedded completely. Embedding all 1.94 million binary items would have cost far more GPU time, and 5000 items per benchmark are enough to train the ridge.
* The embedding ran on a Modal A100-40GB GPU. The vectors are stored in `data/emb/qwen3vl8b/`, one file per benchmark.
* AP's embedding functions are copied word for word into `submission_src/common/ap_embed_core.py`. The submission therefore produces exactly the same vectors at test time.

### 2.4 Stage 2: the ridge regression from embedding to difficulty

**Code:** `pipeline/ridge_train.py` with the exact solver `pipeline/fast_ridge.py`.

The ridge regression learns to predict the IRT difficulty `b` of an item from its embedding:

```
b_hat = ((x - mean) / scale) . coef + intercept
```

* `x` is the 4096-number embedding. `mean` and `scale` standardise each number (as in sklearn's StandardScaler).
* The penalty strength `lambda` (called alpha in sklearn) is chosen by 5-fold cross-validation grouped by benchmark, so the items of one benchmark are never split between training and validation. The selection criterion is R², as in AP's code.
* The search grid is AP's grid (0.01 to 100,000) plus 1e6 and 1e7, and it extends itself if the best value is at the edge.
* `fast_ridge.py` solves the ridge exactly for many lambda values at once. We checked it against sklearn: the same lambda was chosen in all 12 fits, and the predictions differ by at most 4e-6.

**Result for the original ridge:** lambda = 1e7. The ridge explains only a small part of the difficulty. On held-out benchmarks, the correlation between predicted and true difficulty is about 0.06 to 0.14 overall and 0.10 to 0.17 within a benchmark. The predictions are also strongly shrunk: their spread is 0.10, while the true spread is about 1.6.

### 2.5 How the submission computes theta and b_hat

**Code:** `submission_src/common/core.py`, shared by every submission.

* `subject_theta(subject)` cleans the four fields in exactly the same way as the training code, looks up each part in `theta.json`, and adds them. A value that was never seen in training adds 0, and the other parts are still used.
* `item_b_hat(item)` builds the template text, runs Qwen3-VL-8B to get the embedding, and applies the ridge from `ridge.npz`. The result is saved in a small cache file named by the SHA-256 hash of the text, so the same item is never embedded twice.
* `load_model()` loads Qwen3-VL-8B from the local Hugging Face cache only (the hub is offline on the platform), with the literal repo ID `"Qwen/Qwen3-VL-8B-Instruct"` and without `trust_remote_code`. It stops with an error if no GPU is present, so a broken setup can never silently return bad predictions.

---

## 3. Submission `ap_pure.zip`

**Idea:** the plain AP two-stage method. It ignores the revealed labels.

**Formula:**

```
P(correct) = sigmoid(theta(subject) - b_hat(item))
```

**What it uses:**

* `theta.json` from the final IRT fit.
* `ridge.npz`, the original ridge (lambda = 1e7).
* Qwen3-VL-8B-Instruct for the embeddings.

**Files in the zip:** `model.py`, `core.py`, `ap_embed_core.py`, `models.txt`, `theta.json`, `ridge.npz`, `shift_priors.json`. There is no `labeling.py`, so the platform uses its default random label selection. Since the labels are not used, the prediction is the same at every budget.

**Model loading:** the embedding model is loaded only inside prediction workers (processes where the variable `PAIEC_PREDICTION_WORKER_ID` is set). The parent process never needs it.

**Codabench result:** 9 pairs, Brier ALC about 0.217, the same at every budget.

---

## 4. The two stretched versions

The original ridge predicts difficulties that are far too close together (spread 0.10 against about 1.6). The two stretched versions try to fix this in two different ways. Both exist as an `ap_pure` version (`ap_pure_stretch1.zip`, `ap_pure_stretch2.zip`) and as a `label_update` version (`label_update_stretch1.zip`, `label_update_stretch2.zip`). The code is the same as in the base submission. Only the ridge file and one setting change.

### 4.1 Stretch 1: choose lambda by within-benchmark accuracy

**Code:** `pipeline/ridge_train.py --score r2_within --out_dir data/ridge_within`.

The original ridge chooses lambda by the R² over all items together. That R² mostly rewards getting the average difficulty of each benchmark right. Stretch 1 instead chooses lambda by the R² **within** each benchmark, after removing each benchmark's mean. This rewards ranking the items of one benchmark correctly.

* Chosen lambda: 1e5 (1e6 for fold 3), instead of 1e7. A smaller penalty means less shrinking.
* Within-benchmark mean squared error on held-out folds: 2.791, against 2.815 for the original ridge.
* For the label version, sigma_delta becomes 0.587 (see section 5).

### 4.2 Stretch 2: multiply the predictions by a learned factor

**Code:** `pipeline/stretch_factor.py`, output `data/ridge/versionB_4parts_logitnormal/stretch.json`.

Stretch 2 keeps the original ridge but spreads its predictions further apart:

```
b_stretched = c + k * (b_hat - c)
```

* `c` = 0.0146, the mean training difficulty (the ridge intercept).
* `k` = 4.05. It is the slope of a regression of the true difficulty on the predicted difficulty, within benchmarks, using only held-out benchmarks of the 5 folds.
* When `k` is estimated on 4 folds and tested on the fifth, the within-benchmark error improves by about 1% in 4 of the 5 folds.
* For the label version, sigma_delta becomes 0.570.

The value of `k` is stored in `bhat_stretch.json` inside the zip. `core.py` applies it whenever that file is present.

**Codabench result:** the stretched versions were uploaded, and the results are still pending.

---

## 5. Submission `label_update.zip`

**Idea:** use the revealed labels to correct the prediction. Labels can show that a benchmark is easier or harder than the ridge predicted, or that a subject is stronger or weaker on this benchmark than its theta suggests.

### 5.1 Formula

```
P(correct) = sigmoid( (theta(subject) + zeta) - (b_hat(item) + delta) )
```

* `delta` is a shift of the benchmark's overall difficulty. It is shared by all subjects on that benchmark, so every subject's labels on the benchmark help to estimate it.
* `zeta` is the subject's own shift on this benchmark. Only that subject's labels on that benchmark inform it.
* Labels of other benchmarks have no effect.

### 5.2 How delta and zeta are estimated

Both shifts start at 0 and have Normal priors:

```
delta ~ Normal(0, sigma_delta^2)       zeta ~ Normal(0, sigma_zeta^2)
```

* `sigma_zeta` = 1.110, taken from the IRT fit.
* `sigma_delta` = 0.456. It is measured as the root mean square, over the 115 held-out benchmarks of the 5 folds, of (true mean difficulty of the benchmark minus mean predicted difficulty). It tells how far the ridge's guess of a benchmark's overall level is usually off.

Given the labels of one benchmark, the code finds the most probable values of delta and all zetas together (the MAP estimate). The objective is the logistic log-likelihood of the labels plus the two Gaussian priors. The objective is strictly concave, so there is one unique best answer, and it is found with Newton's method with backtracking. The result is stored in memory, so it is computed only once for each set of labels.

With no labels, both shifts are 0 and the prediction equals `ap_pure`.

### 5.3 Label selection (`labeling.py`)

The platform shows acquisition candidates one at a time, and we must decide at once whether to buy each label. We use **Fisher information**. A label on an item with predicted probability `p` carries information `p(1 - p)` about the shifts. This is largest (0.25) at `p = 0.5` and small when the outcome is nearly certain. The rule is:

```
r   = labels_remaining / items_remaining
buy if r >= 1
otherwise buy if p(1 - p) >= (1 - r^2) / 4        (the same as |p - 0.5| <= r / 2)
```

If predictions were spread evenly between 0 and 1, this rule would buy about a share `r` of the candidates, so the budget is used while the most informative items are preferred. It never buys when no labels remain.

### 5.4 Model loading

The hook needs a prediction for every candidate, and those predictions are made in the parent process. So in this submission the embedding model is loaded at import in every process, including the parent.

**Codabench result:** 8 pairs, Brier ALC about 0.178, against about 0.217 for `ap_pure`. Pairs with a large error at budget 0 improved strongly with labels. For example, one pair went from 0.640 to 0.056, and another from 0.241 to 0.004. The two submissions were scored on different random draws, so this comparison is only rough.

---

## 6. Submission `label_update_3improvements.zip`

This submission adds three changes to `label_update`. It uses the original ridge, sigma_delta = 0.456 and sigma_zeta = 1.110. Each change can be switched on or off in `shift_priors.json`. With all three switched off, it gives exactly the same predictions as `label_update`.

### 6.1 Improvement 1: release date for model names not seen in training

**Code:** `pipeline/name_date_prior.py` fits the rule; `core.py` applies it.

When a test subject's name is not in training, its `theta_name` used to be 0. Newer models are usually stronger, so we now estimate `theta_name` from the release date:

```
years      = (release_date - 1 January 2024) in years, kept between -1 and 2 (that is, 2023 to 2026)
theta_name = -0.154 + 0.217 * years
```

* The two numbers come from an ordinary least squares line through the 595 named training models with a readable release date. The line is fitted once on our computer and stored in `name_date_prior.json`.
* The average `theta_name` by release year in training is about -0.32 for 2022, -0.30 for 2023, -0.03 for 2024, +0.15 for 2025 and +0.22 for 2026.
* When each model's own theta is hidden and predicted from the others, the error falls from 0.516 (predicting 0) to 0.485. The line also beats 0 in each of the 5 fold fits.
* The date is read from `subject["release_date"]`. The formats `YYYY-MM-DD`, `YYYY-MM` (read as the 15th of the month) and `YYYY` (read as 1 July) are accepted. A blank or unreadable date keeps the old value of 0.
* Known names are not affected. A blank name counts as unseen.
* For an unseen name, the prior variance of zeta is also increased by the error of the line (0.485²), or by 0.516² when the date is missing, because its theta is less certain.

### 6.2 Improvement 2: averaging over the remaining uncertainty

The old formula treats `theta`, `b_hat`, `delta` and `zeta` as exact numbers. They are not. For the Brier score, the best prediction is the **average** of `sigmoid(z)` over our uncertainty about `z`. When `z` has mean `m` and variance `v`, the standard probit approximation gives

```
P(correct) = sigmoid( m / sqrt(1 + pi * v / 8) )
```

This pulls the prediction toward 0.5 when we are unsure, and it never moves it past 0.5. The variance `v` adds:

* the uncertainty of the fitted shifts, from the Laplace approximation (the inverse of the Hessian of the objective at the best fit);
* the full prior variance of every shift that has no labels yet (delta, zeta, and the item shift of section 6.3).

As labels arrive, the fitted shifts become more certain, so `v` shrinks and the pull toward 0.5 becomes weaker.

### 6.3 Improvement 3: item difficulty from other subjects' labels

**Idea:** if one subject answers an item wrongly, that item is probably also harder for other subjects. A pair has about 100 to 150 items and buys up to 31 labels, and a test benchmark often has 2 or 3 subjects. So a good share of each subject's evaluation items can receive a label from another subject.

We add a shift `epsilon` for every labelled item:

```
P(correct) = sigmoid( (theta + zeta) - (b_hat + delta + epsilon) )
epsilon ~ Normal(0, sigma_item^2)
```

* `epsilon` is estimated together with `delta` and `zeta` in the same Newton step. An item with no labels keeps `epsilon = 0`.
* Because of the prior, one label moves `epsilon` only part of the way.
* An item is matched by all 4 of its fields.
* `sigma_item` measures how far an item's true difficulty usually is from the ridge prediction within its benchmark. It is computed on the held-out items of the 5 folds as 1.624, and then tuned (section 6.4).

### 6.4 Tuning sigma_item

The local simulation (section 9.4) showed that the measured value 1.624 is too large. It made the item shifts react too strongly to single labels and made the averaging pull too hard at high budgets. We tried scales from 0.25 to 1.25 of the measured value:

| Scale | sigma_item | Brier ALC |
|---|---|---|
| 0.25 | 0.41 | 0.2232 |
| 0.5 | 0.81 | 0.2228 |
| **0.6** | **0.97** | **0.2227** |
| 0.75 | 1.22 | 0.2230 |
| 1.0 | 1.62 | 0.2235 |
| 1.25 | 2.03 | 0.2246 |

When the scale was chosen on 4 folds and tested on the fifth, 0.6 was chosen every time, and it beat `label_update` on all 5 folds. The zip therefore uses **sigma_item = 0.975**.

### 6.5 Result in the local simulation

`label_update` scores a Brier ALC of 0.2246 and this submission scores 0.2227, which is 0.0020 better. The gain is about 4 times its standard error. Budget 0 improves most (from 0.2594 to 0.2525), because averaging helps most when there are no labels.

---

## 7. Submission `label_update_3improvements_tuned.zip` (recommended)

This submission is identical to `label_update_3improvements.zip`, with one change: **sigma_delta = 0.912** instead of 0.456. The code, theta, ridge and date rule files are byte for byte the same. Only `shift_priors.json` differs.

### 7.1 Tuning sigma_delta and sigma_zeta

With sigma_item fixed at 0.975, we tried every combination of sigma_zeta scales (0.5 to 1.5) and sigma_delta scales (0.5 to 4) in the local simulation:

* **sigma_delta should be larger than measured.** Most of the gain is reached at 2 times the measured value. Beyond that, the curve is flat and noisier.
* **sigma_zeta is already right.** Smaller values are clearly worse, and larger values give no consistent gain.
* We chose sigma_delta = 2 × 0.456 = 0.912 and kept sigma_zeta = 1.110, because this setting beats `label_update_3improvements` on all 5 folds.

A likely reason is that the measured sigma_delta only counts the error of the ridge's benchmark mean. On a new benchmark, the overall level of the subjects can also be off, and this extra error is shared by all subjects on the benchmark, so it behaves like delta.

### 7.2 Result in the local simulation

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | Brier ALC |
|---|---|---|---|---|---|---|---|
| `ap_pure` (no labels) | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| `label_update_3improvements` | 0.2525 | 0.2403 | 0.2271 | 0.2151 | 0.2059 | 0.1996 | 0.2227 |
| **`label_update_3improvements_tuned`** | **0.2509** | **0.2382** | **0.2254** | **0.2139** | **0.2051** | **0.1992** | **0.2214** |

The tuned version is the best at every budget. It is about 1.5% better than `label_update`. This is a small gain, so one Codabench upload may not show it clearly, because each upload is scored on a different random draw of pairs.

### 7.3 What each change contributes

We ran all 8 on and off combinations of the three changes (with the untuned sigma_item = 1.62) on the same simulated draws:

* **Averaging** helps most at budgets 0 to 3 (budget 0 improves by 0.009).
* **Item shifts** help most at budgets 15 and 31 (budget 31 improves by 0.005).
* **The release date rule** has no measurable effect in the simulation, because most subjects in the held-out folds have names that were seen in training. It can only help on Codabench when a test subject is a new model.

---

## 8. Other zips in the folder

| Zip | Description |
|---|---|
| `label_update_date.zip` | `label_update` plus the release date rule only |
| `label_update_stretch1_date.zip` | `label_update_stretch1` plus the release date rule |
| `label_update_stretch2_date.zip` | `label_update_stretch2` plus the release date rule |

These were built before the averaging and the item shifts were added. They pass all checks but are not part of the main line.

---

## 9. How we test

### 9.1 Unit and logic tests (on the laptop)

* `analysis/test_label_update.py <zip folder>` checks that:
  * the Newton fit agrees with scipy's BFGS optimiser (difference below 1e-7), and its covariance is the exact inverse Hessian;
  * the item shifts recover the true values when there are many labels;
  * with no labels and all switches off, the prediction equals `ap_pure`;
  * labels of another benchmark change nothing, and correct labels raise the prediction;
  * another subject's wrong answer lowers the prediction for that item only;
  * averaging pulls toward 0.5, never crosses it, and fades as labels arrive;
  * the label hook buys and skips correctly.
* `analysis/test_name_date_prior.py <zip folder>` checks that the submission reads all 363 different training date strings exactly like the fitting script, that odd dates give the expected result, and that known names are unchanged.
* `analysis/ridge_vs_sklearn.py` proves that our fast ridge solver equals sklearn.

### 9.2 Starting kit tools

* `../starting_kit/tools/check_submission_zip.py` checks the zip layout.
* `../starting_kit/tools/run_smoke_test.py` imports the submission and makes test predictions. These tools set `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1`. In that mode, our code does not load the GPU model and uses the ridge intercept as b_hat.

### 9.3 GPU end-to-end check on Modal

`modal run pipeline/modal_check_submission.py --variant <name>` runs the real submission on an A100-40GB with the hub offline, as on Codabench. It checks that:

* the model loads;
* b_hat for 36 real training items matches the ridge prediction from the stored training embedding (the difference is below 1e-15);
* each probability matches the formula;
* the second pass is served from the cache;
* labels move the prediction in the right direction.

On the A100, loading takes about 20 to 30 seconds and a new item takes about 0.16 seconds.

### 9.4 Local competition simulation

**Code:** `analysis/simulate_alc.py`.

The simulation copies the competition's streaming evaluation on our own held-out benchmarks, so that we can compare versions reliably without waiting for Codabench.

* For fold *f*, the test benchmarks are the 23 benchmarks that fold *f*'s IRT fit and ridge never saw. Theta comes from fold *f*'s IRT fit, b_hat from fold *f*'s ridge, and the outcomes are the real recorded responses.
* The submission's own `model.py` and `labeling.py` are run. Only the GPU embedding step is replaced by a lookup of the stored predictions.
* One draw picks 7 benchmarks, keeps 150 random items of each (the same items for all subjects of a benchmark), and adds subject and benchmark pairs with at least 80 items up to 1,000 items in total, preferring benchmarks and subjects that appear less often. This matches what we see on Codabench: about 8 pairs, often with 2 or 3 subjects on one benchmark.
* Each pair is split 50/50. Labels are bought with the real hook before each budget, and the Brier score is computed at budgets 0, 1, 3, 7, 15 and 31.
* We ran 30 draws per fold, so 150 draws with 1,111 pairs in total. Every version sees exactly the same draws, splits and label outcomes, so the differences between versions are fair.
* To check that tuning does not fit noise, each value is chosen on 4 folds and scored on the fifth.
* Limitation: the simulated benchmarks are our training benchmarks, not the real test benchmarks, and the absolute scores differ from Codabench. The simulation is meant for comparing versions, not for predicting the leaderboard score.

---

## 10. How a submission runs on Codabench

* **Zip layout:** all files sit at the root of the zip. `models.txt` lists the Hugging Face repos we use (only `Qwen/Qwen3-VL-8B-Instruct`). The platform downloads them in advance, and the hub is offline while our code runs.
* **Literal repo ID:** the platform requires model loading calls to use the literal repo ID written in `models.txt`. Our first upload failed with the error PAIEC-HF-REMOTE-CODE because the loader passed the repo name through a variable. `core.py` now calls `from_pretrained("Qwen/Qwen3-VL-8B-Instruct", ...)` directly, without `trust_remote_code`.
* **Processes:** the parent process imports `model.py` for validation and label selection. GPU submissions use one prediction worker process. Workers restart at every budget checkpoint.
* **Cache:** because workers restart, the b_hat cache is written to disk in the submission folder, so the evaluation items are embedded only once. Files are written under a temporary name and then renamed, so a half-written file is never read.
* **Memory:** in the label versions, the parent and the worker each hold a copy of the 8B model (about 16 GB each). The `label_update` upload ran successfully, so the GPU has enough memory.
* **Time:** most of the running time is spent on embedding new items. It depends on how many pairs and items the draw contains and on the platform queue.

---

## 11. What is in the codebase

### Data preparation and training (`pipeline/`)

| File | Purpose |
|---|---|
| `irt_build_data.py` | Builds the IRT response tables, the cleaned subject fields and the 5 folds |
| `irt_fit.py` | Fits the version B IRT model (6 fits) |
| `build_embed_manifest.py` | Chooses up to 5000 items per binary benchmark and builds their template texts |
| `embed_manifest.py`, `modal_embed.py`, `ap_embeddings.py` | Computes the Qwen3-VL-8B embeddings with AP's code on a Modal GPU |
| `fast_ridge.py`, `ridge_train.py` | Trains the embedding to difficulty ridge for every fold and the final fit |
| `ridge_sklearn.py` | Re-fits the ridges with sklearn as a check |
| `stretch_factor.py` | Computes the stretch factor k for stretch 2 |
| `name_date_prior.py` | Fits the release date line for unseen names |
| `build_submission.py` | Assembles a submission folder and zip from the source code and the trained files |
| `modal_check_submission.py` | GPU end-to-end check of a built submission |

### Submission source code (`submission_src/`)

| Folder | Files |
|---|---|
| `common/` | `core.py` (theta, b_hat, model loading, cache), `ap_embed_core.py` (AP's embedding functions, copied verbatim), `models.txt` |
| `ap_pure/` | `model.py` |
| `label_update/` | `model.py` (shifts, item shifts, averaging), `labeling.py` (Fisher information hook) |

### Builder options

```
python pipeline/build_submission.py --variant ap_pure|label_update --name <zip name>
    [--ridge_dir data/ridge_within]       stretch 1 ridge
    [--stretch <stretch.json>]            stretch 2 factor
    [--name_prior <name_date_prior.json>] release date rule
    [--item_shifts] [--average]           improvements 3 and 2
    [--sigma_item_scale 0.6] [--sigma_delta_scale 2.0]   tuned settings
```

For example, the recommended zip is built with:

```
python pipeline/build_submission.py --variant label_update --name label_update_3improvements_tuned \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
    --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0
```

### Tests and analysis (`analysis/`)

`test_label_update.py`, `test_name_date_prior.py`, `ridge_vs_sklearn.py` and `simulate_alc.py`, as described in section 9. The simulation results are saved as `analysis/simulate_alc_*.csv`.

---

## 12. Summary and next steps

| Zip | Labels used | Changes | Status |
|---|---|---|---|
| `ap_pure` | No | Plain AP two-stage method | Codabench ALC about 0.217 |
| `ap_pure_stretch1`, `ap_pure_stretch2` | No | Less shrunk difficulty predictions | Results pending |
| `label_update` | Yes | Benchmark shift delta and subject shift zeta, Fisher hook | Codabench ALC about 0.178 |
| `label_update_stretch1`, `label_update_stretch2` | Yes | As above, with the stretched ridges | Results pending |
| `label_update_3improvements` | Yes | Release date rule, averaging, item shifts (sigma_item = 0.975) | Simulation ALC 0.2227 |
| **`label_update_3improvements_tuned`** | Yes | As above, with sigma_delta = 0.912 | **Simulation ALC 0.2214, recommended** |

Possible improvements and a plan for the remaining time are described in section 13.

---

## 13. Possible improvements and plan for the best score

### 13.1 Where the score is lost today

The simulation results show where the error comes from:

* **At budget 0, the error is mostly a level error.** Without labels, we do not know how hard a new benchmark is for a subject, so whole pairs can be predicted too high or too low. The label updates and the averaging already fix a large part of this, and the budget 0 score improved from 0.2594 to 0.2509.
* **At high budgets, the error is mostly an item ranking error.** After 31 labels, the overall level of a pair is known quite well. The remaining error comes from not knowing which items of the benchmark are hard. Our ridge predicts this poorly: the correlation between predicted and true difficulty within a benchmark is only about 0.10 to 0.17. If only the level of a pair is known, the Brier score cannot go below a × (1 - a), where a is the share of correct answers. For a pair with a = 0.5 this is 0.25, and for a = 0.7 it is 0.21. Going below this needs better item difficulty predictions.

So there are two main directions: make better use of the labels, and predict item difficulty better. The second has the larger possible gain but is also the most expensive.

### 13.2 Improvements to the item difficulty prediction

These target the largest remaining error.

1. **Use the best ridge version with the tuned settings.** Stretch 1 and stretch 2 predict the ranking of items within a benchmark slightly better than the original ridge. They have not yet been combined with the three improvements. This is cheap: build the two combinations, run the simulation, and keep the better one. Note that sigma_item and sigma_delta must be re-tuned for each ridge, because they measure that ridge's errors.
2. **Try other embedding layers.** We use the last layer, as AP does. Middle layers of a language model often describe the content of a text better than the last layer. A test on the embedded items of a few benchmarks would show whether the within-benchmark correlation improves before we spend GPU time on all items.
3. **Try a larger or different embedding model.** `models.txt` allows up to 5 repos. A larger model may rank items better, but it needs more GPU memory and time at test time. In the label versions, the parent and the worker each hold a copy of the model, so memory is the main limit.
4. **Add simple item features.** Features such as the length of the item text or the contents of `item_features` can be added to the ridge input. They are cheap to compute and can be tested on the existing data without a GPU.

### 13.3 Improvements to the use of labels

1. **Per-benchmark discrimination.** Our model assumes that a difference of 1 in ability has the same effect on every benchmark. On some benchmarks, outcomes depend more strongly on ability than on others. A second shift per benchmark that scales (theta - b_hat), with a prior centred at 1, could be estimated from the labels in the same Newton step.
2. **The same subject on several benchmarks.** If a subject does better than expected on one benchmark, it is probably also stronger on another. One shared shift per subject across benchmarks could capture this. It only helps when a subject appears in more than one pair of a draw, which seems rare on Codabench.
3. **Check the label selection rule.** The Fisher information rule has not been compared with the platform's default random selection. With item shifts, it may also be useful to prefer items that other subjects of the same benchmark have not been labelled on. Both can be tested in the simulation.

### 13.4 Improvements to the subject ability

1. **Matching new names to known model families.** A new name such as a newer version of a known model could borrow the theta of its closest known name. This could be stronger than the release date rule, but it needs careful name matching and has not been tested.
2. **Unseen harnesses.** An unseen harness currently adds 0. A similar rule based on the harness name or version could be added in the same way as the release date rule.

The simulation showed no measurable effect from the release date rule, because most held-out subjects have known names. Improvements in this group should therefore have a lower priority unless the Codabench results show many new subjects.

### 13.5 Plan for the remaining time (deadline 30 October 2026)

1. **Now:** upload `label_update_3improvements_tuned.zip`, and read the results of the stretch uploads when they arrive.
2. **Next 1 to 2 days (CPU only):** combine stretch 1 and stretch 2 with the three improvements, re-tune sigma_item and sigma_delta for each, and compare everything in the simulation. Build the best combination as a new zip.
3. **Next 2 to 3 days (CPU only):** test per-benchmark discrimination and the label selection variants in the simulation. Keep only changes that improve the score on all or almost all of the 5 folds.
4. **In parallel, if GPU budget allows:** test other embedding layers on a small set of benchmarks. Only re-embed everything if the within-benchmark correlation clearly improves.
5. **Last 2 days before the deadline:** stop adding features. Run every test, the GPU check and the simulation on the final zip, upload it, and finish the written report.

### 13.6 Rules we follow to keep the work safe

* A change is kept only if it improves the simulated score in a way that is clearly larger than the noise, and if it also wins when tuned on 4 folds and tested on the fifth.
* Every switch can be turned off, and with all switches off the code must give exactly the same predictions as the previous version. The tests check this.
* Every zip goes through the logic tests, the starting kit tools, and the GPU check (or is byte-identical in code to a zip that passed it) before upload.
* We never use the private test data or the Codabench feedback to fit model parameters. Codabench results are used only to check that a submission works.
