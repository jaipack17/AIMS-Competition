# Experiments after the first handover (23–28 October 2026)

This report covers everything Guneesh tried after the first version of this folder was pushed on 23 October. `README.md` explains the base method (IRT + embedding + ridge + label update) and the submissions up to `label_update_3improvements_tuned`. Read that first if you are new.

**Main question of this period:** the oracle test (section 1) showed that most of our remaining error comes from **ranking the items inside a benchmark**. With perfect item difficulties, the local Brier ALC would drop from about 0.225 to about 0.13. So almost every experiment below tries to predict item difficulty better from the item text.

**Short answer:** only one idea helped clearly: a small **MLP** in place of the ridge (within-benchmark r 0.133 → 0.158, local ALC 0.2214 → 0.2198). Everything else (MIRT, Truong's amortized calibration, low-rank subspace, STAN, the QDET text features) gave no gain. The best final zip is still a ridge: `label_update_stretch1_tuned` (local ALC 0.2196).

All work in this period ran on the laptop CPU, except the MIRT and Truong fits, which ran on Modal A100 GPUs.

---

## Contents

0. [Summary table](#0-summary-table)
1. [Oracle ceiling](#1-oracle-ceiling-how-good-could-we-get)
2. [Multidimensional IRT (MIRT)](#2-multidimensional-irt-mirt)
3. [MLP difficulty model](#3-mlp-difficulty-model)
4. [Truong et al.: amortized calibration](#4-truong-et-al-amortized-calibration)
5. [Step A: does better ranking lower the ALC?](#5-step-a-does-better-ranking-lower-the-alc)
6. [Per-benchmark directions](#6-per-benchmark-directions)
7. [Shared low-rank subspace](#7-shared-low-rank-subspace)
8. [Why the predictions look flat](#8-why-the-predictions-look-flat)
9. [STAN: adversarial domain adaptation](#9-stan-adversarial-domain-adaptation)
10. [QDET: linguistic, readability and TF-IDF features](#10-qdet-linguistic-readability-and-tf-idf-features)
11. [New submissions and Codabench results](#11-new-submissions-and-codabench-results)
12. [What we learned and what is left](#12-what-we-learned-and-what-is-left)

---

## How we measure

Two numbers appear everywhere. Both are computed only on benchmarks the model never saw in training.

* **Within-benchmark r**: inside one held-out benchmark, the Pearson correlation between predicted and true item difficulty (true = the final IRT fit). We average it over the 114–115 held-out binary benchmarks (each benchmark is held out in exactly one of the 5 folds). It measures **ranking**. Higher is better. The ridge gets 0.133.
* **Local Brier ALC**: our 5-fold simulation of the competition (`analysis/simulate_alc.py`, README section 8): 150 draws, 1,111 subject–benchmark pairs, identical draws for every method. Lower is better. Differences are paired draw by draw and given as mean ± standard error.

Selection rule used throughout: settings are chosen on 20% of the **training** benchmarks (seed 0, never trained on), never on the held-out benchmarks and never on Codabench.

---

## 0. Summary table

| # | Experiment | Question | Result | Verdict |
|---|---|---|---|---|
| 1 | Oracle | How much would perfect difficulties help? | ALC 0.225 → 0.131 | Difficulty is the main gap |
| 2 | MIRT | Do extra ability dimensions help? | AIC/BIC prefer K = 1; item loadings cannot be predicted from text (R² ≈ 0) | No |
| 3 | MLP | Is a small MLP better than the ridge? | r 0.133 → 0.158 (better on 72 of 114) | **Yes** |
| 4 | Truong et al. | Train b = f(text) on responses directly? | r 0.107 | No |
| 5 | Step A | Does better r lower the ALC? | MLP 0.2198 vs ridge 0.2214 (−0.0016) | Small gain |
| 6 | Directions | Average per-benchmark directions | r 0.157 (uniform), 0.161 (similarity-weighted) | Same as MLP |
| 7 | Low-rank | Do benchmarks share a small difficulty subspace? | PCA spectrum flat; never beats a from-scratch fit | No |
| 9 | STAN | Adversarial adaptation to each new benchmark | r 0.041 vs 0.106 without adaptation | No |
| 10 | QDET | Text features of Benedetto (2023) | Alone: 0.06–0.15 below the embedding; added to it: +0.000 | No |

---

## 1. Oracle ceiling: how good could we get?

**File:** `analysis/oracle_alc.py` → `analysis/oracle_alc_label_update_3improvements_tuned.csv`, `logs/oracle_alc.log`.

The same draws as the simulation, but `b_hat` is replaced by the true IRT difficulty of the held-out items (from the final fit, which saw these benchmarks). This is not reachable at test time; it only shows where the error is. Small run: 1 draw per fold, 37 pairs (the full run was stopped by low memory).

| Rows | ap_pure | label_update | all 3 improvements |
|---|---|---|---|
| real (fold ridge b_hat) | 0.2695 | 0.2277 | 0.2247 |
| oracle b (true difficulty) | 0.1687 | 0.1270 | 0.1313 |
| IRT model scoring its own training responses | | | 0.1020 |

**Meaning:** with perfect difficulties, even the method without labels (0.169) would beat our best label method (0.225). The difficulty prediction is the biggest remaining opportunity. Guneesh agreed with this conclusion and asked to focus on ranking.

---

## 2. Multidimensional IRT (MIRT)

**Files:** `pipeline/mirt_fit.py`, `pipeline/modal_mirt.py`, `pipeline/mirt_ridge.py`, `analysis/test_mirt_fit.py`, `analysis/mirt_mlp_check.py`, `analysis/mirt_fit_summary.csv`, `data/ridge_mirt/*/fold*/summary.json`, `pipeline/hf_upload_mirt.py`, `pipeline/modal_volume_backup.py`.

**Idea:** one ability number per subject may be too little. MIRT gives each subject K abilities and each item K loadings.

**Two forms** (same data, likelihoods, priors and Pyro SVI settings as version B):

```
nested:  logit = theta + zeta + u(subject) . v(item) - b(item)     (K = 0 is exactly version B)
m2pl:    logit = a(item) . theta(subject) - b(item)                 (K = 1 is the 2PL)
```

Warm start from the 1D fit plus an SVD of its residuals. `analysis/test_mirt_fit.py` checks the fitter on simulated data with a known 2-dimensional structure (it recovers the structure; K = 0 matches the 1D fit).

**Fits:** 70 (2 forms × K ∈ {1, 2, 3, 5, 8, 10, 15} × folds 0–4) on Modal A100 GPUs. All fits are on the Hugging Face dataset (section 10 of the README), with `irt/MIRT_README.md`.

**Results:**

1. **Model selection (mean over folds, millions):** AIC and BIC are lowest at K = 1. AIC: m2pl K1 48.67, nested K1 48.79, m2pl K2 50.98, … m2pl K15 98.25. The larger K fits were still improving slowly (loss still dropping 0.1–0.5% in the last 500 epochs).
2. **Stage 2 fails.** To use MIRT on a new item, its loadings v must be predicted from the text. The ridge `pipeline/mirt_ridge.py` (folds 0–1; stopped by low memory) gets cross-validated R² ≈ 0.000 for every loading, at the largest penalties. Regularised MLPs (`analysis/mirt_mlp_check.py`, 9 unseen validation benchmarks of fold 0) also get R² ≤ 0 and within-benchmark r ≈ 0 for the loadings. The variance of the loadings is almost entirely inside benchmarks.
3. **Side discovery:** in the same check, a small MLP ranked **b** better than the ridge (within r 0.182 vs 0.099 on those 9 benchmarks). This led to section 3.

**Verdict:** no gain for unseen items, so MIRT was not simulated or built into a zip. The submission code still supports it (`core.base_logit` with a `mirt.json`; `simulate_alc.py --mirt_run`), and gives exactly the old predictions without it (regression check `analysis/simulate_alc_check_tuned_v2_regression.csv`, max difference 0.0).

---

## 3. MLP difficulty model

**Files:** `pipeline/mlp_b_train.py`, `pipeline/stretch_factor_mlp.py`, `analysis/test_mlp_submission.py`, `data/mlp_b/`, `data/mlp_b_deep7/` (summaries), logs `mlp_b_*.log`.

**Method:** same rows, targets and output as `ridge_train.py` (embedding → 1D IRT b), but an ensemble of small MLPs:

```
standardised embedding -> Gaussian noise (sd 0.1) -> input dropout 0.2
  -> [Linear -> LayerNorm -> GELU -> dropout 0.5] per hidden layer -> Linear -> b
AdamW (lr 1e-4, weight decay 0.05), batch 512, gradient clipping 1, learning rate halved on plateau,
early stopping (patience 6, at most 40 epochs) on the mean within-benchmark r of 20% unseen training benchmarks
prediction = average of 6 networks: hidden sizes (64) and (256, 256), 3 seeds each, fixed in advance
```

Strong regularisation was needed: the first version (lr 1e-3, dropout 0.2) overfit from the first epoch.

**Held-out within-benchmark r** (114 benchmarks where every model has a value; recomputed for this report):

| Model | Mean r | − ridge | Better than ridge on |
|---|---|---|---|
| ridge | 0.133 | | |
| **MLP (6 nets)** | **0.158** | +0.025 ± 0.011 | 72 of 114 |
| deep MLP "deep7" (4096 → 1024 → 512 → 256 → 256 → 128 → 64 → 32 → 1, residual, 3 seeds) | 0.163 | +0.030 ± 0.012 | 77 of 114 |

deep7 is better than the MLP by only +0.005 and on only 49 of 114 benchmarks, so the smaller MLP was kept. Per fold, the MLP gets 0.152 / 0.227 / 0.116 / 0.204 / 0.092.

**In the submission:** the final fit is packed into `mlp.npz` (33 MB). `core.py` runs it in numpy (exact GELU and LayerNorm), so no extra library is needed. Tests (`analysis/test_mlp_submission.py`): numpy = torch to 3.8e-7, and the bundled file reproduces the training predictions to 2.4e-7. `pipeline/stretch_factor_mlp.py` gives the stretch-2 factor for the MLP: k = 0.649 (below 1, because the MLP's predictions are already spread out, unlike the ridge's k = 4.05).

---

## 4. Truong et al.: amortized calibration

**Paper:** Truong et al., "Reliable and Efficient Amortized Model-based Evaluation", ICML 2025 (arXiv 2503.13335). **Results pushed:** `data/truong_amortized/<run>/fold*/summary.json` and `losses.csv`. **Code not pushed:** `pipeline/truong_amortized.py` and `pipeline/modal_truong.py` copy code from `github.com/sangttruong/reeval`, which has no licence.

**Idea:** instead of copying fitted difficulties (two stages), train b = f(embedding) directly on the right/wrong responses, so items with many responses count more.

**What we ran:** their code (linear head, EM with 64-node Gauss–Hermite quadrature over N(0, 1) abilities, Adam lr 1e-3, weight decay 1e-4, up to 10,000 iterations), changed only where our data needs it: a sparse likelihood, test taker = subject–benchmark pair, and an affine map to our b scale. 5 folds on Modal A100s.

**Result:** within-benchmark r **0.107** (ridge 0.133; better than the ridge on 42 of 114). Guneesh confirmed that their final prediction model is the plain Rasch model, so no separate submission was built.

`pipeline/amortized_b_train.py` is our own earlier variant (abilities fixed from our IRT fit; penalised linear or MLP head). It was only run in debug mode and was dropped in favour of the exact method.

---

## 5. Step A: does better ranking lower the ALC?

**Files:** `analysis/simulate_alc_label_update_3improvements_tuned_stepA_{ridge,mlp,truong}.csv`, `logs/sim_stepA_*.log`.

The tuned submission was simulated with each model's held-out b_hat (`simulate_alc.py --ridge_dir`). The sigmas were re-measured for each model with the same rules as the tuned zip (sigma_delta = 2 × measured, sigma_item = 0.6 × measured).

| b_hat from | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| ridge | 0.2517 | 0.2382 | 0.2247 | 0.2144 | 0.2051 | 0.1994 | 0.2214 |
| **MLP** | 0.2485 | 0.2362 | 0.2234 | 0.2130 | 0.2041 | 0.1983 | **0.2198** |
| Truong | 0.2557 | 0.2390 | 0.2244 | 0.2137 | 0.2047 | 0.1989 | 0.2216 |

**Meaning:** the MLP's +0.025 in r becomes −0.0016 in ALC (better on 4 of 5 folds). The gain is real but small: the oracle gap (0.09) needs a much larger improvement in ranking. This result led to the MLP submissions (section 11).

---

## 6. Per-benchmark directions

**Files:** `pipeline/avgdir_train.py`, `data/avgdir_{uniform,simw}/<run>/fold*/summary.json`, `logs/avgdir_train.log`.

**Teammate's finding (24 Oct):** a ridge fitted inside one benchmark ranks its items well (r 0.556), but every benchmark uses its own direction in the embedding space, and these directions are nearly orthogonal (mean cosine 0.009). One pooled ridge mostly learns benchmark identity (about 75% of the embedding's variance is benchmark identity).

**Method:** fit one unit direction per training benchmark (kernel ridge inside the benchmark), then combine:

* **uniform:** the plain average of the directions;
* **simw:** each item weights the directions by its similarity to each training benchmark (softmax of cosine / tau).

Penalty and tau chosen on the selection benchmarks; a final affine map to the b scale.

**Result:** within-benchmark r **0.157** (uniform) and **0.161** (simw), better than the ridge on 73 of 114. This is as good as the MLP, but not better, so it was not built into a zip.

---

## 7. Shared low-rank subspace

**Files:** `pipeline/lowrank_dir.py`, `analysis/test_lowrank_dir.py`, `analysis/lowrank_report.py`, `data/lowrank/*/` (summary, selection and held-out tables), `logs/lowrank_dir.log`.

**Idea** (item-side version of observational scaling laws, Ruan et al. 2024; also considered as a workshop paper): if all benchmark directions lie in a small shared subspace P (k dimensions), a new benchmark is described by only k numbers, which a few labelled items could calibrate:

```
b_j = c_m + s_m * a_m' P z_j        (z_j = item embedding, m = benchmark, P from PCA of the directions)
```

ObsScaling's `perform_pca` is copied verbatim (Apache-2.0). `analysis/test_lowrank_dir.py` checks the method on synthetic benchmarks with a true 3-dimensional subspace (it recovers it).

**Result (110 held-out benchmarks):**

* The PCA spectrum is **flat**: PC1 explains 2.4%, PC10 1.7%.
* Without labels the average direction gives r 0.160.
* With n labelled items, r keeps rising with k up to 70 dimensions, and the shared basis **never beats** a kernel ridge fitted from scratch on the same items:

| n labelled items | 10 | 30 | 100 | 300 | half |
|---|---|---|---|---|---|
| shared basis, k = 1 | 0.118 | 0.153 | 0.182 | 0.177 | 0.181 |
| shared basis, k = 10 | 0.137 | 0.199 | 0.273 | 0.290 | 0.309 |
| shared basis, k = 70 | 0.140 | 0.220 | 0.312 | 0.343 | 0.390 |
| from scratch (no sharing) | 0.145 | 0.230 | 0.328 | 0.371 | 0.441 |

**Verdict:** difficulty is not low-rank across benchmarks. (Small known issue: the per-fold "headline" line in `lowrank_report.py` shows `nan` for some n, because the "all directions" k differs between selection and the final run; the main table is not affected.)

---

## 8. Why the predictions look flat

A diagnosis on 113 held-out benchmarks (scratchpad script, not in the repo):

| | True b | Ridge b_hat | MLP b_hat |
|---|---|---|---|
| Spread inside a benchmark (SD) | 1.56 | 0.06 | 0.30 |
| Spread between benchmark means (SD) | 0.34 | | |
| Within-benchmark r | | 0.136 | 0.163 |
| Share of squared error that is within-benchmark | | 98% | 90% |

The ridge's predictions are almost constant inside a benchmark. But this is a symptom: the ridge shrinks because the text only weakly predicts the ranking. Stretching the predictions (stretch 2) does not add information, which is why it did not help.

---

## 9. STAN: adversarial domain adaptation

**Paper:** Huang et al., "STAN: Adversarial Network for Cross-domain Question Difficulty Prediction", ICDM 2021. **Pushed:** `pipeline/stan_readability.py`, `data/stan/*/fold*/targets/*.json`, `logs/stan_fold*.log`. **Not pushed:** `pipeline/stan_train.py` and its test, which copy code from `github.com/bigdata-ustc/STAN` (no licence).

**Idea:** treat each new benchmark as a new domain and train the encoder so that a discriminator cannot tell training items from the new benchmark's (unlabelled) items.

**What we ran** (choices agreed with Guneesh): our embedding + a small trained layer as the encoder; 8 parser-free readability features; one held-out benchmark per target domain (75% of its items unlabelled for adaptation, 25% for testing); the sampling window of the authors' notebook (w = 0.008). The released repo has no training code and three bugs, which we fixed and unit-tested: the gradient reversal received no gradient, its coefficient stayed at 0, and the attention depended on the batch. The control is the same network trained on the source only.

**Result:** stopped early on purpose (each benchmark took from 30 minutes to several hours on the shared CPU, and the results were negative). On the 20 finished benchmarks: STAN r **0.041** vs source-only **0.106** (−0.065 ± 0.052, STAN better on 7 of 20). The discriminator stayed at 75–98% accuracy, so the domains were never aligned.

---

## 10. QDET: linguistic, readability and TF-IDF features

**Paper:** Benedetto, "A quantitative study of NLP approaches to question difficulty estimation", AIED 2023 (arXiv 2305.10236); code `qdet-comparison`, `text2props`, `r2de` (all MIT). **Files:** `pipeline/qdet_features.py`, `analysis/test_qdet_features.py`, `pipeline/qdet_upper_bound.py`, `pipeline/qdet_combined.py`, `analysis/qdet_report.py`, `analysis/qdet_r_all.csv`, `data/qdet/upper_bound/`, `data/qdet/combined/`, `logs/qdet_*.log`, `logs/qdet_report.txt`.

**Idea (Guneesh's):** the embedding already knows the benchmark; hand-made text features might separate easy from hard items **inside** a benchmark.

**Features** (all 255,286 items): 14 linguistic features (word, sentence and option counts, lengths, nouns and verbs per sentence, long and short words), 7 readability indices (textstat), and R2DE's TF-IDF text. Each is computed on the question only ("q") or the full item JSON ("full"). Two bugs of the published code were fixed and tested: 8 of its linguistic features are silently dropped, and two "per sentence" features count letters instead of words. With the dropped features restored, the linguistic model improves by r +0.036 (6 of 7 benchmarks).

**Test 1: the paper's own setting** (`qdet_upper_bound.py`). Inside each benchmark, train on a random half of the items and test on the other half. This is easier than our task (the benchmark is seen), so it is an upper bound. Baseline: kernel ridge on our embedding with the same split. **Partial run** (still running when pushed; 44 benchmarks with every model):

| Model | r | − embedding |
|---|---|---|
| embedding (kernel ridge) | 0.466 | |
| hybrid: features + TF-IDF (full text) | 0.402 | −0.064 ± 0.020 |
| TF-IDF of question + answers | 0.364 | −0.102 ± 0.027 |
| linguistic + readability (full text) | 0.333 | −0.133 ± 0.022 |
| linguistic only (question) | 0.321 | −0.145 ± 0.025 |

**Test 2: features added to the embedding** (`qdet_combined.py`, all 110 benchmarks):

* **joint:** one kernel ridge on both blocks, with the features' weight chosen by cross-validation (it chose weight 0 on 55 of 110 benchmarks);
* **correction:** a random forest on the features learns what the embedding model got wrong.

| Model | Embedding r → model r | Change | Better / same / worse |
|---|---|---|---|
| correction (question) | 0.451 → 0.451 | +0.000 ± 0.005 | 30 / 43 / 37 |
| joint (question) | 0.451 → 0.450 | −0.001 ± 0.003 | 11 / 77 / 22 |
| correction (full) | 0.451 → 0.448 | −0.003 ± 0.005 | 28 / 39 / 43 |
| joint (full) | 0.451 → 0.448 | −0.003 ± 0.002 | 11 / 79 / 20 |

**Verdict:** the paper's features are much weaker than our embedding and add nothing to it. Big wins on a few benchmarks (auditing_sabotage_bench +0.31, legal_rag_bench +0.17) are cancelled by losses on others (medhelm_medec −0.21). Note that these r values (0.45) are inside-benchmark numbers with half the items labelled, so they are not comparable with the cold-start r of sections 3–7 (0.13–0.16).

---

## 11. New submissions and Codabench results

All are in `submissions/` with their own README; the index is `submissions/README.md`.

* **`label_update_stretch1_tuned`**: the tuned method on the stretch 1 ridge. Local ALC **0.2196**, 0.0018 ± 0.0005 better than `label_update_3improvements_tuned`, at every budget, on 4 of 5 folds. **Recommended final.**
* **7 MLP zips** (`<ridge zip>_mlp`): every label zip with `mlp.npz` instead of `ridge.npz`. `analysis/mlp_variants.py` builds each one and its ridge twin with the same code; the twins reproduce the earlier results exactly. Best: `label_update_3improvements_tuned_mlp` **0.2198** (−0.0016 ± 0.0006 vs its twin), which ties stretch1_tuned. The MLP helps most with stretch 2 (−0.004 on 5 of 5 folds).
* `sim_stretch1`, `sim_stretch2`: the test builds used in the stretch simulations; also uploaded.
* The Fisher-information `labeling.py` is byte-identical in all 18 label submissions.

**Codabench (26–27 Oct):** label_update_stretch2_date 0.1596, label_update_mlp 0.1641, label_update_stretch1_date 0.1649, … ap_pure_stretch1 0.359 (full list in `submissions/README.md`). These do **not** agree with the local ranking, and they should not: each upload is scored on about 8 random pairs. In our simulation, the ALC of one draw of 7–8 pairs has a standard deviation of 0.022 (90% of draws fall between 0.174 and 0.248), and the same zip uploaded twice differs by 0.055 or more in 9% of cases. **We choose on the local result only.**

---

## 12. What we learned and what is left

1. **The gap is item ranking inside a new benchmark.** Perfect difficulties would take the ALC from 0.225 to 0.13.
2. **The text embedding carries only a weak, benchmark-specific ranking signal.** Inside a seen benchmark it ranks well (r ≈ 0.45–0.56). On an unseen benchmark, every method we tried lands between 0.13 and 0.16: ridge, MLP, averaged directions, low-rank, Truong, STAN.
3. **What helped:** the MLP (+0.025 r, −0.0016 ALC) and, earlier, using labels well (−0.034 ALC).
4. **Ideas not tried** (from the discussions): learn the ranking direction of a new benchmark from its bought labels (an online per-benchmark slope, "version C"); share item shifts between similar items (a Gaussian process on the embedding); an LLM rating of difficulty through the approved inference APIs (how API keys reach the scoring run is still unclear); a stretch 1 version of the MLP (about 4–6 hours of CPU).

---

## Code and data not in this repository

* `pipeline/truong_amortized.py`, `pipeline/modal_truong.py`, `pipeline/stan_train.py`, `analysis/test_stan.py`: they copy code from repositories without a licence. Available from Guneesh on request.
* Large outputs: all `pred.parquet` files of the experiments (about 30 MB each), the per-network weights, `data/qdet/features.parquet` (21 MB) and `tfidf_text.parquet` (327 MB), `data/stan/readability.parquet`, the Modal volume backups (40 GB). Every one can be rebuilt with the script named in its section. The 70 MIRT fits are on the Hugging Face dataset.
* Logs that show account names (Modal launch logs, Hugging Face upload logs, backup logs).
