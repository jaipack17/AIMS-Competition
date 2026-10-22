# Guneesh's PAIEC work: the complete handover guide

This folder holds **all** of Guneesh's code, trained parameters, submissions, test results and notes for the Stanford AIMS **Predictive AI Evaluation Competition (PAIEC)**, NeurIPS 2026 (Codabench competition 17828).

It is written so that another person, or another Claude session, can take over the project without asking any questions. Read sections 0 to 3 first. The other sections are reference material.

* Branch: `Guneesh` of `https://github.com/jaipack17/AIMS-Competition`. Everything is inside the folder `guneesh/`. The team's other work (`paiec/`, `scripts/`, `baselines/`, `starting_kit/`, ...) is on `main`; this branch holds only Guneesh's new files.
* **This branch contains only this folder.** The official starting kit is not copied here, because it is already on `main` as `starting_kit/`. Before running the checks, get it next to this folder with:

  ```
  git fetch origin main
  git checkout origin/main -- starting_kit
  ```

  The commands below then call it as `../starting_kit/tools/...`.
* **All commands in this README are run from inside `guneesh/`** (we call it the "project root"). The scripts use paths such as `data/irt/...` and `submissions/...` relative to it.
* State of this README: **22 October 2026**. Deadline for code and report: **30 October 2026, Anywhere on Earth**.

---

## Contents

0. [Quick status: read this first](#0-quick-status-read-this-first)
1. [The competition in short](#1-the-competition-in-short)
2. [Folder map](#2-folder-map)
3. [Results of every method](#3-results-of-every-method)
4. [The method, step by step](#4-the-method-step-by-step)
5. [The submission code in detail](#5-the-submission-code-in-detail)
6. [Every submission](#6-every-submission)
7. [Testing](#7-testing)
8. [The local simulation (our "80/20" test)](#8-the-local-simulation-our-8020-test)
9. [How to rebuild everything from scratch](#9-how-to-rebuild-everything-from-scratch)
10. [What is not in this repository](#10-what-is-not-in-this-repository)
11. [Codabench: how it runs and what happened](#11-codabench-how-it-runs-and-what-happened)
12. [Open questions and next steps](#12-open-questions-and-next-steps)
13. [Working rules](#13-working-rules)
14. [Glossary](#14-glossary)

---

## 0. Quick status: read this first

| Question | Answer |
|---|---|
| What should be uploaded now? | `submissions/label_update_3improvements_tuned/label_update_3improvements_tuned.zip`. Local Brier ALC **0.2214**. All checks passed. |
| What is the best idea found so far? | The same method with the **stretch 1 ridge** (`data/ridge_within`): local ALC **0.2196** (0.0018 ± 0.0005 better, wins at every budget, wins on 4 of 5 folds). **Not built as a zip yet.** Next step: tune `sigma_item` and `sigma_delta` for this ridge, build `label_update_stretch1_tuned.zip`, run all checks. Build command in section 12. |
| Codabench scores so far | `ap_pure` ≈ 0.217 (9 pairs), `label_update` ≈ 0.178 (8 pairs). The four stretch uploads were still pending. Codabench was down on 22 Oct; the organizers said to test locally on the training data instead. |
| Biggest open decision | The organizers confirmed that **calling LLM APIs during scoring is allowed** (Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI, Groq; no web search or browsing). It is not yet clear how API keys are supplied at scoring time or who pays. See section 12.2. |
| Where is the long technical report? | `report.md` in this folder (about 6,300 words). This README contains everything in it and more. |
| Which file must never be read or used? | `SUBMISSION_1.md` in Guneesh's local project. Guneesh asked that it not be used until Guneesh says so. It is **not** in this repository. |
| Secrets | No tokens or keys are in this repository. Never commit a Hugging Face token or API key. Two Modal logs were left out because they contain an account e-mail. |

---

## 1. The competition in short

### 1.1 The task

The platform calls our function

```python
predict([subject, item], labeled) -> float   # probability in [0, 1]
```

and we return the probability that the **subject** (an AI system: a model plus its settings) answers the **item** (one task of a benchmark) correctly. The true outcome is 0 or 1. We never run the subject; we only predict.

* **Subject:** 8 text fields: `normalized_name`, `provider`, `release_date`, `access_date`, `harness`, `reasoning_effort`, `harness_version`, `subject_features_extra`. All are strings; a missing value is `""`. The same base model with different settings is a different subject.
* **Item:** 4 text fields: `item_content` (the task text shown to the subject), `item_features`, `interactors`, `benchmark_id` (an anonymous alias such as `benchmark_274926`).
* **Cold start at the benchmark level:** the test benchmarks are **not** in the training data.

### 1.2 Labels and the streaming protocol

* A test subject–benchmark pair needs at least **80 distinct items**.
* Each pair is split at random, **50/50, separately for each pair**, into an **acquisition pool** and an **evaluation pool**. (Odd count: the extra item goes to acquisition.) Repeated runs of an item stay in the same pool.
* Acquisition candidates arrive **one at a time** in a fixed random order. Our optional `acquisition_function` in `labeling.py` decides at once whether to reveal ("buy") the true outcome. Without `labeling.py`, the platform buys at random.
* At most **31 labels per pair**.
* The revealed labels are passed to `predict()` in `labeled`, a list of `[[subject, item], outcome]`. **It holds the labels of all sampled pairs**, not only the current one. So labels of other subjects on the same benchmark are visible.
* A **draw** contains at most **1,000 distinct items**. The platform picks complete (benchmark, subject) groups, preferring benchmarks and subjects that are represented less often in the draw. On Codabench we saw about 8 or 9 pairs per draw, often 2 or 3 subjects on the same benchmark.

### 1.3 The score: Brier ALC

* At label budgets **0, 1, 3, 7, 15, 31** the platform computes the **Brier score** on the evaluation pool: the mean of (prediction − outcome)². Lower is better. Predicting 0.5 everywhere gives exactly 0.25.
* All response runs of an evaluation item count, with the same prediction.
* **Brier ALC** = weighted average of the six Brier scores with weights **0.1, 0.2, 0.2, 0.2, 0.2, 0.1**. Every pair counts equally.

### 1.4 Runtime rules that matter for our code

* One run has **8 hours**, including setup.
* `models.txt`: at most **5** Hugging Face repos (≤ 300B parameters and ≤ 1,000 GB each, ≤ 1,024 GB together). They are downloaded **before** our code runs; the hub is **offline** at runtime.
* Model loading must use the **literal repo ID** in the call, for example `from_pretrained("Qwen/Qwen3-VL-8B-Instruct", ...)`, not a variable. Otherwise the platform fails with error code `PAIEC-HF-REMOTE-CODE`. `trust_remote_code` is disabled.
* Processes: the **parent** imports `model.py` for validation and runs `labeling.py`. Predictions run in **worker** processes (1 worker for GPU submissions, up to 4 for CPU, up to 16 for API submissions). **Workers restart at every budget checkpoint.** `PAIEC_PREDICTION_WORKER_ID` is set in workers. The submission directory is shared between processes.
* Predictions must be finite and in [0, 1]. Any exception or NaN fails the run. Error codes: 10/11 invalid `model.py`, 12 invalid `labeling.py`, 40 prediction exception, 41 timeout, 42 invalid probability, 50 total timeout.
* **Network:** only the inference APIs of Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI and Groq. "Web search, browsing, and grounding are prohibited, including when offered through an approved provider's API." An organizer (Sang) confirmed on the forum that these APIs are approved, and that "additional assets required by the item will be provided" (for example images).
* **Training data:** only the data sources in the official benchmark inventory (the Hugging Face dataset `aims-foundations/measurement-db`). Do not try to extract private test data or to identify the subjects or benchmarks behind anonymous IDs.
* Pretrained models may be used and must be disclosed in the report. Use of coding assistants (like Claude) must also be disclosed.
* Teams: up to 6 members. Our team: mentor Aman and 5 students.
* **Organizer advice (22 Oct):** "it's much more efficient to test locally using the 80/20 split on the permitted training set", because the submission website was down. Our local 5-fold simulation (section 8) is exactly this kind of test.

The official starting kit is on the `main` branch in `starting_kit/`. The full interface is in `starting_kit/README.md` and `starting_kit/sample_code_submission/model.py`.

---

## 2. Folder map

```
guneesh/
├── README.md                 this file
├── report.md                 the technical report (shorter, more formal version of this README)
├── pipeline/                 training code: data tables, IRT, embeddings, ridge, priors, submission builder, GPU check
├── submission_src/           SOURCE code of the submissions (what goes into the zips)
│   ├── common/               core.py, ap_embed_core.py, models.txt   (shared by every submission)
│   ├── ap_pure/              model.py
│   └── label_update/         model.py, labeling.py
├── submissions/              every BUILT submission: one folder each, with README.md, the .zip and files/ (the unzipped content)
├── analysis/                 data download and exploration scripts, tests, the local simulation and its results (.csv)
├── data/                     small trained outputs only (see section 10 for what is missing and how to rebuild it)
├── logs/                     logs of the long runs (IRT fits, ridge fits, simulations)
├── probe_submission/         a diagnostic "probe" submission (section 11.3) and probe_submission.zip
├── run_ridge.bat             Windows helper: runs ridge_train.py then ridge_sklearn.py, resumable
└── resume_irt_soft.bat       Windows helper: resumes the "soft" IRT fits
```

### 2.1 `pipeline/` (training and building)

| File | What it does | Main output |
|---|---|---|
| `irt_build_data.py` | Builds the IRT response tables from the downloaded raw data. Normalises subject fields. Makes the 5 folds. | `data/irt/cells_binary.parquet`, `cells_cont.parquet`, `subjects.parquet`, `folds.csv` |
| `irt_fit.py` | Fits the "version B" IRT model with Pyro SVI. 6 fits: `fold0`..`fold4` and `final`. | `data/irt/<run>/<fit>/` with `items.parquet` (b), `theta_<part>.parquet`, `zeta.parquet`, `hyper.json`, `losses.csv`, `DONE` |
| `build_embed_manifest.py` | Chooses up to 5,000 items per binary benchmark and builds the AP prompt text for each. | `data/embed_manifest/binary_cap5000.parquet` |
| `ap_embeddings.py` | AP's embedding code, copied verbatim (MIT licence). | (library) |
| `embed_manifest.py` | Embeds every manifest item with Qwen3-VL-8B, one `.npz` per benchmark, resumable. `--parity N` checks against AP's own function. | `data/emb/qwen3vl8b/<benchmark>.npz` |
| `modal_embed.py` | Runs `embed_manifest.py` on a Modal A100 and downloads the results. Modes: `parity`, `pilot`, `full`, `download`, `status`. | Modal volume `paiec-emb`, then `data/emb/qwen3vl8b/` |
| `run_embed.sh`, `requirements-gpu.txt` | The same on any GPU machine without Modal. | |
| `fast_ridge.py` | Exact ridge regression for many penalties and folds at once (same answer as sklearn, much faster). Adds the `r2_within` score. | (library) |
| `ridge_train.py` | Trains the embedding → difficulty ridge for each fit. `--score r2` (default) or `r2_within`; `--out_dir`. | `data/ridge/<run>/<fit>/model.npz`, `pred.parquet`, `summary.json`, `DONE` |
| `ridge_sklearn.py` | The same fits with scikit-learn, as a check. | `data/ridge_sklearn/...` |
| `stretch_factor.py` | Computes the stretch factor `k` for "stretch 2". | `data/ridge/<run>/stretch.json` |
| `name_date_prior.py` | Fits the release-date line for unseen names. `--fold final` (default) or `fold0`..`fold4`. | `data/irt/<run>/<fold>/name_date_prior.json` |
| `build_submission.py` | Assembles a submission folder and zip from `submission_src/` and the trained files. | `submissions/<name>/` and `submissions/<name>.zip` |
| `modal_check_submission.py` | Runs a built submission on a Modal A100 with the hub offline, like the platform, and checks it. | printed report |

### 2.2 `analysis/` (data, tests, simulation)

| File | What it does |
|---|---|
| `fetch_small_tables.py`, `fetch_responses.py`, `fetch_items_binary.py` | Download tables from the Hugging Face dataset `aims-foundations/measurement-db` into `data/measurement-db/` |
| `summarize_db.py` | Summary of response scales and subject counts |
| `clean_benchmarks.py` | Keeps benchmarks that are "higher is better", have known bounds, are not pairwise win-rates, and whose responses lie inside the declared scale → `clean_benchmarks.csv`, `excluded_benchmarks.csv` |
| `select_training_benchmarks.py` | Clean benchmarks with at least 5 subjects → `train_benchmarks.csv` (152 benchmarks) |
| `subject_fields.py`, `subject_overlap.py` | Explore subject fields; check how often a subject of a held-out benchmark is known from other benchmarks |
| `item_text_sizes.py`, `token_lengths.py`, `embed_cost_table.py`, `embed_cost_train.py` | Estimate token counts and GPU cost of embedding |
| `ridge_vs_sklearn.py` | Proves `fast_ridge.py` = sklearn (same chosen penalty, predictions differ by at most 4e-6) |
| `simulate_baselines.py` | An early simulator of the streaming protocol for simple baselines |
| `simulate_alc.py` | **The main local test** (section 8) |
| `test_label_update.py` | Logic tests of a built label submission (section 7) |
| `test_name_date_prior.py` | Tests of the release-date rule in a built submission |
| `simulate_alc_*.csv` | Results of every simulation run (one row per draw × pair × configuration) |
| `*.parquet`, `*.txt`, `*.csv` | Small tables made by the scripts above (benchmark lists, subject lists, token lengths) |

### 2.3 `submissions/`

One folder per built submission. Each has a `README.md` that explains that submission in detail, the `.zip` that is uploaded, and `files/` with the same files unzipped.

> Note: `build_submission.py` writes to `submissions/<name>/` and `submissions/<name>.zip` directly. In this repository we moved each zip into its folder and the files into `files/` so that the README fits next to them. A rebuild will produce the flat layout again; that is expected.

---

## 3. Results of every method

### 3.1 Local results (our 5-fold "80/20" simulation)

All numbers below come from `analysis/simulate_alc.py`: 5 folds × 30 draws = 150 draws, **1,111 subject–benchmark pairs**, the **same draws, splits and label outcomes for every method**. Lower is better. "ALC" is the mean over draws of the mean ALC of the draw's pairs (the per-budget columns are means over pairs, so the two can differ slightly in the last digit).

**Main table (Brier ALC):**

| Method | Original ridge | Stretch 1 ridge | Stretch 2 |
|---|---|---|---|
| ap_pure (labels not used) | 0.2588 | 0.2584 | 0.2639 |
| label_update | 0.2246 | 0.2250 | 0.2275 |
| + date rule only | 0.2249 | 0.2254 | 0.2273 |
| + item shifts only | 0.2241 | 0.2243 | 0.2269 |
| + averaging only | 0.2236 | 0.2213 | 0.2236 |
| date + item shifts | – | 0.2243 | 0.2268 |
| date + averaging | – | 0.2212 | 0.2233 |
| item shifts + averaging | – | 0.2209 | 0.2233 |
| all 3, sigma_item 0.6× (stretch: not tuned further) | 0.2227 | 0.2209 | 0.2229 |
| all 3, sigma_item 1.0× (untuned) | 0.2235 | – | – |
| **all 3, tuned (sigma_item 0.6×, sigma_delta 2×)** | **0.2214** | **0.2196** | 0.2214 |

Note on this table: in the "Original ridge" column, the single-switch rows (+date, +item, +avg) used the measured sigma_item = 1.624. In the two stretch columns, every row used 0.6 × the measured sigma_item. sigma_item only matters for item shifts and averaging. "–" means not run.

**Per budget:**

| Method | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| ap_pure | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| label_update | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| label_update_3improvements | 0.2525 | 0.2403 | 0.2271 | 0.2151 | 0.2059 | 0.1996 | 0.2227 |
| label_update_3improvements_tuned | 0.2517 | 0.2382 | 0.2247 | 0.2144 | 0.2051 | 0.1994 | 0.2214 |
| same, stretch 1 ridge | 0.2489 | 0.2359 | 0.2232 | 0.2127 | 0.2041 | 0.1976 | **0.2196** |
| same, stretch 2 | 0.2517 | 0.2367 | 0.2256 | 0.2144 | 0.2058 | 0.2000 | 0.2214 |

**Difference to `label_update_3improvements_tuned` (matched draws, ± standard error over the 150 draws, folds where the method is better):**

* stretch 1 tuned: −0.0018 ± 0.0005, better on 4 of 5 folds.
* stretch 1, all 3 with sigma_delta 1×: −0.0006 ± 0.0005, 3 of 5 folds.
* stretch 2 tuned: −0.0000 ± 0.0006, 3 of 5 folds.
* label_update_3improvements: +0.0013 ± 0.0003, 0 of 5 folds.
* label_update: +0.0032 ± 0.0005, 0 of 5 folds.
* ap_pure: +0.0374 ± 0.0024.

**Which CSV has which result:**

| File | Submission loaded | Configurations | Settings |
|---|---|---|---|
| `simulate_alc_label_update_3improvements.csv` | label_update_3improvements | ap_pure, label_update, +date, +item, +avg, all3 | sigma_item 1.0× (1.624) |
| `..._scale<s>.csv` | same | all3 | sigma_item × s |
| `..._final.csv` | same | all3 | the zip as built (sigma_item 0.6×) |
| `..._z<a>_d<b>.csv` | same | all3 | sigma_zeta × a, sigma_delta × b |
| `..._tuned_final.csv` | label_update_3improvements_tuned | all3 | the tuned zip as built |
| `..._pairs.csv` | same | | an earlier run used to check the draw sizes |
| `simulate_alc_sim_stretch1.csv`, `..._sim_stretch2.csv` | test builds with the stretch ridges (current code, sigma_item 0.6×) | all 9 configurations | sigma_delta 1× |
| `simulate_alc_sim_stretch1_d2.csv`, `..._sim_stretch2_d2.csv` | same | all3 | sigma_delta 2× (the "tuned" rows) |

Note: the rows for the older zips (`label_update`, `*_date`, `*_stretch*`) were simulated with the **current** code with the matching switches. With the switches off, the current code gives exactly the old predictions (tested). The only small difference is the `*_date` zips: their older `core.py` did not add the extra prior variance for unseen names.

### 3.2 Codabench results

| Submission | Codabench Brier ALC | Pairs | Notes |
|---|---|---|---|
| ap_pure | ≈ 0.217 | 9 | Same value at every budget, as expected |
| label_update | ≈ 0.178 | 8 | Big gains on pairs that were badly off at budget 0 (e.g. 0.640 → 0.056, 0.241 → 0.004) |
| ap_pure_stretch1, ap_pure_stretch2, label_update_stretch1, label_update_stretch2 | pending | | Uploaded; results not back when Codabench went down |
| label_update_3improvements(_tuned), *_date | not uploaded / unknown | | |

Each Codabench upload is scored on its own random draw of only 8 or 9 pairs, so single uploads are noisy and cannot be compared closely. The ranking agrees with the local test (label_update beats ap_pure). **Never tune anything on Codabench scores.**

### 3.3 What the results teach us

1. **Using labels is by far the biggest gain** (ap_pure → label_update: −0.034).
2. **Averaging over uncertainty** helps most at budgets 0–3 (budget 0 improves by about 0.009).
3. **Item shifts** help most at budgets 15–31 (budget 31 improves by about 0.005).
4. **The date rule** shows no effect locally, because the held-out folds remove benchmarks, not models, so almost every held-out subject's name is known. It can only help on Codabench if a test subject is a new model.
5. **sigma_item** measured (1.624) is too big; 0.6× is best and was chosen in every leave-one-fold-out test.
6. **sigma_delta** measured (0.456) is too small; 2× is best (flat beyond).
7. **sigma_zeta** from the IRT fit (1.110) is right.
8. **Stretching** alone does not help; stretch 2 hurts. The stretch 1 ridge helps only with averaging.
9. **Where the remaining error is:** at budget 0 it is mostly the overall level of a pair; at high budgets it is the ranking of items within a benchmark. With only the level known, a pair cannot score below a × (1 − a), where a is its share of correct answers (0.25 at a = 0.5, 0.21 at a = 0.7). Our ridge ranks items poorly (within-benchmark correlation of predicted and true difficulty only about 0.10–0.17). **Better item difficulty is the biggest remaining opportunity.**

---

## 4. The method, step by step

All submissions use the two-stage method of the **Agent Psychometrics (AP)** paper (Ge, Kryvosheieva, Fried, Girit, Hariharan, COLM 2026; code: `https://github.com/dariakryvosheieva/agent-psychometrics`, commit `1c34d34a0ee82c92b1c63deb834495c5440bd362`, MIT licence) with our changes.

### 4.1 Training data

* Source: Hugging Face dataset **`aims-foundations/measurement-db`** (one folder per benchmark with `formatted_tables/items.parquet`, `responses.parquet`, `subjects.parquet`, `benchmarks.parquet`). Downloaded with the `analysis/fetch_*.py` scripts into `data/measurement-db/` (3.8 GB, not in this repo).
* Benchmark selection (`analysis/clean_benchmarks.py` → `analysis/select_training_benchmarks.py`): direction "higher is better"; known bounds that map to [0, 1]; not a pairwise win-rate; responses inside the declared scale; at least 5 subjects. Result: **152 benchmarks** in `analysis/train_benchmarks.csv`: **115 binary** (0/1) and **37 non-binary** (14 bounded continuous, 8 continuous, 5 five-point Likert, 5 ordinal, 4 fraction, 1 ten-point Likert).
* A **cell** = one subject on one item. Repeated attempts are kept in the same cell as counts.

| | Binary | Non-binary | Total |
|---|---|---|---|
| Benchmarks | 115 | 37 | 152 |
| Items | 1,939,246 | 424,058 | 2,363,304 |
| Subject–benchmark pairs | 9,893 | 3,655 | 13,548 |
| Cells | 24,998,450 | 3,553,143 | 28,551,593 |
| Responses | 58,188,963 | 4,397,434 | 62,586,397 |

* Subject table: 13,658 subject–benchmark rows (including rows without responses), 638 public model names, 160 harnesses, 84 providers, 8 reasoning-effort values (blank included). 9,974 rows have a blank name.
* 8,426 binary pairs have at least 80 distinct items (the size of a test pair).
* **Folds:** the 115 binary benchmarks are split at random (seed 0) into **5 folds of 23** (`data/irt/folds.csv`). Fold *f*'s IRT fit and ridge never see fold *f*'s benchmarks. Non-binary benchmarks are always in training (never held out).

### 4.2 Stage 1: the IRT model ("version B")

Code: `pipeline/irt_build_data.py`, `pipeline/irt_fit.py`. Based on AP's `swebench_irt/train_model_scaffold_shared.py`.

**Model (1PL / Rasch):**

```
mu(subject, item) = theta_name + theta_harness + theta_provider + theta_effort + zeta(pair) - b(item)
binary:      outcome ~ Bernoulli(sigmoid(mu))           (repeated attempts grouped as Binomial: same likelihood)
non-binary:  logit(y) ~ Normal(mu, sigma_m^2)           y scaled to [0,1], clipped to [0.01, 0.99]; one sigma_m per benchmark
```

**Subject field normalisation** (identical in `irt_build_data.py` and `core.py`; any change must be made in both):

* `name` = `normalized_name`, stripped. A blank name in training gets a private id `"<benchmark>::<subject_id>"` that never matches anything (these private names are removed from `theta.json`).
* `harness` = lower case, runs of spaces/underscores → `-`, a trailing version such as `-v1.2` or `-1.2.3` removed, then `openai-codex` and `codex-cli` → `codex`.
* `provider` = lower case, whitespace collapsed; `unknown` → blank.
* `effort` = `reasoning_effort`, lower case, whitespace collapsed.
* A blank part adds 0. A value never seen in training adds 0 (except unseen names with the date rule, section 5.1).
* `zeta` is per (benchmark, subject_id) pair: the per-benchmark ability deviation, as in the LEGO-MB model (with independent priors instead of LEGO's LKJ covariance, because new benchmarks have no learned correlations).

**Priors:** every theta part ~ Normal(0, sigma_part²); zeta ~ Normal(0, sigma_zeta²); b ~ Normal(0, sigma_b²); the sigmas have HalfNormal(1) priors. Abilities are mean-centred.

**Fitting:** Pyro SVI exactly as AP: Trace_ELBO, ClippedAdam (lr 0.01, betas 0.9/0.999, clip_norm 5), full batch, 2000 epochs, seed 0. Six fits (`--fold all`): the final fit took about 23 minutes on CPU.

**Learned spreads (final fit, `data/irt/versionB_4parts_logitnormal/final/hyper.json`):**

| sigma_name | sigma_harness | sigma_provider | sigma_effort | sigma_zeta | sigma_b |
|---|---|---|---|---|---|
| 0.377 | 0.647 | 0.455 | 0.528 | **1.110** | 1.811 |

The final fit has 626 public names, 157 harnesses, 82 providers, 7 effort levels.

A second variant, `versionB_4parts_soft` (`--cont soft`: a "soft" Bernoulli likelihood for non-binary scores), was also fitted with its ridges. **The submissions use `versionB_4parts_logitnormal`.** The soft run is not needed.

### 4.3 Item embeddings

Code: `pipeline/build_embed_manifest.py`, `pipeline/embed_manifest.py`, `pipeline/modal_embed.py`, `pipeline/ap_embeddings.py`.

* Model: **`Qwen/Qwen3-VL-8B-Instruct`**, bfloat16, loaded with `AutoModelForImageTextToText`, text sub-model selected with AP's `_select_text_submodel`.
* Text: AP's "no solution" template:

  ```
  Task statement:
  {item_content}

  How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:
  ```

* Vector: last layer, **last token** hidden state, **4096 numbers**. Max 8192 tokens, truncation from the **left**, batch size 1, seed 0, deterministic algorithms (AP's defaults).
* Items: up to **5,000 per binary benchmark**, drawn at random (RNG seeded by crc32 of the benchmark name) from items with at least one response. **255,286 items** in total; 33 benchmarks were capped, 82 embedded fully.
* Ran on a Modal **A100-40GB**. Output: `data/emb/qwen3vl8b/<benchmark>.npz` (AP's cache format: `task_ids`, `X` float32, metadata). 1.9 GB, not in this repo.
* `--parity` mode confirmed our batched path gives the same vectors as AP's unmodified `embed_items()`.

### 4.4 Stage 2: the ridge regression (embedding → difficulty)

Code: `pipeline/ridge_train.py` with `pipeline/fast_ridge.py`.

```
b_hat = ((x - mean) / scale) . coef + intercept
```

* Training rows: embedded items of the binary benchmarks the IRT fit was trained on. Target: that fit's `b`. Every item weighs the same.
* `mean`, `scale`: StandardScaler. Penalty `lambda` chosen by **inner 5-fold cross-validation grouped by benchmark**.
* Grid: AP's (0.01 … 1e5) plus 1e6 and 1e7, extended ×10 automatically if the best value is at the top edge (up to 1e12).
* Selection score: `r2` (sklearn's default R², as AP) for the original ridge `data/ridge`; `r2_within` (R² after removing each benchmark's mean) for the stretch 1 ridge `data/ridge_within`.
* `fast_ridge.py` builds per-fold sums once and solves every penalty with one eigendecomposition: `w(lambda) = V diag(1/(e + lambda)) V' c`. Checked against sklearn: same penalty chosen in all 12 fits, predictions differ by at most 4e-6.

| Ridge | Folder | Chosen lambda (final) | Use |
|---|---|---|---|
| Original | `data/ridge/versionB_4parts_logitnormal/` | **1e7** | ap_pure, label_update, 3improvements(_tuned) |
| Stretch 1 | `data/ridge_within/versionB_4parts_logitnormal/` | **1e5** (fold 3: 1e6) | *_stretch1 |

Quality of the original ridge on held-out benchmarks: correlation with true difficulty about 0.06–0.14 overall and **0.10–0.17 within a benchmark**. Its predictions have a spread of only 0.10, while the true spread is about 1.6 (strong shrinkage). Within-benchmark MSE: 2.815 (original) vs 2.791 (stretch 1).

`pred.parquet` (columns `item`, `benchmark`, `heldout`, `b_pred`) has the prediction for **every** embedded item; `heldout = True` marks the items of the benchmarks that fit did not train on. These held-out predictions drive the simulation and the priors.

### 4.5 Stretch 2

`pipeline/stretch_factor.py`: `b_stretched = c + k (b_hat − c)` with `c` = ridge intercept = **0.01462**, `k` = **4.0502** (within-benchmark slope of true b on b_hat, pooled over the held-out items of the 5 folds). Also writes `sigma_delta_stretched` = **0.5695**. Saved in `data/ridge/versionB_4parts_logitnormal/stretch.json`.

### 4.6 Priors for the label update

Computed by `build_submission.py` (`shift_priors()`), saved in each zip's `shift_priors.json`:

* **sigma_delta** = RMS over the 115 held-out benchmarks of (mean final-fit b − mean held-out b_hat): **0.456** (original), **0.587** (stretch 1), **0.570** (stretch 2).
* **sigma_zeta** = from the IRT fit: **1.110**.
* **sigma_item** = RMS over the held-out items of the within-benchmark error (b − k·b_hat, both centred within the benchmark): **1.624** (original), 1.618 (stretch 1), 1.629 (stretch 2). The zips use 0.6× this.

### 4.7 Release-date prior for unseen names

`pipeline/name_date_prior.py`: OLS of `theta_name` on `years = (release_date − 2024-01-01)/365.25`, clipped to [−1, 2], over the named training subjects with a readable date (median date per name).

Final fit: **a = −0.15382, b = 0.21716**, 595 names, leave-one-name-out RMSE **0.4850** (line) vs **0.5161** (predict 0). Fold fits exist for folds 0–4 (`data/irt/<run>/fold<f>/name_date_prior.json`); the line beats 0 in every fold.

Date parsing (`release_years` = `core._release_years`): regex `\s*(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?!\d)`. `YYYY-MM-DD` as is; `YYYY-MM` → the 15th; `YYYY` → 1 July; invalid dates → None.

---

## 5. The submission code in detail

Source: `submission_src/`. `build_submission.py` copies `common/` plus the variant folder and adds the trained files.

### 5.1 `common/core.py` (shared)

Loaded at import:

* `theta.json` → `THETA = {"name": {...}, "harness": {...}, "provider": {...}, "effort": {...}}`.
* `ridge.npz` → `RIDGE_MEAN`, `RIDGE_SCALE`, `RIDGE_COEF`, `RIDGE_INTERCEPT`.
* `bhat_stretch.json` (optional) → `STRETCH_CENTER`, `STRETCH_K` (default: intercept and 1.0, i.e. no stretch).
* `name_date_prior.json` (optional) → `NAME_PRIOR` (default None, i.e. no date rule).
* `SMOKE_TEST` = environment variable `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST` is 1/true/yes/on (set by the starting-kit tools).

Functions:

| Function | What it does |
|---|---|
| `subject_theta(subject)` | Normalises the four fields as in training, sums the theta parts. If the name is unseen and `NAME_PRIOR` is loaded and the date parses: adds `a + b * clip(years)`. |
| `subject_theta_var(subject)` | Extra prior variance for an unseen name: `loo_rmse_line²` if dated, else `loo_rmse_zero²`; 0 for known names or without the date file. |
| `_release_years(s)` | Date string → years since 2024-01-01, or None. |
| `load_model()` | Raises if no CUDA GPU. Seeds (AP's `seed_everything(0, deterministic=True)`). Loads tokenizer and `AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=bfloat16, cache_dir=..., local_files_only=True, device_map="auto")`. Padding right, truncation left. Picks the text sub-model, turns off `use_cache`. |
| `_hf_cache_dir()` | `HF_HOME`, else `/app/hf_cache`, else `./.hf_cache` (copied from the kit's `hf_submission` template). |
| `_embed(text)` | Tokenises (max 8192, left truncation), runs the text model in `inference_mode`, takes the last layer and the last real token (AP's `_extract_hidden_state`, `last_token_pool`). |
| `item_b_hat(item)` | In smoke-test mode returns the intercept. Otherwise: builds the AP text, checks the disk cache `bhat_cache/<sha256>.txt`, else embeds, checks shape (4096) and finiteness, applies the ridge and the stretch, writes the cache (temporary name, then atomic `os.replace`). |
| `sigmoid(z)` | Numerically safe sigmoid. |

### 5.2 `common/ap_embed_core.py`

The AP functions the submission needs, copied verbatim (MIT licence text included): `DIFFICULTY_INSTRUCTION`, `seed_everything`, `set_torch_determinism`, `_sanitize_text`, `format_qs_instruction`, `_select_text_submodel`, `_get_hidden_states_tuple`, `_extract_hidden_state`, `last_token_pool`. The model loaders are left out, because `core.py` must use the literal repo ID.

### 5.3 `common/models.txt`

One line: `Qwen/Qwen3-VL-8B-Instruct`.

### 5.4 `ap_pure/model.py`

```python
if os.environ.get("PAIEC_PREDICTION_WORKER_ID") is not None and not core.SMOKE_TEST:
    core.load_model()            # only workers need the model
def predict(input, labeled=None):
    subject, item = input
    return core.sigmoid(core.subject_theta(subject) - core.item_b_hat(item))
```

### 5.5 `label_update/model.py` (version 2, current)

```
P(correct) = sigmoid( (theta + zeta) - (b_hat + delta + epsilon) )
```

* Reads `shift_priors.json`: `SIGMA_DELTA`, `SIGMA_ZETA`, `SIGMA_ITEM` (default 0), `ITEM_SHIFTS` (default False), `AVERAGE` (default False).
* Loads the model at import in **every** process unless in smoke-test mode (the parent needs predictions for `labeling.py`).
* `fit_shifts(base, y, A, prec)`: MAP of `w` for `y_k ~ Bernoulli(sigmoid(base_k + A_k · w))`, prior `w ~ N(0, diag(1/prec))`. Newton's method, up to 100 iterations, with backtracking (halve the step until the objective does not increase; stop when the change is below 1e-12 and the step below 1e-10). Returns `w` and the Laplace covariance `inv(Hessian)`, where `Hessian = Aᵀ diag(p(1−p)) A + diag(prec)`.
* `_benchmark_fit(benchmark_id, labeled)`: takes the labels whose item has this `benchmark_id`. Parameters: column 0 = `delta` (A = −1); one column per labelled subject = `zeta` (A = +1, prior variance `SIGMA_ZETA² + subject_theta_var`); if `ITEM_SHIFTS`, one column per labelled item = `epsilon` (A = −1, prior variance `SIGMA_ITEM²`). `base = theta − b_hat` for each label. Subjects and items are keyed by `json.dumps(dict, sort_keys=True)` (all fields). Results are memoised in `_SHIFT_MEMO` by (benchmark, ITEM_SHIFTS, labels).
* `predict(input, labeled)`: builds the vector `a` (−1 for delta; +1 for this subject's zeta if it has labels; −1 for this item's epsilon if it has labels).
  * `mean = theta − b_hat + a · w`.
  * Without averaging: `sigmoid(mean)`.
  * With averaging: `var = aᵀ Cov a` + (subject prior variance if the subject has no labels on this benchmark) + (`SIGMA_ITEM²` if the item has no labels); return `sigmoid(mean / sqrt(1 + π var / 8))`.
* With both switches off and no date file, the predictions equal the old `label_update` (version 1, only delta and zeta), and with no labels they equal `ap_pure`.

### 5.6 `label_update/labeling.py` (Fisher information rule)

```python
def acquisition_function(input, prediction=None, labeled=None, context=None):
    if context is None:                       # old evaluators: numeric priority
        return 0.25 if prediction is None else prediction * (1 - prediction)
    if prediction is None: raise ValueError(...)
    if context["labels_remaining"] <= 0: return False
    r = context["labels_remaining"] / max(context["items_remaining"], 1)
    if r >= 1: return True
    return prediction * (1 - prediction) >= (1 - r*r) / 4     # same as |p - 0.5| <= r/2
```

Reasoning: a label with predicted probability p carries Fisher information p(1−p) about the shifts (largest at p = 0.5). With predictions spread evenly, the rule buys about a share r of candidates, so the budget gets used while informative items are preferred. **It has not yet been compared with random buying** (an open idea).

### 5.7 Files `build_submission.py` writes

| File | Content |
|---|---|
| `theta.json` | theta tables of the final IRT fit, without `private::` names |
| `ridge.npz` | `mean`, `scale`, `coef`, `intercept`, `alpha` from `<ridge_dir>/<run>/final/model.npz` |
| `shift_priors.json` | sigma_delta, sigma_zeta, sigma_item (+ measured value and scale), switches, `ridge_dir`, sources |
| `bhat_stretch.json` | only with `--stretch` |
| `name_date_prior.json` | only with `--name_prior` (a, b, years_min, years_max, loo_rmse_line, loo_rmse_zero) |

Options:

```
python pipeline/build_submission.py --variant ap_pure|label_update [--name <output name>]
    [--run versionB_4parts_logitnormal]
    [--ridge_dir data/ridge | data/ridge_within]
    [--stretch data/ridge/versionB_4parts_logitnormal/stretch.json]
    [--name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json]
    [--item_shifts] [--average]
    [--sigma_item_scale 0.6] [--sigma_delta_scale 2.0]
```

---

## 6. Every submission

Each folder in `submissions/` has its own detailed README. Code versions are given by the first 8 characters of the MD5 hash of the file.

| Submission | model.py | core.py | Ridge | Settings | Local ALC | Codabench |
|---|---|---|---|---|---|---|
| `ap_pure` | ap_pure `933260da` | `16acf1e4` (v1) | original | – | 0.2588 | ≈ 0.217 |
| `ap_pure_stretch1` | `933260da` | `16acf1e4` | stretch 1 | – | 0.2584 | pending |
| `ap_pure_stretch2` | `933260da` | `16acf1e4` | original + k = 4.05 | – | 0.2639 | pending |
| `label_update` | v1 `fa60706c` | `16acf1e4` | original | sd 0.456, sz 1.110 | 0.2246 | ≈ 0.178 |
| `label_update_stretch1` | v1 | `16acf1e4` | stretch 1 | sd 0.587 | 0.2250 | pending |
| `label_update_stretch2` | v1 | `16acf1e4` | original + k | sd 0.570 | 0.2275 | pending |
| `label_update_date` | v1 | `17dd3912` (date rule) | original | sd 0.456 + date | 0.2249 | – |
| `label_update_stretch1_date` | v1 | `17dd3912` | stretch 1 | sd 0.587 + date | 0.2254 | – |
| `label_update_stretch2_date` | v1 | `17dd3912` | original + k | sd 0.570 + date | 0.2273 | – |
| `label_update_3improvements` | v2 `788a51b0` | `bb307981` (current) | original | sd 0.456, si 0.975, item, avg, date | 0.2227 | – |
| **`label_update_3improvements_tuned`** | v2 | `bb307981` | original | **sd 0.912**, si 0.975, item, avg, date | **0.2214** | – |

(sd = sigma_delta, sz = sigma_zeta, si = sigma_item; sigma_zeta is 1.110 everywhere; `labeling.py` is `bdfe8b2d` in every label submission.)

The current source in `submission_src/` equals `ap_pure` `933260da`, v2 `788a51b0`, core `bb307981`, labeling `bdfe8b2d`.

---

## 7. Testing

Every zip must pass all of these before upload.

1. **Logic tests (CPU, seconds):**

   ```
   python analysis/test_label_update.py <submission folder name>
   python analysis/test_name_date_prior.py <submission folder name>     # if it has the date rule
   ```

   `test_label_update.py` (needs the `ap_pure` folder too) checks: Newton fit = scipy BFGS with exact gradient (max diff < 1e-4; L-BFGS was not used because it stopped early); covariance × Hessian = identity (< 1e-8); item shifts recover the truth with 200 labels per cell; switches off + no labels = `ap_pure`; labels of another benchmark change nothing; correct labels raise p; another subject's wrong answer lowers only that item; averaging pulls toward 0.5, never crosses it, and fades with labels; the hook buys/skips correctly.

   `test_name_date_prior.py` checks that all 363 different training date strings parse the same in the submission and in the fitting script, edge cases, and that known names are unchanged.

2. **Starting kit tools:**

   ```
   python ../starting_kit/tools/check_submission_zip.py submissions/<name>.zip
   python ../starting_kit/tools/run_smoke_test.py submissions/<name>
   ```

   In smoke-test mode no model is loaded and b_hat is the ridge intercept.

3. **GPU end-to-end check on Modal (A100-40GB, hub offline):**

   ```
   modal run pipeline/modal_check_submission.py --variant <name>
   ```

   Needs the Modal volume `paiec-hf-cache` with the Qwen model and `data/submission_check_items.parquet` (36 real training items with one real subject per benchmark; per-ridge expected values in `data/submission_check_items_<variant>.parquet`). It checks that the model loads, b_hat matches the stored training prediction (< 0.01; in practice 1e-15 to 1e-17), p matches the formula (< 0.005; in practice 1e-16; with averaging the expected p includes the averaging), the second pass comes from the cache, and labels move p the right way. Prints `SUBMISSION PIPELINE OK`. On the A100, loading takes 20–30 s and a new item about 0.16 s.

4. **Local simulation** (section 8) on the exact built zip, to confirm the expected score.

---

## 8. The local simulation (our "80/20" test)

Code: `analysis/simulate_alc.py`. This is what the organizers mean by testing locally on an 80/20 split of the training set. We do it 5 times (5 folds), so each benchmark is tested once, and we can check that a change wins on every fold.

**How it works:**

* For fold *f*: the 23 benchmarks of fold *f* are the "test" benchmarks. **theta** comes from fold *f*'s IRT fit (trained without them), **b_hat** from fold *f*'s ridge (`pred.parquet`, `heldout` rows), the **name date prior** from fold *f*'s `name_date_prior.json`, and the **outcomes** are the real recorded responses (`data/irt/cells_binary.parquet`), restricted to embedded items.
* The submission's own `model.py` and `labeling.py` are imported and run (smoke-test import). Only `core.item_b_hat` is replaced by a lookup of the stored held-out b_hat, and `core.THETA` / `core.NAME_PRIOR` by the fold's values. If the submission has `bhat_stretch.json`, the stretch is applied around the fold ridge's intercept.
* **One draw:** pick 7 benchmarks of the fold, keep 150 random items of each (same items for all subjects), then add (benchmark, subject) pairs with at least 80 items greedily up to 1,000 distinct items, preferring benchmarks and subjects represented less often (random tie-break). This mimics the about 8 pairs we see on Codabench.
* **Each pair:** random 50/50 split (extra item to acquisition), candidates in random order; a revealed label is one recorded run (1 with probability k/n).
* **Budgets 0, 1, 3, 7, 15, 31:** before each cap, every pair continues its candidate stream until it has `cap` labels or runs out. The hook sees the prediction with all labels so far and the same `context` as on the platform. Then Brier over all runs of the evaluation items.
* 30 draws per fold → 150 draws, 1,111 pairs. Seeds `[seed, fold]` make the draws identical for every submission and configuration, so differences are fair (matched).
* Configurations (`CONFIGS`): tuples of (use labels, date rule, item shifts, averaging): `ap_pure`, `label_update`, `+date`, `+item`, `+avg`, `date+item`, `date+avg`, `item+avg`, `all3`.

**Usage:**

```
python analysis/simulate_alc.py --sub <submission folder> --draws 30 [--configs all3,label_update] [--tag _name]
      [--sigma_item_scale s] [--sigma_zeta_scale s] [--sigma_delta_scale s] [--n_bench 7] [--bench_items 150] [--seed 0]
```

One run of all configurations takes about 2–3 minutes per configuration on a laptop CPU. Output: `analysis/simulate_alc_<sub><tag>.csv` with columns `fold, draw, pair, config, alc, b0, b1, b3, b7, b15, b31`. When `label_update` is among the configurations, it prints a table with the difference to `label_update` and its standard error.

**How to compare two methods properly:** compute the mean ALC per (fold, draw) for each, take the difference draw by draw, report the mean ± std/√150, and count on how many folds the mean is lower. **Tuning rule:** choose a setting on 4 folds, score it on the 5th, and repeat for all 5.

**Limitation:** the simulated benchmarks are training benchmarks, not the real test benchmarks, and almost all subject names are known. Absolute numbers differ from Codabench. Use it to compare versions.

---

## 9. How to rebuild everything from scratch

Environment used: Windows 11, Python 3.12.10, numpy 2.4.3, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.8.0, pyarrow 25.0.1, torch 2.11.0 (CPU) and pyro-ppl 1.9.2 for the IRT, modal 1.5.3. The Modal GPU image uses torch 2.11.0, torchvision 0.26.0, transformers 5.12.1, accelerate 1.15.0 (see `pipeline/modal_check_submission.py`). For Qwen3-VL, transformers must be ≥ 4.57.

```
# 1. data (about 3.8 GB; public dataset, no token needed)
python analysis/fetch_small_tables.py
python analysis/fetch_responses.py
python analysis/clean_benchmarks.py
python analysis/select_training_benchmarks.py           # -> analysis/train_benchmarks.csv
python analysis/fetch_items_binary.py

# 2. IRT (CPU; about 25 min per fit, 6 fits; resumable)
python pipeline/irt_build_data.py
python pipeline/irt_fit.py --run versionB_4parts_logitnormal --cont logitnormal --fold all

# 3. embeddings (GPU; Modal A100-40GB)
python pipeline/build_embed_manifest.py --cap 5000 --tokenizer Qwen/Qwen3-VL-8B-Instruct
modal run pipeline/modal_embed.py --model 8b --mode parity
modal run --detach pipeline/modal_embed.py --model 8b --mode full
modal run pipeline/modal_embed.py --model 8b --mode download          # -> data/emb/qwen3vl8b/

# 4. ridges (CPU; about 10-15 min each with fast_ridge)
python pipeline/ridge_train.py                                          # original, -> data/ridge
python pipeline/ridge_train.py --score r2_within --out_dir data/ridge_within   # stretch 1
python pipeline/stretch_factor.py                                       # stretch 2
python pipeline/ridge_sklearn.py                                        # optional check (about 2 h)

# 5. date prior (final and each fold; the simulation needs the fold files)
python pipeline/name_date_prior.py --fold final
for f in 0 1 2 3 4; do python pipeline/name_date_prior.py --fold fold$f; done

# 6. build + test the recommended submission
python pipeline/build_submission.py --variant label_update --name label_update_3improvements_tuned \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
    --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0
python analysis/test_label_update.py label_update_3improvements_tuned
python analysis/test_name_date_prior.py label_update_3improvements_tuned
python ../starting_kit/tools/check_submission_zip.py submissions/label_update_3improvements_tuned.zip
python ../starting_kit/tools/run_smoke_test.py submissions/label_update_3improvements_tuned
modal run pipeline/modal_check_submission.py --variant label_update_3improvements_tuned
python analysis/simulate_alc.py --sub label_update_3improvements_tuned --draws 30 --configs all3 --tag _final
```

Modal: the volumes `paiec-hf-cache` (the Qwen model cache) and `paiec-emb` (embeddings) live in Guneesh's Modal account. A new person needs their own Modal account and must create them (the first `modal_embed.py` run downloads the model into the cache volume).

---

## 10. What is not in this repository

GitHub cannot hold the large files. Everything missing can be rebuilt with section 9.

| Missing | Size | Needed for | Rebuild with |
|---|---|---|---|
| `data/measurement-db/` (raw dataset) | 3.8 GB | everything upstream | `analysis/fetch_*.py` |
| `data/emb/qwen3vl8b/` (embeddings) | 1.9 GB | ridge training | step 3 (GPU) |
| `data/embed_manifest/binary_cap5000.parquet` | 0.5 GB | embeddings | `build_embed_manifest.py` |
| `data/irt/cells_binary.parquet`, `cells_cont.parquet` | 0.3 GB | IRT fits, **the simulation** | `irt_build_data.py` |
| `data/irt/<run>/<fit>/items.parquet` (item b) | 60 MB each | `build_submission.py` (shift priors), ridge training | `irt_fit.py` |
| `data/irt/versionB_4parts_soft/` details, `data/ridge_sklearn/`, soft ridges | | not needed | |
| Two Modal logs (`download_8b.log`, `modal_full_8b.log`) | | not needed | contain an account e-mail |
| `SUBMISSION_1.md` | | not to be used | |
| `bhat_cache/` folders | | runtime cache only | |
| Test builds `submissions/sim_stretch1`, `sim_stretch2` | | only for the stretch simulation | build commands in section 12.1 (same options with `--name sim_stretch1` / `sim_stretch2`) |

**What is included in `data/`:** `irt/folds.csv`, `irt/subjects.parquet`, every fit's `theta_*.parquet`, `zeta.parquet`, `hyper.json`, `losses.csv`, `name_date_prior.json`; every ridge fit's `model.npz`, `pred.parquet`, `summary.json` (original and stretch 1); `ridge/.../stretch.json`; summaries of the soft and sklearn ridges; `submission_check_items*.parquet`; `irt_build.log`; `emb/qwen3vl8b/_progress.csv`.

So, to run the **simulation** you only need to rebuild `cells_binary.parquet` (steps 1 and the first command of step 2), and to run **build_submission.py** you also need `final/items.parquet` (step 2, final fit only: `--fold final`). Note that refitting the IRT with the same seed should give the same values, but SVI on another machine may differ slightly. The built zips in `submissions/` are the reference.

---

## 11. Codabench: how it runs and what happened

### 11.1 Mechanics

* Upload a zip with all files at the root. The platform downloads the repos in `models.txt`, then imports `model.py` in the parent (validation, labelling) and in one GPU worker.
* Workers restart at every budget checkpoint, which is why b_hat is cached on disk.
* In the label submissions, the parent and the worker each hold the 8B model (about 16 GB each). The `label_update` upload ran successfully, so memory is enough.
* Most of the run time is embedding new items (about 0.16 s per item on an A100).
* Guneesh does all Codabench uploads.

### 11.2 History

1. First upload failed with **`PAIEC-HF-REMOTE-CODE`**: the loader passed the repo name through a variable. Fixed by writing the literal `"Qwen/Qwen3-VL-8B-Instruct"` in each `from_pretrained` call in `core.py`.
2. `ap_pure` ran: ALC ≈ 0.217 (9 pairs).
3. `label_update` ran: ALC ≈ 0.178 (8 pairs).
4. Four stretch zips uploaded; the queue was long and then the site went down (22 Oct). Results pending.

### 11.3 The probe submission

`probe_submission/` (and `probe_submission.zip`) is a diagnostic, not a predictor. Hosted logs are hidden, but the per-budget Brier table is visible. If you predict exactly 0.5 for everything, the Brier score is exactly 0.2500 whatever the outcomes. So the probe predicts 0.5 at budget k if check k passed, and 0.0 if it failed:

| Budget | Check |
|---|---|
| 0 | transformers ≥ 4.57 |
| 1 | Qwen3-VL-8B loads from the prefetched cache and gives a finite 4096-number vector (only tried on a CUDA GPU with ≥ 20 GB) |
| 3 | `torch.cuda.is_available()` |
| 7 | GPU memory ≥ 30 GB |
| 15 | GPU memory ≥ 60 GB |
| 31 | `accelerate` importable |

It inspects only our own environment and never records or sends test data. Its result is not recorded here; since `ap_pure` and `label_update` ran successfully, the model clearly works on the platform.

---

## 12. Open questions and next steps

### 12.1 Immediate (CPU only, about 1–2 days)

1. **Upload `label_update_3improvements_tuned.zip`** when Codabench is back.
2. **Build the stretch 1 version.** Tune `sigma_item` (scales 0.4–1.0) and `sigma_delta` (scales 1–3) for the stretch 1 ridge with `simulate_alc.py` and the leave-one-fold-out rule, then build:

   ```
   python pipeline/build_submission.py --variant label_update --name label_update_stretch1_tuned --ridge_dir data/ridge_within \
       --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
       --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0      # replace the scales with the tuned ones
   ```

   Run all tests in section 7. The GPU check needs `data/submission_check_items_label_update_stretch1.parquet` (present). Keep it only if it beats the tuned zip clearly (it currently wins on 4 of 5 folds).
3. Read the stretch results from Codabench when they arrive; record them (do not tune on them).

### 12.2 The LLM API question (needs a team decision)

The organizers confirmed that calls to the approved inference APIs are allowed during scoring. An LLM could help with our weakest part, **ranking items within a benchmark**:

* ask an LLM to rate each item's difficulty and add that as a feature of b_hat; or
* ask an LLM to answer the item and use "the LLM got it wrong" as a sign of difficulty.

That is about one call per item (≤ 1,000 per run), cached on disk so worker restarts do not repeat calls. Before building anything:

1. **Ask the organizers how API keys reach the submission at scoring time, and who pays.** The page only says "Supply your own development compute and API keys". Putting a key inside the zip is risky (the code must be released for a presentation), so this must be clarified first.
2. Ask how item assets (such as images) are passed (the page describes `item_content` as plain text).
3. Test locally first on a sample of training items (for example 20 per benchmark × 115 benchmarks; a few dollars): does the within-benchmark correlation improve over 0.10–0.17? The key must be set as an environment variable by a team member, never written into code or committed.
4. Build it behind a switch, so that the current zip stays unchanged.

The extra network calls also change the worker count (API submissions get up to 16 workers, but "submissions that also load local models may receive the lower limit").

### 12.3 Other ideas (from `report.md` section 13), in order of expected value

1. **Better item difficulty** (largest possible gain): other embedding layers (middle layers often encode content better; test the within-benchmark correlation on a few benchmarks before re-embedding everything), a larger embedding model (GPU memory is the limit, since the parent and the worker each hold a copy), simple item features (text length, `item_features`).
2. **Per-benchmark discrimination:** a second shift that scales (theta − b_hat), prior centred at 1, estimated in the same Newton fit.
3. **Label selection:** compare the Fisher rule with random buying in the simulation; with item shifts, maybe prefer items no other subject has labelled yet.
4. **One shared shift per subject across benchmarks** (helps only when a subject appears in several pairs of a draw, which looks rare).
5. **Model family matching** for unseen names, and a similar rule for unseen harnesses (low priority: the date rule showed no local effect).

### 12.4 Plan to the deadline (30 Oct AoE)

* Now: upload the tuned zip; tune and build the stretch 1 version.
* Next 2–3 days: label-use ideas in the simulation; decide on the LLM API question.
* If GPU budget allows: embedding layer test on a few benchmarks.
* **Freeze the final zip 2 days before the deadline** (28 Oct), run every check, upload, finish the report (`report.md` is the base; disclose Qwen3-VL-8B and the use of Claude as a coding assistant).

---

## 13. Working rules

These rules were agreed with Guneesh and must be followed by whoever takes over:

* **Ask before deciding.** If there is any doubt, or a decision has to be made, ask Guneesh (or the team) first.
* **Clean, safe code.** Do not break basic things while debugging. Every new feature gets a switch; with all switches off the code must give exactly the old predictions, and a test must check this.
* **Keep a change only if** it beats the noise in the simulation (difference larger than about 2 standard errors) **and** wins when tuned on 4 folds and tested on the 5th.
* **Every zip passes** the logic tests, the starting-kit tools and the GPU check (or is byte-identical in code to a zip that passed it) before upload.
* **Never** fit anything on the test data or on Codabench feedback. Never try to identify anonymous test subjects or benchmarks.
* **Never** commit or paste a Hugging Face token or any API key. Never send personal e-mail addresses to outside services.
* **Do not open or use `SUBMISSION_1.md`** until Guneesh says so.
* Guneesh does all Codabench uploads.
* Write explanations in simple English.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| Subject | An AI system plus its settings (name, harness, provider, effort, ...) |
| Item | One task/question of a benchmark |
| Pair | One subject on one benchmark |
| Cell | One subject on one item (may have several responses) |
| IRT | Item Response Theory: P(correct) = sigmoid(ability − difficulty) |
| theta | Ability. Here: sum of 4 parts (name, harness, provider, effort) |
| b, b_hat | Item difficulty (from IRT) and its prediction from the text (ridge) |
| zeta | A subject's extra ability on one benchmark |
| delta | Shift of a whole benchmark's difficulty, learned from labels |
| epsilon | Shift of one item's difficulty, learned from labels of any subject |
| sigma_* | Prior spread (standard deviation) of a shift |
| MAP | Maximum a posteriori: the most probable values given the data and the priors |
| Laplace approximation | Treat the posterior as a Normal centred at the MAP, with covariance = inverse Hessian |
| Probit approximation | E[sigmoid(z)] ≈ sigmoid(m / √(1 + πv/8)) for z ~ N(m, v) |
| Fisher information | How much a label tells us; p(1−p) for a logistic model |
| Ridge regression | Linear regression with a penalty lambda on the size of the coefficients |
| Embedding | The vector of numbers a language model produces for a text |
| Brier score | Mean of (prediction − outcome)² |
| ALC | Area under the learning curve: weighted mean of the Brier scores at budgets 0, 1, 3, 7, 15, 31 |
| Fold | One of 5 groups of 23 binary benchmarks held out for testing |
| AP | Agent Psychometrics, the paper and code our method is based on |
| LEGO-MB | The multi-benchmark IRT model whose per-benchmark ability deviation (zeta) we use |
| Stretch 1 / 2 | The two ways we tried to make the ridge's difficulty predictions less squeezed |
