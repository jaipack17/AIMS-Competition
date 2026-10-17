# PAIEC Initial Submission: Dataset Analysis and Predictor Plan

Status as of 2026-10-16. Everything here is measured from the downloaded data unless marked **assumption** or **open**. Scripts that produced the numbers are in `scripts/`; intermediate outputs are in `data/`.

## 1. Task

- Predict P(subject answers item correctly) for held-out benchmarks (benchmark-level cold start).
- Score: Brier ALC, lower is better. It is the weighted average of Brier at label budgets 0, 1, 3, 7, 15, 31 with weights 0.1 / 0.2 / 0.2 / 0.2 / 0.2 / 0.1. Each subject-benchmark pair counts equally.
- Reference points: constant 0.5 scores 0.25; the baseline repo provides an empirical-mean predictor and an LLM-agent predictor (BLE).
- Interface: `predict([subject, item], labeled) -> float` in [0, 1], optional `acquisition_function(input, prediction, labeled, context) -> bool`.

## 2. Data: `aims-foundations/measurement-db`

### 2.1 Scale

| Item | Value |
|---|---|
| Benchmarks | 281 |
| Items | 8,471,422 |
| Responses | 92,580,425 |
| Subject rows | 23,048 |
| Downloaded | benchmarks, subjects, responses tables (2.6 GB) |
| Not downloaded | `items.parquet` (12 GB), `traces` (108 GB), `assets` (155 GB) |

### 2.2 Benchmarks

- Items per benchmark: median 1,819 (10th percentile 150, 90th percentile 47,500).
- Subjects per benchmark: median 14 (10th percentile 2, 90th percentile 112).
- About 239 benchmarks have at least 80 items and at least 2 subjects, so they can yield test-eligible pairs.
- Modality: text 271, image 51, GUI screenshot 7, tabular 5, video 3, audio 2.
- Top domains: reasoning 77, safety 42, agents/tool use 36, medicine 36, mathematics 30, software engineering 28, science 27.
- 205 single-turn, 67 multi-turn, 9 both. 121 benchmarks have repeated trials per item.
- Release years: 2023 (48), 2024 (66), 2025 (84), 2026 (40).

### 2.3 Response types

| Type | Benchmarks | Share of responses |
|---|---|---|
| binary | 163 | 77.1% |
| mixed | 49 | 13.7% |
| ordinal | 14 | 3.1% |
| likert_5 | 6 | 2.4% |
| continuous_bounded | 17 | 1.5% |
| other (continuous, fraction, nominal, win rate, likert_10) | 46 | about 2.2% |

**Open:** the organizers' rule for turning non-binary responses into a 0/1 label for test. Not yet found. It decides how much training data is usable and whether test benchmarks include these types.

### 2.4 Subjects

`subject_id` is a different hash in every benchmark, so `normalized_name` is the only link across benchmarks.

| Field | Non-empty | Distinct values |
|---|---|---|
| `normalized_name` | 42% | 808 |
| `provider` | 42% | 91 |
| `release_date` | 33% | 395 |
| `access_date` | 0.7% | 32 |
| `harness` | 94% | 289 |
| `harness_version` | 2.6% | 43 |
| `reasoning_effort` | 0.3% (44 named rows) | 7 |
| `subject_features_extra` | 98% | 21,479 |

Findings:

1. **Names link only part of the data.** 101 of 281 benchmarks have no named subject. Some unnamed rows are LLMs whose model label sits in `display_name` or `subject_features_extra`. Others are not LLMs (fine-tuned classifiers, RL checkpoints, human students).
2. **Name coverage is high on average but the tail is cold.** Per benchmark, the median share of names seen in another benchmark is 100% (mean 89%); 10 of 180 named benchmarks fall below 50%. 348 of 808 names appear in only one benchmark. Popular models appear in 30 to 77 benchmarks.
3. **`harness` is not a reusable scaffold.** It is mostly a per-benchmark label (for example "HELM", "VLMEvalKit"). Only 18% of named rows have a harness value that appears in another benchmark.
4. **`reasoning_effort` and `harness_version` are nearly empty.** The full (name, harness, effort, version) combination is seen elsewhere for only 15% of rows.
5. **Provider is a reliable fallback:** 99% of non-empty values appear in other benchmarks.
6. **No clear era effect:** name coverage by release year is flat at about 80 to 95% for 2022 to 2026.
7. Row counts mislead: one benchmark (alignment faking) contributes 4,278 rows of unique experiment names. Per-benchmark shares are the right unit.

The hidden test set may differ from this; coverage there cannot be observed.

### 2.5 Dimensionality of ability (binary benchmarks)

Within a benchmark (60 qualifying benchmarks, held-out Brier):

| Ability model | Brier |
|---|---|
| 1D (Rasch) | 0.1142 |
| + 1 extra dimension | 0.1098 |
| + 2 extra | 0.1093 |
| + 3 extra | 0.1095 |

Across benchmarks (63 named subjects x 61 benchmarks, 17.5% of cells filled): adding interaction terms to an additive subject + benchmark model made held-out MSE 1.5 to 1.8 times worse. The densest block (13 x 38) shows a possible second axis (62% / 27% of residual variance), but the imputation used a rank-2 fill, so that is circular and the block is tiny.

**Conclusion:** use a one-dimensional ability. A second dimension is plausible but cannot be estimated from this overlap, nor from 0 to 31 labels.

Caveat: the within-benchmark sample was the first 60 (alphabetical) benchmarks passing the size filters (at least 8 subjects, at least 60 items, at most 400k rows), not a random sample.

## 3. Related work used

- **Agent Psychometrics** (COLM 2026, arXiv 2604.00594): IRT `P = sigma(theta_m + theta_s - beta_j)`, with difficulty predicted from task features by a ridge regression. Two-stage fit: first IRT, then ridge on the frozen difficulties. On new benchmarks, AUC was about 0.69 to 0.79 against an oracle of about 0.91. Adaptive testing with Fisher information `p(1-p)` and a MAP ability update.
- **Truong et al. 2025** (ICML, arXiv 2503.13335): Rasch model with a difficulty predictor on question embeddings, adaptive question selection.
- Not read: the appendices of the first paper (including its comparison of two-stage and joint fitting), and the full text of the second (only search summaries).

## 4. Predictor architecture

```
z = theta_subject - beta_item + delta_(subject, benchmark)
p = clip( sigmoid(a * z + c) )
```

**theta_subject** is a one-dimensional ability from all subject fields, additive and shrunk:

- Name: text embedding plus a learned row if seen in training, with token backoff so model variants share strength.
- Provider: embedding row (reliable fallback).
- `harness` and `subject_features_extra`: weak text features, not a categorical harness effect.
- Dates: smooth release-date term and `access_date - release_date`, kept only if they help in leave-benchmark-out evaluation (guarding against dates proxying for benchmark era).
- Reasoning effort: small ordinal effect, mostly unknown.
- Missing fields: a learned "unknown" value, never dropped rows. Subjects with no usable name rely on provider and text features.

**beta_item** from item text: a small pretrained encoder (offline cache, declared in `models.txt`) embeds `item_content`, `item_features`, `interactors`. A ridge head predicts the within-benchmark difficulty deviation. The benchmark mean difficulty is predicted from the average item embedding or left to the label correction.

**delta** is a per-pair scalar from a 1D MAP update on the labels in `labeled`. Its Gaussian prior makes it 0 with no labels. The prior variance is tuned on held-out benchmarks.

**Calibration** `a, c` is fitted for Brier on held-out benchmarks (the papers optimize AUC).

**Acquisition:** buy when `|p - 0.5|` is below a threshold (high `p(1-p)`), and always buy when `items_remaining <= labels_remaining`. The threshold is tuned by simulation and may depend on the remaining budget.

## 5. Training (offline)

1. **Stage 1, IRT fit** on binary responses: `sigmoid(theta_subject - beta_item)` with a hierarchical `beta_ib = mu_b + eps_ib`. Subject abilities are tied across benchmarks through the field-based theta. Repeated trials are averaged or treated as binomial. Items answered the same way by every subject are down-weighted.
2. **Stage 2, difficulty head:** ridge from embeddings to `eps`. The penalty is chosen by leave-benchmark-out cross-validation. Items are weighted by the precision of their stage-1 estimate, and each benchmark's total weight is capped.
3. **Calibration and prior variance** are fitted on held-out benchmarks with simulated label budgets.
4. **Packaging:** small frozen files (ability tables, ridge weights, calibration numbers). The inference path is pure NumPy plus the encoder from `models.txt`.

Items will be embedded one benchmark at a time, keeping only vectors, to avoid storing the 12 GB of item text.

## 6. Validation protocol

- Leave-benchmark-out folds.
- For each held-out pair with at least 80 items, a persistent 50/50 acquisition/evaluation split.
- Simulate budgets 0, 1, 3, 7, 15, 31 and compute Brier ALC.
- Baselines: constant 0.5, pair empirical mean, and BLE-style baselines where feasible.
- Ablations: date term on/off; field-based theta vs name-only; 1D vs low-rank ability; random vs uncertainty acquisition.
- Report budget 0 and budget 31 separately, not only the ALC.

## 7. Risks and open questions

- **Binarization rule** for non-binary responses is unknown (section 2.3).
- **Test subject mix** is unknown; cold-start robustness depends on the field-based fallbacks.
- **Long items:** agentic prompts may exceed the encoder's token limit, so truncation must be chosen deliberately.
- **Runtime:** `predict()` is called once per distinct pair at each of six budgets, so embeddings should be cached per worker.
- **Compliance:** train only on inventory data, and disclose every training source, including the encoder, and any LLM coding-assistant use in the technical report. Final code and report are due 30 October 2026, 23:59 AoE.

## 8. Next steps

1. Find the binarization rule.
2. Build the evaluation harness with the constant and pair-mean baselines.
3. Embed items and fit the two-stage model.
4. Add the label correction, calibration and acquisition.
5. Package, validate with `check_submission_zip.py` and `run_smoke_test.py`, and submit to Codabench.

## Reproducing the numbers

```
.venv/bin/python scripts/list_sizes.py        # per-benchmark table sizes -> data/table_sizes.json
.venv/bin/python scripts/pull_small.py        # benchmarks + subjects tables
.venv/bin/python scripts/pull_responses.py    # responses tables (2.6 GB)
.venv/bin/python scripts/cold_start_audit.py  # subject-field coverage
.venv/bin/python scripts/dimensionality.py    # within- and across-benchmark dimensionality
```
