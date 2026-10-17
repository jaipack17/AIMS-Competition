# Starting Kit — Predictive AI Evaluation Competition

Your submission is a zip containing a `model.py` at its root that defines:

```python
def predict(input: list, labeled: list | None = None) -> float
```
- `predict()` is called once per distinct `[subject, item]` input; it must accept the input pair as its first argument, plus the labeled list as its second, and return the predicted probability (in [0, 1]) that the subject answers the item correctly. The platform reuses that probability for repeated outcomes with identical inputs.
- `input` is a Python `list` of exactly two `dict`s — `[subject, item]` — and every value is a plain `str` (blank fields are the empty string `""`):
  - `subject` describes the AI system being evaluated, with exactly these keys: `normalized_name`, `provider`, `release_date`, `access_date`, `harness`, `reasoning_effort`, `harness_version`, `subject_features_extra`.
  - `item` describes the task, with exactly these keys: `item_content` (the task text), `item_features` (task metadata), `interactors` (other agents/tools involved in the evaluation), and `benchmark_id` (the stable anonymous benchmark alias). In streaming evaluation, the same alias accompanies both targets and acquired observations, so predictors can match evidence by subject and benchmark without receiving benchmark names.
- `labeled` is a `list` of `[[subject, item], label]` entries: each holds a pair shaped exactly like `input`, plus an integer `label` (1 = the subject answered that item correctly, 0 = incorrectly). It is an empty list when no labels were revealed — never rely on it being non-empty.
- The return value must be a single `float` in [0, 1] (anything `float()` can convert is accepted; `NaN`, infinities, or out-of-range values fail the submission). Example `input`: 

```python
[
  { # subject
    "normalized_name":        "CalculatorNet-3000",
    "provider":               "Abacus Inc.",
    "release_date":           "2026-01-01",
    "access_date":            "2026-02-01",
    "harness":                "claude_code",
    "reasoning_effort":       "low",
    "harness_version":        "1.2.3",
    "subject_features_extra": "",
  },
  { # item
    "item_content":  "What is 1 + 1?",
    "item_features": "tier=easy;points=1",
    "interactors":   "",
    "benchmark_id":  "benchmark_274926",
  },
]
```

- We defined a "subject" as a combination of base model together with its evaluation settings (provider, harness setting, reasoning effort, ...). The same base model run under different settings is a different subject. 


- The same item may have multiple independent outcomes for a subject. All those outcomes contribute to scoring, using the same predicted probability. The platform selects complete `(benchmark, subject)` groups under a draw limit of 1,000 distinct source items before the acquisition/evaluation split, with preference for benchmarks and subjects represented less often in the draw and random tie-breaking. Each chosen group contributes all its source items to the split, with all response runs of its evaluation items scored; groups that exceed the remaining budget are skipped. The draw may include fewer than 1,000 source items when no further complete group fits; six-budget evaluation multiplies predictor calls. There is no fixed subject count per benchmark. Subjects are chosen by the platform and are not announced in advance. See `sample_code_submission/model.py` for the full interface documentation. (Naming note: in the runtime CSV files the subject identifier column is called `model_id` for historical reasons — it carries opaque subject aliases like `subject_582731`, which are stable across submissions so you can recognize a subject you have seen before.)

Optional files, all at the zip root:

- `labeling.py` — optional `acquisition_function(input, prediction=None, labeled=None, context=None) -> bool`: choose whether to reveal the current acquisition candidate. `prediction` is already computed with the current evidence; do not call the predictor again to estimate uncertainty. Omitting the file uses deterministic random acquisition. A malformed hook or failed call fails the submission.
- `models.txt` — public HuggingFace repos your code loads, one per line, maximum 5. Required if you load any HF model: repos are pre-downloaded before your code runs and the hub is offline at runtime, so undeclared models fail to load.
- `requirements.txt` — extra pip packages (pre-built wheels only) on top of the preinstalled pinned stack (torch, transformers, scikit-learn, ...).
- `prediction_config.json` — optionally reduce prediction concurrency, for example `{"concurrency": 1}` for sequential execution. Integers from 1 to 16 are accepted; the platform caps the value at its configured limit. API-based submissions normally use up to 16 workers, ordinary CPU submissions use up to four, and GPU submissions use one. Unrecognized API clients or submissions that also load local models may receive the lower limit. The actual worker count is recorded in `labeling_log.json`.

Runtime notes: 
- Network access is restricted to an allowlist (no general internet, which means you cannot fetch data or code at runtime).
- The whole run must finish within 8 hours, so per-call cost of `predict()` matters.
- Heavy training should happen offline before submitting (e.g. publish a trained model to HuggingFace and load it at module init).
- API-based submissions use up to 16 isolated prediction processes by default; ordinary CPU submissions use up to four. Every call still receives one `[subject, item]` pair and returns one probability; no batch function or participant-managed concurrency is required. GPU submissions use one process. Workers share the submission's existing resource allocation and finish within the same eight-hour deadline.
- Each prediction worker imports `model.py` once and keeps its own module state and caches. The parent also imports the model for validation and optional labeling. Predictions must not rely on earlier calls or on state modified by `acquisition_function()`; every call receives its own copy of the evidence available at its checkpoint through `labeled`. Workers restart between checkpoints, including when concurrency is one; acquisition runs sequentially in the parent process. Use `{"concurrency": 1}` for large CPU models that cannot fit multiple copies in memory, or API accounts needing lower request rates. Identical inputs reuse a prediction in both sequential and concurrent execution. Concurrency increases the rate of API requests; it does not provide a per-token discount. One `predict()` call may itself make multiple API requests.
- `labeling_log.json` distinguishes actual `prediction_calls_count`, reused rows in `prediction_reuse_count`, and all scored response rows in `predictions_count`.
- Use unique names for files your predictor writes. Processes have separate temporary directories (`TMPDIR`) and a `PAIEC_PREDICTION_WORKER_ID`, but the submission directory is shared. The platform restores result order, reports progress during prediction, and stops outstanding workers if any prediction fails or exceeds its deadline.

## What's in the kit

- `sample_code_submission/` — a minimal working submission: `model.py` (constant-probability baseline) plus an optional `labeling.py` showing the adaptive-labeling hook (`acquisition_function`), which lets your submission request a few ground-truth labels mid-run.
- `templates/` — starting points for common setups:
  - `hf_submission/` — load one HuggingFace model (declared in `models.txt`).
  - `multi_hf_submission/` — load several HuggingFace models.
  - `labeling_addon/` — drop-in `labeling.py` for adaptive labeling.
- `tools/` — run these before submitting:
  - `python tools/check_submission_zip.py your_submission.zip` — validates the zip layout, `models.txt`, and HuggingFace references.
  - `python tools/run_smoke_test.py your_submission_dir/` — runs your `predict()` (and `labeling.py` if present) against the sample data.
  - `python tools/run_smoke_test.py your_submission_dir/ --concurrency 4` — exercises the same isolated worker pool used for CPU/API submissions. The optional `prediction_config.json` can reduce this concurrency too.
- `sample_data/` — tiny CSVs in the exact format the ingestion program reads at runtime (`test/test_items.csv`, `test/test_pairs.csv`, `test/models.csv`) plus example ground truth (`ref/ground_truth.csv`). The ingestion program joins these into the `[subject, item]` pairs your `predict()` receives; `run_smoke_test.py` builds the same pairs locally.

## Streaming test-time adaptation

Eligible subject–benchmark pairs have **at least 80 distinct items**. Each pair
has a persistent random 50/50 split into an acquisition pool and a fixed
unqueryable evaluation pool. Repeated response runs of an item stay in the same
pool. Existing assignments persist across submissions and snapshot updates;
new items balance the pools without moving old items. Odd counts give the extra
item to acquisition. Eligibility counts distinct items, not response runs.

Acquisition candidates arrive one at a time in a fixed random order. The hook
receives the same `[subject, item]` input as the predictor, its current prediction,
previously revealed examples in `labeled`, and this context:

```python
{
    "subject_id": "subject_582731", "benchmark_id": "benchmark_274926",
    "labels_remaining": 31, "labels_acquired": 0, "max_labels": 31,
    "items_remaining": 40,  # includes the current candidate
}
```

Return a Python `True` to reveal one recorded 0/1 response for the current item,
or `False` to skip it permanently. The maximum is 31 labels per pair. The
predictor and hook receive all evidence purchased so far, across sampled pairs,
as `[[[subject, item], outcome], ...]`; this list is copied for each call.
Future acquisition candidates and evaluation outcomes are not supplied.

The platform runs one acquisition trajectory with nested prefixes, evaluating
at caps **0, 1, 3, 7, 15, 31**. All pairs finish a checkpoint before any pair gets
higher-budget labels. Each checkpoint evaluates the same fixed pool with frozen
evidence, with up to 16 concurrent API prediction calls in total. Committed
predictions cannot be revised after more labels are revealed. If the policy
exhausts the acquisition stream before reaching a cap, later caps use the
labels it actually acquired; feedback reports that count.

For `A` acquisition candidates and `E` distinct evaluation inputs, the upper
work estimate is `A + 6E` predictor calls. Acquisition can stop early; repeated
identical evaluation inputs reuse a prediction within each checkpoint. With
no acquisition hook, random selection needs no acquisition predictor calls.
The approximately 1,000-item draw limit applies before the split, not to the
sum of calls across six budgets. The tiny local smoke dataset checks the hook
signature and predictor; it is not a complete simulation of this protocol.

### Compatibility with existing acquisition hooks

Existing `acquisition_function(input) -> float` submissions remain valid. The
runtime inspects the signature and calls them with **one argument only**; a
participant exception is never retried using another call shape. Finite numeric
priorities, including negative scores and zero, retain their ranking meaning.

For these legacy hooks, the platform scores every acquisition candidate once
with no acquired labels available, commits all priorities, and reveals the top
`n` candidates per subject–benchmark pair at budgets 0, 1, 3, 7, 15, 31. Ties use
the fixed random candidate order. The resulting label prefixes are nested.
This compatibility path uses pool ranking; it does not convert priorities into
streaming decisions or make another predictor call on behalf of the hook.
Evaluation items and their outcomes never enter this ranking pool.

New hooks may declare `prediction=None`, `labeled=None`, and `context=None`.
The runtime supplies supported named arguments (including keyword-only subsets),
or all four positional arguments when that is the declared interface. These
hooks return a native boolean decision. If `prediction` is accepted, the runtime
computes it once and passes it in; a hook needing only labels/context avoids that
extra prediction. The example hook also returns a numeric priority when invoked
without context by an older evaluator.

Both paths share the fixed evaluation pool, label caps, curve weights, and
checkpoint commitments. This preserves the old **function interface and ranking
semantics**, not old numerical results: the evaluation split and budgets have
changed. Predictors that assume exactly five labels still need to be updated.

## Scoring

Formative feedback appears in your submission's **Logs → Scoring output** tab. The report starts with an ALC summary and then presents one text section per budget. Each table has five columns and one row per evaluated subject and benchmark, identified by anonymous IDs such as `subject_582731` and `benchmark_274926`. The six-digit numbers are assigned randomly and do not encode dataset size or ordering. Each ID is assigned once and stays fixed across submissions and evaluation snapshot updates; adding or removing subjects or benchmarks does not renumber existing IDs. Matching IDs always identify the same subject or benchmark, and an ID is never reused for another identity. Only evaluated combinations appear in the table, and benchmark names are not shown. Access is restricted to the submitter and organizers; prediction and scoring downloads remain hidden.

- **Brier score:** mean squared error between the predicted probability and the 0/1 outcome at an individual label budget. Lower is better; 0 is perfect and a constant prediction of 0.5 scores 0.25.
- **Brier ALC ↓:** the primary leaderboard score for every submission, with or without test-time adaptation. ALC means normalized area under the Brier learning curve across the six label budgets. It is a weighted average of Brier scores, so it stays on the same scale and lower is better.
- **Calibration ECE:** count-weighted mean absolute difference between mean predicted probability and observed positive rate in ten fixed bins: [0, 0.1), …, [0.9, 1]. Lower is better. It depends on binning and sample counts and is a diagnostic, not a replacement for Brier score.
- **Number of Unique Subject-Item Pairs:** the number of distinct evaluated subject–item pairs. Repeated responses for the same pair count once in this column.

Within each subject–benchmark pair, the six budget scores have weights **0.1, 0.2, 0.2, 0.2, 0.2, 0.1** at budgets **0, 1, 3, 7, 15, 31**. This is the normalized trapezoidal area on `log2(1 + budget)`: the endpoints belong to one interval each and the interior points belong to two, giving the endpoints half the weight. The weights sum to 1. The **ALC summary** table reports Brier ALC and the corresponding weighted ECE for each subject–benchmark pair. Each **Label budget** section reports **Brier score** and ECE at that cap. The five columns are subject ID, benchmark, Number of Unique Subject-Item Pairs, Brier ALC (summary) or Brier score (budget), and Calibration ECE. Budget and labels-acquired columns are omitted; when a policy uses fewer labels than allowed, a short note below that budget’s table reports the actual usage. Counts can differ across groups but remain fixed across budgets for a given group. The website example uses a dropdown to show one view at a time; private Codabench scoring logs present the views as text sections.

The aggregate **Brier ALC ↓** and **ECE** leaderboard columns take the unweighted mean of these curves: every evaluated subject–benchmark combination has equal weight, regardless of its number of responses. ECE is calculated separately within each combination before averaging. The shared Results leaderboard opens separately when announced by the organizers.

All submissions use the same budget grid and scoring rule. Omitting `acquisition_function` uses the default random label-selection policy; the predictor still receives the acquired labels and may use them to adapt. A predictor that ignores labels and produces the same predictions at every budget has a flat learning curve, so its Brier ALC equals its ordinary Brier score. Randomized predictions can make the measured curve fluctuate even without adaptation.

For a fixed evaluation snapshot and subject–benchmark combination, all stored response rows in its fixed evaluation pool are included at every budget. Acquisition items are never scored. Scores can still change with labeling examples or predictor randomness, and coverage can change as benchmarks and subjects are selected. Compare matching subject–benchmark IDs within the same snapshot. Older archived evaluations may contain only subject-level summaries.
