# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The repository is empty (no code, no git history yet). It is being set up for an initial submission to the **Predictive AI Evaluation Competition (PAIEC)**, NeurIPS 2026, run by Stanford AIMS: https://aims.stanford.edu/competition. The user will give step-by-step instructions. Update this file as real structure and commands appear. Everything below comes from the competition page, not from code here.

## Task

`predict(input, labeled)` returns the probability (finite float in [0, 1]) that a given AI system (subject) answers a given benchmark item correctly. It must do this without running the subject on the item. Test benchmarks are fully held out (benchmark-level cold start).

- **Subject inputs:** name, provider, release/access dates, harness and version, reasoning effort, extra settings.
- **Item inputs:** task text, item features, interactors (tools/agents), anonymous benchmark ID.
- All input values are strings, with `""` for missing.
- **Test-time labels:** budgets of 0, 1, 3, 7, 15, 31 revealed outcomes per subject-benchmark pair. An optional `acquisition_function` in `labeling.py` picks which outcomes to reveal. Without it, labels are random. Pairs with ≥80 items are split 50/50 into acquisition and evaluation pools.
- **Metric:** Brier ALC, lower is better. It is the area under the Brier learning curve over log₂(n+1) with budget weights 0.1/0.2/0.2/0.2/0.2/0.1. Each subject-benchmark pair is weighted equally. ECE is diagnostic only.

## Submission format

A ZIP with files at its root:
- `model.py` (required): defines `predict(input, labeled)`.
- `labeling.py` (optional): defines `acquisition_function`.
- `models.txt` (optional): Hugging Face repos to prefetch.
- `requirements.txt` (optional, subject to organizer policy).
- Any other artifacts.

Validate locally with `tools/check_submission_zip.py` and `tools/run_smoke_test.py`. These come from the starting kit (Codabench competition 17828). They are not in this repo yet.

Failure codes: 10/11 invalid `model.py`; 12 invalid `labeling.py`; 40 prediction exception; 41 timeout; 42 invalid probability; 50 total timeout.

## Constraints to respect in code

- 8 hours per run, including setup.
- No runtime model downloads. List weights in `models.txt` instead (max 5 repos, ≤300B params and ≤1,000 GB each, ≤1,024 GB combined).
- Network access only to approved inference APIs. Web search and browsing are prohibited.
- Extra package installs and `trust_remote_code` are disabled by default.
- Train only on inventory data (HF `aims-foundations/measurement-db` and the benchmark inventory). Do not try to extract private test data or de-anonymize subjects or benchmarks.
- Predictions must always be finite and within [0, 1]. Guard against exceptions and NaN, since either fails the run.

## Resources

- Starting kit: Codabench competition 17828.
- Baselines: GitHub `aims-foundations/paiec_baseline` (baseline predictors and acquisition functions).
- Training data: Hugging Face `aims-foundations/measurement-db`.
- Support: aimslab@cs.stanford.edu.

## Deadlines and deliverables

- Final code and technical report are due **30 Oct 2026, 23:59 AoE** via OpenReview. The report uses the NeurIPS template and runs 4–8 pages of main text.
- Disclose all training sources and any LLM coding-assistant use in the report.
- Keep the method reproducible. Organizers rerun released code.
