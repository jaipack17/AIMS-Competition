# Submission `label_update_date`

**In one sentence:** `label_update` plus the **release-date rule**: when a test subject's model name was never seen in training, we guess its name ability from its release date instead of using 0.

| | |
|---|---|
| Zip | `label_update_date.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | `model.py` and `labeling.py` are the same as `label_update`. `core.py` is a newer version that reads `name_date_prior.json`. |
| Codabench result | Not uploaded. |
| Local 5-fold result | Brier ALC **0.2249** (vs 0.2246 for `label_update`; the difference is noise) |
| Status | Experiment, built before averaging and item shifts existed. Its date rule lives on inside `label_update_3improvements`. |

---

## 1. The problem

The subject's ability is `theta_name + theta_harness + theta_provider + theta_effort`. For a model name that is not in training, `theta_name` used to be 0, which means "an average model". But newer models are usually stronger. The test set may contain new models.

## 2. The rule

```
years      = (release_date - 1 January 2024) in years, clipped to [-1, 2]   (that is, 2023 to 2026)
theta_name = a + b * years = -0.154 + 0.217 * years
```

* `a` and `b` come from an ordinary least squares line through the **595** named training models that have a readable release date. A name with several dates uses their median. Fitted by `pipeline/name_date_prior.py`, saved in `name_date_prior.json`.
* Average `theta_name` by release year in training: about -0.32 (2022), -0.30 (2023), -0.03 (2024), +0.15 (2025), +0.22 (2026). Newer is stronger.
* Leave-one-name-out error: 0.485 with the line, against 0.516 for always guessing 0. The line also beats 0 in each of the 5 fold fits.
* Accepted date formats in `subject["release_date"]`: `YYYY-MM-DD`, `YYYY-MM` (read as the 15th), `YYYY` (read as 1 July). Anything else, or blank, keeps the old value 0.
* **Known names are never changed.** A blank name counts as unseen.

## 3. Files

Same as `label_update`, plus `name_date_prior.json`:

```
a = -0.15382, b = 0.21716, years_min = -1, years_max = 2,
loo_rmse_line = 0.48504, loo_rmse_zero = 0.51613
```

## 4. How it was built and tested

```
python pipeline/name_date_prior.py                     # writes data/irt/<run>/final/name_date_prior.json
python pipeline/build_submission.py --variant label_update --name label_update_date \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json
python analysis/test_name_date_prior.py label_update_date
```

The test checks that the submission reads all 363 different training date strings exactly like the fitting script, that odd dates behave as expected, and that known names are unchanged.

## 5. Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| **`label_update_date`** | 0.2597 | 0.2430 | 0.2287 | 0.2159 | 0.2075 | 0.2025 | **0.2249** |

## 6. What we learned

In our local test the rule has no measurable effect, because almost all subjects of the held-out benchmarks have names that **were** seen in training (the held-out folds remove benchmarks, not models). The rule can only help on Codabench when a test subject is a genuinely new model. It is cheap and safe, so we kept it in the later versions.
