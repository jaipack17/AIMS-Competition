# Submission `label_update_stretch1_date`

**In one sentence:** `label_update_stretch1` (labels + the stretch 1 ridge) plus the release-date rule for model names not seen in training.

| | |
|---|---|
| Zip | `label_update_stretch1_date.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | Same `model.py` and `labeling.py` as `label_update`; same `core.py` as `label_update_date`. |
| Codabench result | Not uploaded. |
| Local 5-fold result | Brier ALC **0.2254** |
| Status | Experiment. Not better. |

---

## What it combines

* **Stretch 1 ridge** (`lambda` = 1e5, chosen by within-benchmark R²) and `sigma_delta` = 0.587. See `label_update_stretch1/README.md`.
* **Release-date rule**: an unseen name gets `theta_name = -0.154 + 0.217 * years since 2024` (clipped to 2023–2026). See `label_update_date/README.md`.
* `sigma_zeta` = 1.110, Fisher-information label choice.

## How it was built

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch1_date --ridge_dir data/ridge_within \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json
```

## Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update_stretch1` | 0.2591 | 0.2456 | 0.2285 | 0.2155 | 0.2066 | 0.2017 | 0.2250 |
| **`label_update_stretch1_date`** | 0.2589 | 0.2459 | 0.2298 | 0.2161 | 0.2071 | 0.2018 | **0.2254** |

As with `label_update_date`, the date rule makes no real difference in the local test, because the held-out subjects mostly have known names.
