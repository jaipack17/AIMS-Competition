# Submission `label_update_stretch2_date`

**In one sentence:** `label_update_stretch2` (labels + difficulties stretched by `k = 4.05`) plus the release-date rule for model names not seen in training.

| | |
|---|---|
| Zip | `label_update_stretch2_date.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | Same `model.py` and `labeling.py` as `label_update`; same `core.py` as `label_update_date`. Contains `bhat_stretch.json`. |
| Codabench result | Not uploaded. |
| Local 5-fold result | Brier ALC **0.2273** |
| Status | Experiment. Worse. Do not use. |

---

## What it combines

* **Stretch 2**: `b = 0.0146 + 4.05 * (b_hat - 0.0146)` and `sigma_delta` = 0.570. See `label_update_stretch2/README.md`.
* **Release-date rule**: see `label_update_date/README.md`.
* `sigma_zeta` = 1.110, Fisher-information label choice.

## How it was built

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch2_date \
    --stretch data/ridge/versionB_4parts_logitnormal/stretch.json \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json
```

## Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update_stretch2` | 0.2647 | 0.2463 | 0.2312 | 0.2180 | 0.2095 | 0.2044 | 0.2275 |
| **`label_update_stretch2_date`** | 0.2646 | 0.2460 | 0.2313 | 0.2179 | 0.2095 | 0.2041 | **0.2273** |
