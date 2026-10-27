# Submission `label_update_stretch1_tuned`

**In one sentence:** the tuned method (`label_update_3improvements_tuned`) with the **stretch 1 ridge** (`data/ridge_within`, lambda chosen for ranking inside a benchmark). The best local result with a ridge.

| | |
|---|---|
| Zip | `label_update_stretch1_tuned.zip` (files unpacked in `files/`) |
| Uses labels? | Yes (Fisher-information `labeling.py`, identical in every label submission) |
| Difficulty model | stretch 1 ridge (`data/ridge_within`, lambda 1e5) |
| Settings | sigma_delta 1.173 (2 x 0.587), sigma_zeta 1.110, sigma_item 0.971 (0.6 x 1.618); item shifts, averaging, date rule on |
| Local 5-fold result | Brier ALC **0.2196** |
| Codabench | 0.193 (partial: the run was still "Running" at the last check on 27 Oct) |
| Status | **Recommended final submission** (backup: `label_update_3improvements_tuned`). |

## Build command

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch1_tuned \
    --ridge_dir data/ridge_within --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0
```

## Local simulation (150 draws, 1,111 pairs; lower is better)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| this zip | 0.2489 | 0.2359 | 0.2232 | 0.2127 | 0.2041 | 0.1976 | **0.2196** |
| `label_update_3improvements_tuned` | 0.2517 | 0.2382 | 0.2247 | 0.2144 | 0.2051 | 0.1994 | 0.2214 |

0.0018 +/- 0.0005 better than `label_update_3improvements_tuned`, at every budget, on 4 of 5 folds (fold 4: +0.0009). The simulation of this exact zip reproduced the earlier test build to 0.0 (`analysis/simulate_alc_label_update_stretch1_tuned_final.csv`).

## Code

`core.py` (`09ff0baa`) and `model.py` (`40ce932f`) are the newer MIRT-capable versions; with no `mirt.json` they give exactly the old predictions (regression test `analysis/simulate_alc_check_tuned_v2_regression.csv`). Not GPU-checked on Modal; the logic tests, the starting-kit checks and the smoke test passed.
