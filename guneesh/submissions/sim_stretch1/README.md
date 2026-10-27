# Submission `sim_stretch1`

**In one sentence:** a test build: the 3 improvements on the stretch 1 ridge, **without** the doubled sigma_delta. Built to simulate stretch 1; later uploaded to Codabench too.

| | |
|---|---|
| Zip | `sim_stretch1.zip` (files unpacked in `files/`) |
| Uses labels? | Yes (Fisher-information `labeling.py`, identical in every label submission) |
| Difficulty model | stretch 1 ridge (`data/ridge_within`) |
| Settings | sigma_delta 0.587, sigma_zeta 1.110, sigma_item 0.971; item shifts, averaging, date rule on |
| Local 5-fold result | Brier ALC **0.2209** |
| Codabench | no result (still "Scoring" at the last check on 27 Oct) |
| Status | Experiment; replaced by `label_update_stretch1_tuned`. |

## Build command

```
python pipeline/build_submission.py --variant label_update --name sim_stretch1 \
    --ridge_dir data/ridge_within --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json --item_shifts --average --sigma_item_scale 0.6
```

## Local simulation (150 draws, 1,111 pairs; lower is better)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| this zip | 0.2503 | 0.2375 | 0.2251 | 0.2143 | 0.2046 | 0.1981 | 0.2209 |

Replaced by `label_update_stretch1_tuned` (same with sigma_delta x 2: 0.2196).
