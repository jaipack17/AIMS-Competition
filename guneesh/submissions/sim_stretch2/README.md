# Submission `sim_stretch2`

**In one sentence:** a test build: the 3 improvements with stretch 2 (ridge predictions stretched by k = 4.05).

| | |
|---|---|
| Zip | `sim_stretch2.zip` (files unpacked in `files/`) |
| Uses labels? | Yes (Fisher-information `labeling.py`, identical in every label submission) |
| Difficulty model | original ridge (`data/ridge`) stretched by k = 4.05 around c = 0.0146 |
| Settings | sigma_delta 0.570 (stretched), sigma_zeta 1.110, sigma_item 0.978; item shifts, averaging, date rule on |
| Local 5-fold result | Brier ALC **0.2229** |
| Codabench | 0.2025 |
| Status | Experiment (worse locally). |

## Build command

```
python pipeline/build_submission.py --variant label_update --name sim_stretch2 \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json --item_shifts --average --sigma_item_scale 0.6 --stretch data/ridge/versionB_4parts_logitnormal/stretch.json
```

## Local simulation (150 draws, 1,111 pairs; lower is better)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| this zip | 0.2535 | 0.2397 | 0.2270 | 0.2161 | 0.2066 | 0.2002 | 0.2229 |

Worse than `label_update_3improvements` (0.2227) without stretching.
