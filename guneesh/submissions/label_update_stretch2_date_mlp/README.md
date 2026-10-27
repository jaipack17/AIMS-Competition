# Submission `label_update_stretch2_date_mlp`

**In one sentence:** `label_update_stretch2_date` with the **MLP difficulty model** (`pipeline/mlp_b_train.py`: average of 6 small MLPs,
bundled as `mlp.npz`) in place of the ridge. Everything else is the same as its ridge twin.

| | |
|---|---|
| Zip | `label_update_stretch2_date_mlp.zip` (17.9 MB; files unpacked in `files/`, see the note below) |
| Uses labels? | Yes (Fisher-information `labeling.py`, identical in every label submission) |
| Difficulty model | MLP ensemble (h64 and h256x256 x 3 seeds), final fit; `core.mlp_b` runs it in numpy; stretch 2 with k = 0.649 around c = 0.0146 |
| Settings | sigma_delta 0.510 (stretched), sigma_zeta 1.110, date rule on (sigmas re-measured for the MLP with the same rules as the ridge zips) |
| Local 5-fold result | Brier ALC **0.2234** (ridge twin 0.2273) |
| MLP minus ridge twin | -0.0039 +/- 0.0009 (better on 5/5 folds) |
| Codabench | 0.1839 |

## Build and test

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch2_date_mlp \
    --ridge_dir data/mlp_b --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json --stretch data/mlp_b/versionB_4parts_logitnormal/stretch.json
```

Built and checked by `analysis/mlp_variants.py --model mlp` (the kit's `check_submission_zip.py`, `run_smoke_test.py`,
then the simulation with configuration `+date`). The ridge twin was rebuilt with the same code
(`analysis/mlp_variants.py --model ridge`) and reproduced the earlier local result exactly. `analysis/test_mlp_submission.py`
checks the numpy MLP against the torch one (max diff 3.8e-7) and the bundled `mlp.npz` against the training predictions
(2.4e-7). Not GPU-checked on Modal.

## Local simulation (150 draws, 1,111 pairs; lower is better)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| this zip | 0.2571 | 0.2426 | 0.2267 | 0.2149 | 0.2056 | 0.2011 | **0.2234** |

Paired comparison with the ridge twin: `PYTHONPATH=analysis python analysis/mlp_variants_report.py`.

Note: `files/` has every file of the zip except `mlp.npz` (33 MB). The same `mlp.npz` (byte-identical in all MLP zips) is in `../../data/mlp_b/versionB_4parts_logitnormal/final/mlp.npz`.
