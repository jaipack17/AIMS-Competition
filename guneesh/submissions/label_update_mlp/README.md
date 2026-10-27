# Submission `label_update_mlp`

**In one sentence:** `label_update` with the **MLP difficulty model** (`pipeline/mlp_b_train.py`: average of 6 small MLPs,
bundled as `mlp.npz`) in place of the ridge. Everything else is the same as its ridge twin.

| | |
|---|---|
| Zip | `label_update_mlp.zip` (17.9 MB; files unpacked in `files/`, see the note below) |
| Uses labels? | Yes (Fisher-information `labeling.py`, identical in every label submission) |
| Difficulty model | MLP ensemble (h64 and h256x256 x 3 seeds), final fit; `core.mlp_b` runs it in numpy |
| Settings | sigma_delta 0.624, sigma_zeta 1.110 (sigmas re-measured for the MLP with the same rules as the ridge zips) |
| Local 5-fold result | Brier ALC **0.2252** (ridge twin 0.2246) |
| MLP minus ridge twin | +0.0006 +/- 0.0009 (better on 3/5 folds) |
| Codabench | 0.1641 |

## Build and test

```
python pipeline/build_submission.py --variant label_update --name label_update_mlp \
    --ridge_dir data/mlp_b
```

Built and checked by `analysis/mlp_variants.py --model mlp` (the kit's `check_submission_zip.py`, `run_smoke_test.py`,
then the simulation with configuration `label_update`). The ridge twin was rebuilt with the same code
(`analysis/mlp_variants.py --model ridge`) and reproduced the earlier local result exactly. `analysis/test_mlp_submission.py`
checks the numpy MLP against the torch one (max diff 3.8e-7) and the bundled `mlp.npz` against the training predictions
(2.4e-7). Not GPU-checked on Modal.

## Local simulation (150 draws, 1,111 pairs; lower is better)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| this zip | 0.2593 | 0.2443 | 0.2288 | 0.2166 | 0.2071 | 0.2023 | **0.2252** |

Paired comparison with the ridge twin: `PYTHONPATH=analysis python analysis/mlp_variants_report.py`.

Note: `files/` has every file of the zip except `mlp.npz` (33 MB). The same `mlp.npz` (byte-identical in all MLP zips) is in `../../data/mlp_b/versionB_4parts_logitnormal/final/mlp.npz`.
