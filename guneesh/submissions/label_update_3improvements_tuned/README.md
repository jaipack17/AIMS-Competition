# Submission `label_update_3improvements_tuned` (current recommended upload)

**In one sentence:** exactly `label_update_3improvements`, but with the benchmark-shift prior `sigma_delta` doubled from 0.456 to **0.912**, which the local tuning showed is better.

| | |
|---|---|
| Zip | `label_update_3improvements_tuned.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | **Byte-identical** to `label_update_3improvements` (`model.py`, `labeling.py`, `core.py`, `theta.json`, `ridge.npz`, `name_date_prior.json`). Only `shift_priors.json` differs. |
| Settings | **sigma_delta = 0.912** (2 × 0.456), sigma_zeta = 1.110, sigma_item = 0.975, item shifts on, averaging on, date rule on |
| Codabench result | Not known at the time of writing (Codabench was down). |
| Local 5-fold result | Brier ALC **0.2214** (best submission that is already built) |
| Status | **The zip to upload now.** A slightly better candidate (stretch 1 ridge, 0.2196) has been found but not yet built as a zip. See below. |

---

## 1. What changed and why

Everything is explained in `label_update_3improvements/README.md`. The only change is one number:

```
sigma_delta: 0.456  ->  0.912
```

`sigma_delta` is the prior spread of `delta`, the shift of a whole benchmark's difficulty. A larger value lets the labels move the benchmark's level further and faster, and makes the averaging more cautious before any labels arrive.

## 2. How the value was chosen

With `sigma_item` fixed at 0.975, we tried every combination of:

* sigma_zeta scales: 0.5, 0.75, 1.0, 1.25, 1.5
* sigma_delta scales: 0.5, 0.75, 1.0, 1.5, 2.0 (and 2.5, 3.0, 4.0 for sigma_zeta scales 1.0 and 1.25)

in the local simulation (all on the same 150 draws, 1,111 pairs). Results files: `analysis/simulate_alc_label_update_3improvements_z<zeta scale>_d<delta scale>.csv`.

Findings:

* **sigma_delta should be larger than measured.** Most of the gain is reached at 2× the measured value. Above 2× the curve is flat and noisier.
* **sigma_zeta is already right.** Smaller values are clearly worse; larger values give no consistent gain.
* sigma_delta = 2 × 0.456 and sigma_zeta = 1 × 1.110 beats `label_update_3improvements` on **all 5 folds**.

A likely reason: the measured `sigma_delta` only counts the error of the ridge's guess of the benchmark's mean difficulty. On a new benchmark, the overall level of the subjects can also be off, and that error is shared by all subjects on the benchmark, so it behaves like `delta` too.

## 3. How it was built and tested

```
python pipeline/build_submission.py --variant label_update --name label_update_3improvements_tuned \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
    --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0
python analysis/test_label_update.py label_update_3improvements_tuned
python analysis/test_name_date_prior.py label_update_3improvements_tuned
python ../starting_kit/tools/check_submission_zip.py submissions/label_update_3improvements_tuned.zip
python ../starting_kit/tools/run_smoke_test.py submissions/label_update_3improvements_tuned
python analysis/simulate_alc.py --sub label_update_3improvements_tuned --draws 30 --configs all3 --tag _final
```

All checks passed, and the simulation of this exact zip reproduced 0.2214. No new GPU run was needed, because every file except `shift_priors.json` is byte-identical to `label_update_3improvements`, which passed the Modal A100 check.

## 4. Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `ap_pure` | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2594 | 0.2588 |
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| `label_update_3improvements` | 0.2525 | 0.2403 | 0.2271 | 0.2151 | 0.2059 | 0.1996 | 0.2227 |
| **`label_update_3improvements_tuned`** | 0.2517 | 0.2382 | 0.2247 | 0.2144 | 0.2051 | 0.1994 | **0.2214** |

## 5. Same method on the other ridges (found later, not built as zips)

We later ran the same method and settings on the two stretched ridges (test builds `sim_stretch1`, `sim_stretch2`; results in `analysis/simulate_alc_sim_stretch*.csv`):

| Ridge | 0 | 1 | 3 | 7 | 15 | 31 | ALC | Folds better than this zip |
|---|---|---|---|---|---|---|---|---|
| original (this zip) | 0.2517 | 0.2382 | 0.2247 | 0.2144 | 0.2051 | 0.1994 | 0.2214 | – |
| **stretch 1** | 0.2489 | 0.2359 | 0.2232 | 0.2127 | 0.2041 | 0.1976 | **0.2196** | 4 of 5 |
| stretch 2 | 0.2517 | 0.2367 | 0.2256 | 0.2144 | 0.2058 | 0.2000 | 0.2214 | 3 of 5 |

**Stretch 1 is 0.0018 ± 0.0005 better and wins at every budget**, but only on 4 of 5 folds, and its `sigma_item` has not been re-tuned for that ridge yet. To build it as a zip:

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch1_tuned --ridge_dir data/ridge_within \
    --name_prior data/irt/versionB_4parts_logitnormal/final/name_date_prior.json \
    --item_shifts --average --sigma_item_scale 0.6 --sigma_delta_scale 2.0
```

Note: with `--ridge_dir data/ridge_within`, the measured sigma_delta is 0.587 and sigma_item is 1.618 before scaling, so this gives sigma_delta = 1.173 and sigma_item = 0.971. The simulation result above used exactly these values.
