# Submission `label_update_stretch1`

**In one sentence:** `label_update` (shifts `delta` and `zeta` learned from labels, Fisher-information label choice) using the **stretch 1** ridge, which was chosen for ranking items inside a benchmark.

| | |
|---|---|
| Zip | `label_update_stretch1.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | Byte-identical to `label_update` (`model.py`, `labeling.py`, `core.py`). Only `ridge.npz` and `shift_priors.json` differ. |
| Codabench result | Uploaded. Still pending when this README was written (Codabench was down). |
| Local 5-fold result | Brier ALC **0.2250** (vs 0.2246 for `label_update`) |
| Status | Experiment. Not better on its own. |

---

## What is different from `label_update`

1. **The ridge.** `ridge.npz` comes from `data/ridge_within`. Its penalty `lambda` = 1e5 was chosen by the within-benchmark R² (`r2_within`) instead of the overall R². See the `ap_pure_stretch1` README for why. Its predictions are less squeezed together.
2. **The prior of `delta`.** `sigma_delta` = **0.587** instead of 0.456. It is measured in the same way (root mean square, over the 115 held-out benchmarks, of true mean difficulty minus predicted mean difficulty), but for this ridge. A less squeezed ridge makes slightly bigger errors in the benchmark mean, so `sigma_delta` is larger.
3. Everything else is the same: `sigma_zeta` = 1.110, the same formula `P = sigmoid((theta + zeta) - (b_hat + delta))`, and the same Fisher-information `labeling.py`.

## How it was built

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch1 --ridge_dir data/ridge_within
```

## Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| **`label_update_stretch1`** | 0.2591 | 0.2456 | 0.2285 | 0.2155 | 0.2066 | 0.2017 | **0.2250** |

It is slightly better at high budgets (7, 15, 31) and slightly worse at budget 1.

## What we learned

The stretch 1 ridge only pays off together with **averaging over uncertainty** (see `label_update_3improvements`). With averaging and the tuned settings, stretch 1 gives our best local score of **0.2196** (see the main README, section "Results").
