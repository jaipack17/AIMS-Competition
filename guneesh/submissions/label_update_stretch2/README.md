# Submission `label_update_stretch2`

**In one sentence:** `label_update` (shifts `delta` and `zeta` learned from labels, Fisher-information label choice) with the **stretch 2** difficulties: `b = c + 4.05 * (b_hat - c)`.

| | |
|---|---|
| Zip | `label_update_stretch2.zip` (files unpacked in `files/`) |
| Uses labels? | Yes |
| Code | Byte-identical to `label_update`. It adds `bhat_stretch.json`, which `core.py` applies whenever the file is present. |
| Codabench result | Uploaded. Still pending when this README was written (Codabench was down). |
| Local 5-fold result | Brier ALC **0.2275** (worse than `label_update`, 0.2246) |
| Status | Experiment. Worse. Do not use. |

---

## What is different from `label_update`

1. **Stretched difficulties.** `bhat_stretch.json` holds `center c = 0.01462` and `k = 4.0502`. `core.py` computes `b = c + k * (b_hat - c)`. See the `ap_pure_stretch2` README for how `k` was found.
2. **The prior of `delta`.** `sigma_delta` = **0.570**, measured for the stretched predictions by `pipeline/stretch_factor.py` (field `sigma_delta_stretched` in `stretch.json`).
3. Everything else is the same as `label_update`.

## How it was built

```
python pipeline/build_submission.py --variant label_update --name label_update_stretch2 --stretch data/ridge/versionB_4parts_logitnormal/stretch.json
```

## Scores (local 5-fold simulation)

| Budget | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---|---|---|---|---|---|
| `label_update` | 0.2594 | 0.2424 | 0.2280 | 0.2160 | 0.2072 | 0.2026 | 0.2246 |
| **`label_update_stretch2`** | 0.2647 | 0.2463 | 0.2312 | 0.2180 | 0.2095 | 0.2044 | **0.2275** |

It is worse at every budget.

## What we learned

Multiplying the difficulties by 4 makes predictions too confident. Even with all improvements and tuning, stretch 2 only ties the original ridge (0.2214). We dropped this direction.
