# Built submissions

Each folder holds one built submission: a `README.md` that explains it in simple English, the `.zip` that is uploaded to Codabench, and `files/` with the same files unzipped. The full story and all results are in `../README.md`.

**Recommended final:** `label_update_stretch1_tuned` (best local ALC with a ridge, 0.2196). **Backup:** `label_update_3improvements_tuned`.

How to read the numbers:

* **Local ALC** comes from our 5-fold simulation (`analysis/simulate_alc.py`): 150 draws, 1,111 subject–benchmark pairs, the same draws for every submission. Lower is better. This is the number we use to choose.
* **Codabench** scores each upload on its own random draw of only about 8 pairs. The same zip uploaded twice can differ by 0.05 or more, so these numbers are noisy and **must not be used to choose or tune**.
* Every submission that uses labels has the same Fisher-information `labeling.py`.

## Ridge submissions

| Folder | One-line summary | Local ALC | Codabench | Status |
|---|---|---|---|---|
| `ap_pure` | AP two-stage IRT: sigmoid(theta - b_hat), labels ignored | 0.2588 | 0.217 | baseline |
| `ap_pure_stretch1` | `ap_pure` with the ridge chosen for within-benchmark ranking (lambda 1e5) | 0.2584 | 0.359 | experiment |
| `ap_pure_stretch2` | `ap_pure` with difficulties stretched by k = 4.05 | 0.2639 | 0.2457 | worse |
| `label_update` | + benchmark shift delta and subject shift zeta learned from labels, Fisher-information label choice | 0.2246 | 0.178 | base label method |
| `label_update_stretch1` | `label_update` with the stretch 1 ridge | 0.2250 | – | experiment |
| `label_update_stretch2` | `label_update` with stretch 2 | 0.2275 | – | worse |
| `label_update_date` | `label_update` + release-date rule for unseen names | 0.2249 | 0.1948 | experiment |
| `label_update_stretch1_date` | stretch 1 + date rule | 0.2254 | 0.1649 | experiment |
| `label_update_stretch2_date` | stretch 2 + date rule | 0.2273 | 0.1596 | worse locally |
| `label_update_3improvements` | + date rule, averaging over uncertainty, item shifts (sigma_item 0.975) | 0.2227 | 0.2099 | replaced by tuned |
| `label_update_3improvements_tuned` | same with sigma_delta 0.912 | 0.2214 | 0.2145 | **backup final** |
| `sim_stretch1` | 3 improvements on the stretch 1 ridge, sigma_delta not doubled (test build) | 0.2209 | – | replaced |
| `sim_stretch2` | 3 improvements with stretch 2 (test build) | 0.2229 | 0.2025 | worse |
| **`label_update_stretch1_tuned`** | tuned method on the stretch 1 ridge | **0.2196** | 0.193 (partial) | **recommended final** |

## MLP submissions (difficulty from the MLP ensemble instead of the ridge)

Each one is the ridge zip in its name with `mlp.npz` (33 MB) in place of `ridge.npz`. "MLP − ridge" is the paired difference to the ridge twin on the same 150 draws (negative = the MLP is better).

| Folder | Local ALC | MLP − ridge | Folds better | Codabench |
|---|---|---|---|---|
| `label_update_mlp` | 0.2252 | +0.0006 ± 0.0009 | 3/5 | 0.1641 |
| `label_update_date_mlp` | 0.2258 | +0.0009 ± 0.0010 | 3/5 | failed (probably a platform error) |
| `label_update_stretch2_mlp` | 0.2232 | −0.0042 ± 0.0009 | 5/5 | 0.2087 |
| `label_update_stretch2_date_mlp` | 0.2234 | −0.0039 ± 0.0009 | 5/5 | 0.1839 |
| `label_update_3improvements_mlp` | 0.2212 | −0.0015 ± 0.0006 | 3/5 | not uploaded |
| `label_update_3improvements_tuned_mlp` | 0.2198 | −0.0016 ± 0.0006 | 4/5 | 0.1995 |
| `sim_stretch2_mlp` | 0.2211 | −0.0018 ± 0.0006 | 5/5 | 0.2106 |

The best MLP zip (`label_update_3improvements_tuned_mlp`, 0.2198) ties `label_update_stretch1_tuned` (0.2196), so the recommendation stays with the smaller ridge zip. A stretch 1 version of the MLP was not built (it would need a new "within" MLP, about 4–6 hours of CPU).

The MLP folders keep `mlp.npz` only inside the zip. The same file (byte-identical in all 7 zips) is in `../data/mlp_b/versionB_4parts_logitnormal/final/mlp.npz`.
