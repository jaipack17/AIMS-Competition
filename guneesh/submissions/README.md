# Built submissions

Each folder holds one built submission: a `README.md` that explains it in simple English, the `.zip` that is uploaded to Codabench, and `files/` with the same files unzipped. The full story and all results are in `../README.md`.

| Folder | One-line summary | Local ALC | Codabench | Status |
|---|---|---|---|---|
| `ap_pure` | AP two-stage IRT: sigmoid(theta - b_hat), labels ignored | 0.2588 | ≈ 0.217 | baseline |
| `ap_pure_stretch1` | `ap_pure` with the ridge chosen for within-benchmark ranking (lambda 1e5) | 0.2584 | pending | experiment |
| `ap_pure_stretch2` | `ap_pure` with difficulties stretched by k = 4.05 | 0.2639 | pending | worse |
| `label_update` | + benchmark shift delta and subject shift zeta learned from labels, Fisher-information label choice | 0.2246 | ≈ 0.178 | uploaded |
| `label_update_stretch1` | `label_update` with the stretch 1 ridge | 0.2250 | pending | experiment |
| `label_update_stretch2` | `label_update` with stretch 2 | 0.2275 | pending | worse |
| `label_update_date` | `label_update` + release-date rule for unseen names | 0.2249 | – | experiment |
| `label_update_stretch1_date` | stretch 1 + date rule | 0.2254 | – | experiment |
| `label_update_stretch2_date` | stretch 2 + date rule | 0.2273 | – | worse |
| `label_update_3improvements` | + date rule, averaging over uncertainty, item shifts (sigma_item 0.975) | 0.2227 | – | replaced by tuned |
| **`label_update_3improvements_tuned`** | same with sigma_delta 0.912 | **0.2214** | – | **upload this now** |

Best idea not yet built as a zip: the tuned method with the stretch 1 ridge, local ALC 0.2196 (see `label_update_3improvements_tuned/README.md`, section 5).
