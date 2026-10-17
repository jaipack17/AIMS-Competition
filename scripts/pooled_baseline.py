"""Shared-evidence baselines on exactly the same pairs/folds as eval_subject_only.py."""
import sys; sys.path.insert(0, ".")
import numpy as np
from paiec import fit, harness
from baselines import simple
T = fit.load_table(); nb = len(T["benches"]); names = [str(b) for b in T["benches"]]
perm = np.random.default_rng(0).permutation(nb)
groups = []
for i in range(5):
    byb = {}
    for b in perm[i::5]:
        for p in harness.make_pairs(names[b], 8, 300): byb.setdefault(p["bench"], []).append(p)
    groups += list(byb.values())
for label, fn in [("pooled mean, strength 2", simple.pooled_mean), ("pooled mean, strength 8", lambda i, l=None: simple.pooled_mean(i, l, 8.0))]:
    rows = [r for g in groups for r in harness.run_group(g, fn)]
    arr = np.array(rows); print(f"{label:26s} ALC {(arr @ harness.WEIGHTS).mean():.4f} | by budget {np.round(arr.mean(0), 4).tolist()} | pairs {len(rows)}")
