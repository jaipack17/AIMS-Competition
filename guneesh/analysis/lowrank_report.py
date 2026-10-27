"""Summarise pipeline/lowrank_dir.py over folds: within-benchmark r on held-out benchmarks vs k and n.

Every number is the mean over held-out benchmarks (each benchmark once, in its own fold) of the r averaged over draws;
penalties were chosen on the selection benchmarks.  "ours@k*" uses the k chosen on the selection benchmarks per fold.
Run from the project root:  python analysis/lowrank_report.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

RUN_DIR = "data/lowrank/versionB_4parts_logitnormal"

held = pd.concat([pd.read_csv(p, dtype={"n": str}) for p in sorted(glob.glob(f"{RUN_DIR}/fold*/heldout.csv"))])
summ = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in sorted(glob.glob(f"{RUN_DIR}/fold*/summary.json"))}
print(f"folds: {sorted(summ)}; held-out benchmarks: {held.benchmark.nunique()}")
print(f"zero-shot (average direction, no labels): r = {held[held.method == 'zero'].r.mean():.3f}\n")

ns = [n for n in ("10", "30", "100", "300", "half") if n in set(held.n)]
ours = held[held.method == "ours"].pivot_table(index="k", columns="n", values="r", aggfunc="mean")[ns]
scratch = held[held.method == "scratch"].groupby("n").r.mean()[ns]
cnt = held[held.method == "scratch"].groupby("n").benchmark.nunique()[ns]
print("ours (shared basis, k numbers fitted from n items) -- mean within-benchmark r")
print(ours.round(3).to_string())
print("\nscratch (ridge on the full embedding, same n items, no sharing)")
print(scratch.round(3).to_string())
print("\nbenchmarks per n:", cnt.to_dict())

print("\nheadline per fold (k chosen on selection benchmarks):")
for f, s in summ.items():
    print(f"  {f}: " + "; ".join(f"n={n}: k={v['k']} ours {v['ours_r']:.3f} vs scratch {v['scratch_r']:.3f}"
                                  for n, v in s["headline"].items()))
star = []
for f, s in summ.items():
    for n, v in s["headline"].items():
        h = held[(held.fold == f) & (held.n == n)]
        o = h[(h.method == "ours") & (h.k == v["k"])].set_index("benchmark").r
        sc = h[h.method == "scratch"].set_index("benchmark").r
        star.append(pd.DataFrame({"n": n, "ours": o, "scratch": sc.reindex(o.index)}))
star = pd.concat(star).dropna()
print("\nours@k* vs scratch, pooled over folds (paired by benchmark):")
for n, g in star.groupby("n"):
    d = g.ours - g.scratch
    print(f"  n={n:>4}: ours {g.ours.mean():.3f}  scratch {g.scratch.mean():.3f}  diff {d.mean():+.3f} "
          f"± {d.std() / np.sqrt(len(d)):.3f}  better {int((d > 0).sum())}/{len(d)}")
evr = np.mean([s["explained_variance_ratio_first10"] for s in summ.values()], 0)
print("\nPCA of the training directions, explained variance of PC1..PC10 (mean over folds):", np.round(evr, 3))
