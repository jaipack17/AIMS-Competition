"""Summary of pipeline/qdet_upper_bound.py and pipeline/qdet_combined.py (same benchmarks, split and seeds): per model,
mean within-benchmark Pearson r / Spearman rho / RMSE over the benchmarks where every compared model has a value, and
the paired difference to the embedding baseline (emb_kridge) with its standard error over benchmarks.  Also q vs
full text, and ling_paper vs ling_code on the first benchmarks (ling_code and read alone were dropped after them).
The per-benchmark r of every model is saved to --csv (one row per benchmark, one column per model).
Run from the project root:  python analysis/qdet_report.py [--dir data/qdet/upper_bound] [--combined data/qdet/combined]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DROPPED = ("ling_code[q]", "ling_code[full]", "read[q]", "read[full]")


def load(d: str, rows: list, skipped: set) -> None:
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        res = json.load(open(p))
        if "skipped" in res:
            skipped.add(res["benchmark"])
            continue
        for model, m in res["models"].items():
            row = {"benchmark": res["benchmark"], "model": model, **{k: m.get(k, np.nan) for k in ("r", "rho", "rmse")}}
            if m.get("constant") or ("r" in m and np.isnan(m["rho"])):  # constant prediction (files written before
                row["r"] = np.nan                                         # the "constant" flag: rho is nan, r ~1e-16)
            rows.append(row)


def paired(wide: pd.DataFrame, a: str, b: str) -> str:
    d = (wide[a] - wide[b]).dropna()
    return f"{d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}  ({a} better on {(d > 0).sum()}/{len(d)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/qdet/upper_bound")
    ap.add_argument("--combined", default="data/qdet/combined")
    ap.add_argument("--csv", default="analysis/qdet_r_all.csv")
    args = ap.parse_args()
    rows, skipped = [], set()
    load(args.dir, rows, skipped)
    load(args.combined, rows, skipped)
    R = pd.DataFrame(rows)
    wide = {k: R.pivot_table(index="benchmark", columns="model", values=k, dropna=False) for k in ("r", "rho", "rmse")}
    main_models = [m for m in wide["r"].columns if m not in DROPPED and m != "emb_kridge_check"]
    have_all = wide["r"][main_models].notna().all(axis=1)   # missing r = constant prediction, failed fit or not run yet
    common = have_all[have_all].index
    print(f"{R.benchmark.nunique()} benchmarks with results ({len(skipped)} skipped: {', '.join(sorted(skipped))}); "
          f"{len(common)} with a value for every model below\n")
    if len(common):
        base = wide["r"].loc[common, "emb_kridge"]
        out = []
        for model in main_models:
            d = wide["r"].loc[common, model] - base
            out.append({"model": model, "r": wide["r"].loc[common, model].mean(),
                        "rho": wide["rho"].loc[common, model].mean(), "rmse": wide["rmse"].loc[common, model].mean(),
                        "r - emb": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                        "better than emb": f"{int((d > 0).sum())}/{len(d)}",
                        "r (all with value)": wide["r"][model].mean(), "n": int(wide["r"][model].notna().sum())})
        print(pd.DataFrame(out).sort_values("r", ascending=False).round(3).to_string(index=False))
    W = wide["r"]
    W.round(3).to_csv(args.csv)
    print(f"\nper-benchmark r of every model saved to {args.csv}")
    if "emb_kridge_check" in W:
        d = (W["emb_kridge_check"] - W["emb_kridge"]).abs().dropna()
        print(f"\nsetup check: emb_kridge recomputed by qdet_combined vs qdet_upper_bound, max |diff r| "
              f"{d.max():.1e} over {len(d)} benchmarks")
        print("\ncombined models vs the embedding, over every benchmark qdet_combined ran (same/better/worse: |diff| "
              "< 0.005 counts as same):")
        for model in ("joint[q]", "joint[full]", "correction[q]", "correction[full]"):
            d = (W[model] - W["emb_kridge_check"]).dropna()
            print(f"  {model:17s} emb r {W['emb_kridge_check'][d.index].mean():.3f} -> {W[model][d.index].mean():.3f}"
                  f"  diff {d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}  better/same/worse "
                  f"{(d >= 0.005).sum()}/{(d.abs() < 0.005).sum()}/{(d <= -0.005).sum()} of {len(d)}")
    print("\nq vs full text (r, paired over the benchmarks with both):")
    for g in ("ling_paper", "ling_paper+read", "joint", "correction"):
        if f"{g}[q]" in W and f"{g}[full]" in W:
            print(f"  {g:16s} {paired(W, f'{g}[full]', f'{g}[q]')}")
    if "ling_code[q]" in W:
        print(f"\nfirst benchmarks only: ling_paper[q] vs ling_code[q]: {paired(W, 'ling_paper[q]', 'ling_code[q]')}")


if __name__ == "__main__":
    main()
