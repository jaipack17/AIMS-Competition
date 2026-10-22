"""Assemble a Codabench submission -> submissions/<variant>/ and submissions/<variant>.zip

Code: submission_src/common/ (core.py, ap_embeddings.py) + submission_src/<variant>/ (model.py, labeling.py, ...).
Parameters written next to model.py:
  theta.json  {part: {key: theta}} for name / harness / provider / effort from the IRT fit
              (private "private::" names of blank-name training subjects are left out: they never match a test subject)
  ridge.npz   mean, scale, coef, intercept of the embedding -> difficulty ridge (fast_ridge, float64), from --ridge_dir
  bhat_stretch.json  (only with --stretch) center c and factor k of b = c + k (b_hat - c)
  name_date_prior.json  (only with --name_prior) a, b of theta_name = a + b * years for names unseen in training
  shift_priors.json  sigma_zeta (IRT fit) and sigma_delta = RMS over the 115 held-out benchmarks of the 5 folds of
              (mean final-fit b - mean held-out b_hat), sigma_item = RMS of the within-benchmark error of held-out
              b_hat: the prior scales of the label-update shifts; plus the label_update switches item_shifts and
              average_over_uncertainty (--item_shifts, --average)
The zip holds the files at its root, as tools/check_submission_zip.py requires.

Run from the project root:
  python pipeline/build_submission.py --variant ap_pure [--run versionB_4parts_logitnormal]
  python pipeline/build_submission.py --variant ap_pure --name ap_pure_stretch1 --ridge_dir data/ridge_within
  python pipeline/build_submission.py --variant ap_pure --name ap_pure_stretch2       --stretch data/ridge/versionB_4parts_logitnormal/stretch.json
then: python "Starting Kit/tools/check_submission_zip.py" submissions/ap_pure.zip
      python "Starting Kit/tools/run_smoke_test.py" submissions/ap_pure
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

PARTS = ["name", "harness", "provider", "effort"]


def shift_priors(run: str, ridge_dir: str, k: float = 1.0) -> dict:
    """sigma_item = RMS over the held-out items of the 5 folds of the within-benchmark error of b_hat (k times the
    held-out prediction, both centred within the benchmark): the prior scale of the label-update item shift."""
    final_b = pd.read_parquet(os.path.join("data", "irt", run, "final", "items.parquet"), columns=["item", "b"])
    gaps, within = [], []
    for f in range(5):
        p = pd.read_parquet(os.path.join(ridge_dir, run, f"fold{f}", "pred.parquet"))
        p = p[p.heldout].merge(final_b, on="item")
        g = p.groupby("benchmark")[["b", "b_pred"]].mean()
        gaps.append(g.b - g.b_pred)
        centred = p[["b", "b_pred"]] - p.groupby("benchmark")[["b", "b_pred"]].transform("mean")
        within.append(centred.b - k * centred.b_pred)
    gaps, within = pd.concat(gaps), pd.concat(within)
    with open(os.path.join("data", "irt", run, "final", "hyper.json")) as f:
        sigma_zeta = json.load(f)["sigma_zeta"]
    return {"sigma_delta": float(np.sqrt(np.mean(gaps ** 2))), "sigma_zeta": float(sigma_zeta),
            "sigma_item": float(np.sqrt(np.mean(within ** 2))), "n_benchmarks_for_sigma_delta": int(len(gaps)),
            "run": run, "ridge_dir": ridge_dir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run", default="versionB_4parts_logitnormal")
    ap.add_argument("--name", default="", help="output name (default: the variant name)")
    ap.add_argument("--ridge_dir", default="data/ridge", help="which ridge to bundle (e.g. data/ridge_within)")
    ap.add_argument("--stretch", default="", help="stretch.json from pipeline/stretch_factor.py to bundle")
    ap.add_argument("--name_prior", default="", help="name_date_prior.json from pipeline/name_date_prior.py to bundle")
    ap.add_argument("--item_shifts", action="store_true", help="label_update: also fit a shift per labeled item")
    ap.add_argument("--average", action="store_true",
                    help="label_update: average the prediction over the remaining uncertainty")
    ap.add_argument("--sigma_item_scale", type=float, default=1.0,
                    help="multiply the measured sigma_item by this factor (tuned with analysis/simulate_alc.py)")
    ap.add_argument("--sigma_delta_scale", type=float, default=1.0,
                    help="multiply the measured sigma_delta by this factor (tuned with analysis/simulate_alc.py)")
    args = ap.parse_args()

    src = os.path.join("submission_src", args.variant)
    out = os.path.join("submissions", args.name or args.variant)
    shutil.rmtree(out, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "bhat_cache", "*.pyc")
    shutil.copytree(os.path.join("submission_src", "common"), out, ignore=ignore)
    shutil.copytree(src, out, ignore=ignore, dirs_exist_ok=True)

    theta = {}
    for p in PARTS:
        t = pd.read_parquet(os.path.join("data", "irt", args.run, "final", f"theta_{p}.parquet"))
        t = t[~t.key.str.startswith("private::")]
        theta[p] = {k: float(v) for k, v in zip(t.key, t.theta)}
    with open(os.path.join(out, "theta.json"), "w", encoding="utf-8") as f:
        json.dump(theta, f, ensure_ascii=False)

    m = np.load(os.path.join(args.ridge_dir, args.run, "final", "model.npz"))
    np.savez(os.path.join(out, "ridge.npz"), mean=m["mean"], scale=m["scale"], coef=m["coef"],
             intercept=m["intercept"], alpha=m["alpha"])

    st = {}
    if args.stretch:
        with open(args.stretch) as f:
            st = json.load(f)
    priors = shift_priors(args.run, args.ridge_dir, st.get("k", 1.0))
    priors.update(item_shifts=args.item_shifts, average_over_uncertainty=args.average,
                  sigma_item_measured=priors["sigma_item"], sigma_item_scale=args.sigma_item_scale,
                  sigma_item=priors["sigma_item"] * args.sigma_item_scale)
    stretch = f"; item shifts {args.item_shifts}, averaging {args.average}, sigma_item {priors['sigma_item']:.3f}"
    if args.stretch:
        priors.update(sigma_delta=st["sigma_delta_stretched"], sigma_delta_source=args.stretch)
        with open(os.path.join(out, "bhat_stretch.json"), "w") as f:
            json.dump({"center": st["center"], "k": st["k"], "source": args.stretch}, f, indent=1)
        stretch += f"; stretch k={st['k']:.3f}"
    if args.sigma_delta_scale != 1.0:
        priors.update(sigma_delta_measured=priors["sigma_delta"], sigma_delta_scale=args.sigma_delta_scale,
                      sigma_delta=priors["sigma_delta"] * args.sigma_delta_scale)
        stretch += f"; sigma_delta {priors['sigma_delta']:.3f}"
    with open(os.path.join(out, "shift_priors.json"), "w") as f:
        json.dump(priors, f, indent=1)
    if args.name_prior:
        with open(args.name_prior) as f:
            npr = json.load(f)
        with open(os.path.join(out, "name_date_prior.json"), "w") as f:
            json.dump({k: npr[k] for k in ("a", "b", "years_min", "years_max", "loo_rmse_line", "loo_rmse_zero")}
                      | {"source": args.name_prior}, f, indent=1)
        stretch += f"; unseen-name date prior {npr['a']:.3f} + {npr['b']:.3f} * years"

    zpath = out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(out)):
            z.write(os.path.join(out, name), arcname=name)
    print(f"{zpath}: {sorted(os.listdir(out))}; theta keys " + ", ".join(f"{p} {len(theta[p])}" for p in PARTS)
          + f"; ridge {args.ridge_dir} alpha {float(m['alpha']):g}{stretch}; {os.path.getsize(zpath) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
