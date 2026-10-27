"""Ceilings for the local simulation: what Brier ALC would we get with perfect item difficulties?

Same draws, splits and labels as analysis/simulate_alc.py (same seeds), on the same held-out benchmarks.
Three kinds of rows:
  real       b_hat = fold f's ridge prediction (what the submission really has; matches simulate_alc.py)
  oracle_b   b_hat replaced by the IRT difficulty b of the FINAL fit, which was trained on these benchmarks'
             own responses (including the evaluation items). Theta still comes from fold f's fit. The submission's
             model.py / labeling.py run unchanged, so its priors (tuned for ridge errors) are kept.
  irt_fit    no prediction at all: sigmoid(theta + zeta(pair) - b) with theta, zeta and b all from the FINAL fit,
             i.e. the IRT model scoring the very responses it was fitted on (labels are not needed and not used).
"oracle" rows are not reachable at test time: b and zeta were fitted on the evaluation responses themselves.
They show how much of the error is due to item difficulty, and how far the model family can go.

Run from the project root:  python analysis/oracle_alc.py --sub label_update_3improvements_tuned --draws 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_alc as sa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="label_update_3improvements_tuned")
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, model, labeling, core = sa.load_submission(args.sub)
    final_dir = os.path.join("data", "irt", sa.RUN, "final")
    b_final = pd.read_parquet(os.path.join(final_dir, "items.parquet"), columns=["item", "b"])
    b_final = dict(zip(b_final["item"], b_final.b.astype(float)))
    zeta = pd.read_parquet(os.path.join(final_dir, "zeta.parquet"))
    zeta = dict(zip(zeta.pair, zeta.zeta.astype(float)))
    theta_final = {}
    for p in sa.PARTS:
        t = pd.read_parquet(os.path.join(final_dir, f"theta_{p}.parquet"))
        theta_final[p] = dict(zip(t.key, t.theta.astype(float)))   # private:: names kept: they are the pair's own
    subj = pd.read_parquet(os.path.join("data", "irt", "subjects.parquet")).set_index("pair")

    def theta_of_pair(pair: str) -> float:  # the fitted parts of this training subject (normalised keys)
        s = subj.loc[pair]
        keys = {"name": s["name"], "harness": s.harness_n, "provider": s.provider_n, "effort": s.effort_n}
        return float(sum(theta_final[p].get(k, 0.0) for p, k in keys.items() if k))

    configs = {"ap_pure": sa.CONFIGS["ap_pure"], "label_update": sa.CONFIGS["label_update"], "all3": sa.CONFIGS["all3"]}
    rows, t0 = [], time.time()
    for fold in range(5):
        data = sa.fold_data(fold, "data/ridge", None)
        core.THETA = data["theta"]
        rng = np.random.default_rng([args.seed, fold])
        for r in range(args.draws):
            pairs = sa.draw(data, rng, 7, 150)
            for kind, bmap in (("real", data["bhat"]), ("oracle_b", b_final)):
                core.item_b_hat = lambda item, _b=bmap: _b[item["item_content"]]
                for name, config in configs.items():
                    B = sa.run_config(pairs, config, model, labeling, core, data["name_prior"])
                    rows += [{"fold": fold, "draw": r, "pair": p["id"], "config": f"{kind}:{name}",
                              **{f"b{bud}": v for bud, v in zip(sa.BUDGETS, b)}} for p, b in zip(pairs, B)]
            for p in pairs:  # the IRT fit on its own training responses
                z = theta_of_pair(p["id"]) + zeta.get(p["id"], 0.0) - np.array([b_final[i] for i in p["eval"]])
                q = 1 / (1 + np.exp(-z))
                brier = float(np.sum(p["k"] * (1 - q) ** 2 + (p["n"] - p["k"]) * q ** 2) / np.sum(p["n"]))
                rows.append({"fold": fold, "draw": r, "pair": p["id"], "config": "irt_fit",
                             **{f"b{bud}": brier for bud in sa.BUDGETS}})
            print(f"fold {fold} draw {r}: {len(pairs)} pairs ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R["alc"] = R[[f"b{b}" for b in sa.BUDGETS]].to_numpy() @ sa.WEIGHTS
    out = os.path.join("analysis", f"oracle_alc_{args.sub}.csv")
    R.to_csv(out, index=False)
    per_draw = R.groupby(["config", "fold", "draw"]).alc.mean().groupby("config").mean()
    by_budget = R.groupby("config")[[f"b{b}" for b in sa.BUDGETS]].mean()
    print(f"\n{R.pair.nunique()} pairs; mean Brier per budget and ALC (lower is better)")
    for name in by_budget.index:
        print(f"{name:22s} " + " ".join(f"{v:.4f}" for v in by_budget.loc[name]) + f" | ALC {per_draw[name]:.4f}")
    print("budgets:               " + " ".join(f"{b:>6}" for b in sa.BUDGETS))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
