"""Local copy of the competition's streaming evaluation (Brier ALC) on our held-out benchmarks.

For each fold f the 23 binary benchmarks held out of fold f's IRT fit and ridge are the "test" benchmarks:
theta comes from fold f's IRT fit, b_hat from fold f's ridge (never trained on them), and the real recorded
responses (data/irt/cells_binary.parquet) are the outcomes. The submission's own model.py and labeling.py are
run (smoke-test import, no GPU), with core.item_b_hat replaced by a lookup of the stored held-out b_hat and
core.THETA / core.NAME_PRIOR replaced by fold f's values.

One draw, as described in the starting kit README:
  * pick N_BENCH benchmarks of the fold (the test pool seen on Codabench has about 7) and keep a random subset of
    BENCH_ITEMS items of each (Codabench pairs have ~90-150 items; same subset for all subjects), then (benchmark, subject)
    pairs with at least 80 distinct items, greedily up to 1,000 distinct items, preferring benchmarks and
    subjects represented less often in the draw (random tie-break);
  * each pair: random 50/50 split of its items (odd count: extra item to acquisition), acquisition candidates
    in random order; a revealed label is one of the item's recorded runs (1 with probability k / n);
  * caps 0, 1, 3, 7, 15, 31: before each cap every pair continues its candidate stream until it holds `cap`
    labels or runs out; the hook sees the prediction with all labels so far and the README's context;
  * each cap: Brier over all recorded runs of each pair's evaluation items; ALC = 0.1/.2/.2/.2/.2/.1 weights.
Every configuration sees the same draws, splits, candidate orders and label outcomes, so differences are paired.

Configurations (switches of the submission's model.py / core.py):
  ap_pure        no labels used, no date prior
  label_update   delta + zeta (the submission already uploaded)
  +date          plus the release-date prior for unseen names
  +item          plus item shifts
  +avg           plus averaging over uncertainty
  all3           date + item + avg (= label_update_3improvements)
The priors sigma_delta / sigma_item are those bundled in the submission (measured over all 5 folds, so very
slightly optimistic for the fold being scored).

Run from the project root:  python analysis/simulate_alc.py --sub label_update_3improvements --draws 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
RUN = "versionB_4parts_logitnormal"
BUDGETS, WEIGHTS = [0, 1, 3, 7, 15, 31], np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MAX_LABELS, DRAW_ITEMS, MIN_ITEMS = 31, 1000, 80
FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort",
          "harness_version", "subject_features_extra"]
PARTS = ["name", "harness", "provider", "effort"]
#                 labels, date,  item,  avg
CONFIGS = {"ap_pure":      (False, False, False, False),
           "label_update": (True, False, False, False),
           "+date":        (True, True, False, False),
           "+item":        (True, False, True, False),
           "+avg":         (True, False, False, True),
           "date+item":    (True, True, True, False),
           "date+avg":     (True, True, False, True),
           "item+avg":     (True, False, True, True),
           "all3":         (True, True, True, True)}


def load_submission(name: str):
    d = os.path.abspath(os.path.join("submissions", name))
    sys.path.insert(0, d)
    mods = []
    for module in ("model", "labeling"):
        spec = importlib.util.spec_from_file_location(f"sub_{module}", os.path.join(d, f"{module}.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods.append(m)
    return d, mods[0], mods[1], sys.modules["core"]


def fold_data(fold: int, ridge_dir: str, stretch: dict | None) -> dict:
    folds = pd.read_csv(os.path.join("data", "irt", "folds.csv"))
    benches = sorted(folds.benchmark[folds.fold == fold])
    fdir = os.path.join("data", "irt", RUN, f"fold{fold}")
    theta = {}
    for p in PARTS:
        t = pd.read_parquet(os.path.join(fdir, f"theta_{p}.parquet"))
        t = t[~t.key.str.startswith("private::")]
        theta[p] = dict(zip(t.key, t.theta.astype(float)))
    with open(os.path.join(fdir, "name_date_prior.json")) as f:
        name_prior = json.load(f)
    pred = pd.read_parquet(os.path.join(ridge_dir, RUN, f"fold{fold}", "pred.parquet"))
    pred = pred[pred.heldout]
    b = pred.b_pred.to_numpy(dtype=float)
    if stretch is not None:  # b = c + k (b - c) around this fold's training mean (its ridge intercept)
        c = float(np.load(os.path.join(ridge_dir, RUN, f"fold{fold}", "model.npz"))["intercept"])
        b = c + stretch["k"] * (b - c)
    cells = pq.read_table(os.path.join("data", "irt", "cells_binary.parquet"), columns=["pair", "item", "n", "k"],
                          filters=[("benchmark", "in", benches)]).to_pandas()
    cells = cells[cells.item.isin(pred["item"])]  # only embedded items have a b_hat
    subj = pd.read_parquet(os.path.join("data", "irt", "subjects.parquet")).set_index("pair")
    size = cells.groupby("pair").item.nunique()
    pairs = {}
    for pair, g in cells[cells.pair.isin(size.index[size >= MIN_ITEMS])].groupby("pair", sort=True):
        s = subj.loc[pair]
        pairs[pair] = {"benchmark": s.benchmark, "subject": {f: str(s[f]) for f in FIELDS},
                       "items": g.item.to_numpy(), "n": g.n.to_numpy(dtype=float), "k": g.k.to_numpy(dtype=float)}
    return {"theta": theta, "name_prior": name_prior, "bhat": dict(zip(pred["item"], b)), "pairs": pairs}


def draw(data: dict, rng: np.random.Generator, n_bench: int, bench_items: int) -> list[dict]:
    by_bench = sorted({p["benchmark"] for p in data["pairs"].values()})
    chosen_b = sorted(rng.choice(by_bench, size=min(n_bench, len(by_bench)), replace=False))
    # Codabench pairs have ~90-150 items, ours often thousands: keep a random subset of each chosen benchmark's
    # items, the same subset for all its subjects (so subjects of one benchmark share items, as on Codabench)
    pairs = {}
    for b in chosen_b:
        ks = [k for k, p in data["pairs"].items() if p["benchmark"] == b]
        all_items = np.unique(np.concatenate([data["pairs"][k]["items"] for k in ks]))
        keep = set(rng.choice(all_items, size=min(bench_items, len(all_items)), replace=False))
        for k in ks:
            p = data["pairs"][k]
            m = np.array([it in keep for it in p["items"]])
            if m.sum() >= MIN_ITEMS:
                pairs[k] = {**p, "items": p["items"][m], "n": p["n"][m], "k": p["k"][m]}
    data = {"pairs": pairs}
    cand = sorted(pairs)
    order = {k: i for i, k in enumerate(rng.permutation(cand))}
    bcount, scount, used, out = Counter(), Counter(), 0, []
    while True:
        fits = [k for k in cand if len(data["pairs"][k]["items"]) <= DRAW_ITEMS - used]
        if not fits:
            break
        key = lambda k: (bcount[data["pairs"][k]["benchmark"]],
                         scount[json.dumps(data["pairs"][k]["subject"], sort_keys=True)], order[k])
        k = min(fits, key=key)
        p = data["pairs"][k]
        cand.remove(k)
        used += len(p["items"])
        bcount[p["benchmark"]] += 1
        scount[json.dumps(p["subject"], sort_keys=True)] += 1
        perm = rng.permutation(len(p["items"]))
        n_acq = math.ceil(len(perm) / 2)
        acq, ev = perm[:n_acq], perm[n_acq:]
        labels = (rng.random(len(acq)) < p["k"][acq] / p["n"][acq]).astype(int)
        out.append({"id": k, "benchmark": p["benchmark"], "subject": p["subject"],
                    "acq": [(p["items"][i], int(y)) for i, y in zip(acq, labels)],
                    "eval": p["items"][ev], "n": p["n"][ev], "k": p["k"][ev]})
    return out


def item_dict(item_id: str, benchmark: str) -> dict:
    return {"item_content": item_id, "item_features": "", "interactors": "", "benchmark_id": benchmark}


def run_config(pairs: list[dict], config: tuple, model, labeling, core, name_prior: dict) -> np.ndarray:
    """Brier score of each pair at each budget (pairs x 6)."""
    use_labels, date, item_shifts, avg = config
    core.NAME_PRIOR = name_prior if date else None
    model.ITEM_SHIFTS, model.AVERAGE = item_shifts, avg
    model._SHIFT_MEMO.clear()
    labeled, acquired, pos = [], [0] * len(pairs), [0] * len(pairs)
    brier = np.zeros((len(pairs), len(BUDGETS)))
    for c, cap in enumerate(BUDGETS):
        if use_labels:
            for i, p in enumerate(pairs):
                while acquired[i] < cap and pos[i] < len(p["acq"]):
                    item_id, y = p["acq"][pos[i]]
                    inp = [p["subject"], item_dict(item_id, p["benchmark"])]
                    ctx = {"subject_id": p["id"], "benchmark_id": p["benchmark"],
                           "labels_remaining": MAX_LABELS - acquired[i], "labels_acquired": acquired[i],
                           "max_labels": MAX_LABELS, "items_remaining": len(p["acq"]) - pos[i]}
                    pos[i] += 1
                    if labeling.acquisition_function(inp, model.predict(inp, labeled), labeled, ctx):
                        labeled.append([inp, y])
                        acquired[i] += 1
        for i, p in enumerate(pairs):
            q = np.array([model.predict([p["subject"], item_dict(it, p["benchmark"])], labeled) for it in p["eval"]])
            brier[i, c] = float(np.sum(p["k"] * (1 - q) ** 2 + (p["n"] - p["k"]) * q ** 2) / np.sum(p["n"]))
    return brier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="label_update_3improvements")
    ap.add_argument("--draws", type=int, default=10, help="draws per fold")
    ap.add_argument("--n_bench", type=int, default=7)
    ap.add_argument("--bench_items", type=int, default=150, help="items kept per chosen benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="", help="comma-separated subset of CONFIGS (default: all)")
    ap.add_argument("--tag", default="", help="suffix of the output csv")
    ap.add_argument("--sigma_item_scale", type=float, default=1.0,
                    help="multiply the bundled sigma_item (item-shift prior and averaging) by this factor")
    ap.add_argument("--sigma_zeta_scale", type=float, default=1.0, help="multiply the bundled sigma_zeta")
    ap.add_argument("--sigma_delta_scale", type=float, default=1.0, help="multiply the bundled sigma_delta")
    args = ap.parse_args()

    configs = {k: CONFIGS[k] for k in args.configs.split(",")} if args.configs else CONFIGS
    d, model, labeling, core = load_submission(args.sub)
    model.SIGMA_ITEM *= args.sigma_item_scale
    model.SIGMA_ZETA *= args.sigma_zeta_scale
    model.SIGMA_DELTA *= args.sigma_delta_scale
    with open(os.path.join(d, "shift_priors.json")) as f:
        ridge_dir = json.load(f).get("ridge_dir", "data/ridge")
    stretch = None
    if os.path.exists(os.path.join(d, "bhat_stretch.json")):
        with open(os.path.join(d, "bhat_stretch.json")) as f:
            stretch = json.load(f)
    rows, t0 = [], time.time()
    for fold in range(5):
        data = fold_data(fold, ridge_dir, stretch)
        core.THETA = data["theta"]
        core.item_b_hat = lambda item, _b=data["bhat"]: _b[item["item_content"]]
        rng = np.random.default_rng([args.seed, fold])
        for r in range(args.draws):
            pairs = draw(data, rng, args.n_bench, args.bench_items)
            for name, config in configs.items():
                B = run_config(pairs, config, model, labeling, core, data["name_prior"])
                for p, b in zip(pairs, B):
                    rows.append({"fold": fold, "draw": r, "pair": p["id"], "config": name, "alc": float(b @ WEIGHTS),
                                 **{f"b{bud}": v for bud, v in zip(BUDGETS, b)}})
            print(f"fold {fold} draw {r}: {len(pairs)} pairs on {len({p['benchmark'] for p in pairs})} benchmarks "
                  f"({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    out = os.path.join("analysis", f"simulate_alc_{args.sub}{args.tag}.csv")
    R.to_csv(out, index=False)

    # each draw is one "submission": mean over its pairs; compare configurations draw by draw
    per_draw = R.groupby(["config", "fold", "draw"]).alc.mean().unstack("config")
    if "label_update" not in per_draw:
        print(f"saved {out}")
        return
    base = per_draw["label_update"]
    print(f"\n{len(per_draw)} draws, {R[R.config == 'ap_pure'].shape[0]} pairs; mean Brier per budget and ALC "
          f"(lower is better); difference vs label_update with its standard error over draws")
    by_budget = R.groupby("config")[[f"b{b}" for b in BUDGETS]].mean()
    for name in configs:
        diff = per_draw[name] - base
        print(f"{name:13s} " + " ".join(f"{v:.4f}" for v in by_budget.loc[name]) +
              f" | ALC {per_draw[name].mean():.4f}  diff {diff.mean():+.4f} ± {diff.std() / np.sqrt(len(diff)):.4f}")
    print("budgets:      " + " ".join(f"{b:>6}" for b in BUDGETS))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
