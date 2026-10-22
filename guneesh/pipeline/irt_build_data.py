"""Build the (grouped) response tables for the IRT fits, once, and save them to data/irt/.

Benchmarks: analysis/train_benchmarks.csv (152 = 115 binary + 37 non-binary).
Each row is one (benchmark, subject, item) cell; repeated attempts are kept as counts, which gives exactly
the same likelihood as one row per attempt:
  binary  : n attempts, k correct                      -> Binomial(n, sigmoid(mu)) == product of n Bernoullis
  non-bin : y01 = (x - min) / (max - min) clipped to [0, 1]; n, sum(y01), and for the logit-normal model
            sum(z), sum(z^2) with z = logit(clip(y01, eps, 1 - eps))  (sufficient statistics of the Normal)

Subject parts (all from fields the test input also has):
  name     = normalized_name (blank -> private id "<benchmark>::<subject_id>", never matches anything else)
  harness  = harness, lowercased, spaces/underscores -> "-", trailing version numbers removed, codex variants merged
  provider = provider, lowercased (blank / "unknown" -> no provider)
  effort   = reasoning_effort, lowercased (blank -> no effort)
  pair     = "<benchmark>::<subject_id>" for the per-benchmark ability deviation (LEGO-MB zeta)
Folds: the 115 binary benchmarks are split into 5 folds at random (seed 0); non-binary benchmarks are never held out.

Outputs (data/irt/): cells_binary.parquet, cells_cont.parquet, subjects.parquet, folds.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "data/measurement-db"
OUT = "data/irt"
EPS = 0.01  # clip for the logit-normal (LEGO) likelihood; scores of exactly 0 or 1 become logit(0.01), logit(0.99)
HARNESS_MERGE = {"openai-codex": "codex", "codex-cli": "codex"}


def norm_harness(h: str) -> str:
    h = re.sub(r"[\s_]+", "-", h.strip().lower())
    h = re.sub(r"-v?\d+(\.\d+)+$", "", h)  # openhands-0.14.2 -> openhands
    return HARNESS_MERGE.get(h, h)


def main() -> None:
    import os

    os.makedirs(OUT, exist_ok=True)
    tb = pd.read_csv("analysis/train_benchmarks.csv")
    subj, cb, cc = [], [], []
    for _, r in tb.iterrows():
        b = r.benchmark
        s = pd.read_parquet(f"{ROOT}/{b}/formatted_tables/subjects.parquet").fillna("")
        s["benchmark"] = b
        subj.append(s)
        x = pq.read_table(f"{ROOT}/{b}/formatted_tables/responses.parquet",
                          columns=["subject_id", "item_id", "response"]).to_pandas()
        x = x[x.response.notna()]
        x["item"] = b + "::" + x.item_id.astype(str)
        x["pair"] = b + "::" + x.subject_id.astype(str)
        if r.response_type == "binary":
            assert x.response.isin([0.0, 1.0]).all(), b
            g = x.groupby(["pair", "item"], sort=False).response.agg(["size", "sum"]).reset_index()
            g.columns = ["pair", "item", "n", "k"]
            cb.append(g)
        else:
            y = ((x.response - r["min"]) / (r["max"] - r["min"])).clip(0, 1)
            z = np.log(y.clip(EPS, 1 - EPS)) - np.log1p(-y.clip(EPS, 1 - EPS))
            x = x.assign(y=y, z=z, z2=z * z)
            g = x.groupby(["pair", "item"], sort=False).agg(n=("y", "size"), sy=("y", "sum"),
                                                            sz=("z", "sum"), sz2=("z2", "sum")).reset_index()
            cc.append(g)
        print(f"{b}: {len(x):,} responses", flush=True)

    S = pd.concat(subj, ignore_index=True)
    S["pair"] = S.benchmark + "::" + S.subject_id.astype(str)
    nm = S.normalized_name.str.strip()
    S["name"] = np.where(nm == "", "private::" + S.pair, nm)
    S["harness_n"] = S.harness.map(norm_harness)
    S["provider_n"] = S.provider.map(lambda v: "" if v.strip().lower() in ("", "unknown") else re.sub(r"\s+", " ", v.strip().lower()))
    S["effort_n"] = S.reasoning_effort.map(lambda v: re.sub(r"\s+", " ", v.strip().lower()))
    S.to_parquet(f"{OUT}/subjects.parquet")

    B = pd.concat(cb, ignore_index=True)
    B["benchmark"] = B.item.str.split("::", n=1).str[0]
    B.to_parquet(f"{OUT}/cells_binary.parquet")
    C = pd.concat(cc, ignore_index=True)
    C["benchmark"] = C.item.str.split("::", n=1).str[0]
    C.to_parquet(f"{OUT}/cells_cont.parquet")

    binb = sorted(tb.benchmark[tb.response_type == "binary"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(binb))
    folds = pd.DataFrame({"benchmark": [binb[i] for i in perm], "fold": np.arange(len(binb)) % 5})
    folds.sort_values(["fold", "benchmark"]).to_csv(f"{OUT}/folds.csv", index=False)
    print(f"binary cells {len(B):,} ({B.n.sum():,} responses); continuous cells {len(C):,} ({C.n.sum():,} responses); "
          f"subjects {len(S):,}; names {S.name.nunique():,} harness {S.harness_n.nunique()} "
          f"provider {S.provider_n.nunique()} effort {S.effort_n.nunique()}")


if __name__ == "__main__":
    main()
