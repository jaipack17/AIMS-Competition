"""Prior for the name ability of a subject whose name was never seen in training:  theta_name = a + b * years,
years = release_date in years since 2024-01-01, clipped to [YEARS_MIN, YEARS_MAX].

a, b = ordinary least squares over the named training subjects of (years of their release date, their theta_name
from the IRT fit). A name with several release dates uses their median. Blank / unreadable dates are left out.
Reported as well: leave-one-name-out RMSE of this line against the current guess of 0.

release_years() is the same function as core._release_years (submission_src/common/core.py), checked by
analysis/test_name_date_prior.py.

Output: data/irt/<run>/<fold>/name_date_prior.json.  Run from the project root:  python pipeline/name_date_prior.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re

import numpy as np
import pandas as pd

YEARS_MIN, YEARS_MAX = -1.0, 2.0  # 2023-01-01 .. 2026-01-01: the line is held flat outside this range
_DATE = re.compile(r"\s*(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?!\d)")


def release_years(release_date: str) -> float | None:
    """'YYYY-MM-DD' -> years since 2024-01-01; 'YYYY-MM' -> the 15th of that month; 'YYYY' -> 1 July; else None."""
    m = _DATE.match(release_date)
    if m is None:
        return None
    month = int(m[2]) if m[2] else 7
    day = int(m[3]) if m[3] else (15 if m[2] else 1)
    try:
        d = datetime.date(int(m[1]), month, day)
    except ValueError:
        return None
    return (d - datetime.date(2024, 1, 1)).days / 365.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="versionB_4parts_logitnormal")
    ap.add_argument("--fold", default="final")
    args = ap.parse_args()

    theta = pd.read_parquet(os.path.join("data", "irt", args.run, args.fold, "theta_name.parquet"))
    S = pd.read_parquet(os.path.join("data", "irt", "subjects.parquet"))
    S = S[~S.name.str.startswith("private::")]
    S = S.assign(years=S.release_date.map(release_years)).dropna(subset=["years"])
    g = S.groupby("name").years.median().to_frame().join(theta.set_index("key").theta, how="inner")
    x = g.years.clip(YEARS_MIN, YEARS_MAX).to_numpy()
    y = g.theta.to_numpy()
    A = np.c_[np.ones(len(x)), x]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    loo = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        w = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        loo.append(y[i] - A[i] @ w)
    rmse_zero, rmse_line = float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(np.square(loo))))

    out = {"a": float(a), "b": float(b), "years_min": YEARS_MIN, "years_max": YEARS_MAX, "n_names": int(len(x)),
           "loo_rmse_zero": rmse_zero, "loo_rmse_line": rmse_line, "run": args.run, "fold": args.fold}
    path = os.path.join("data", "irt", args.run, args.fold, "name_date_prior.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{path}: theta_name = {a:.4f} + {b:.4f} * years (clipped to [{YEARS_MIN}, {YEARS_MAX}]), "
          f"{len(x)} names; leave-one-out RMSE {rmse_line:.4f} vs {rmse_zero:.4f} for 0")


if __name__ == "__main__":
    main()
