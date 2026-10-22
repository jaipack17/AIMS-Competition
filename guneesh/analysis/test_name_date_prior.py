"""Checks of the unseen-name date prior in a built submission (run after build_submission.py --name_prior ...,
from the project root):  python analysis/test_name_date_prior.py <submission folder name>

1. core._release_years == pipeline/name_date_prior.release_years on every training release date and on edge cases.
2. A known name: theta is exactly the old sum of its four parts (the prior is not used).
3. An unseen name: theta = old sum + a + b * clipped years; blank or unreadable date -> old sum.
The model is imported in local smoke-test mode (no GPU).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pandas as pd

os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
sys.path.insert(0, "pipeline")
from name_date_prior import release_years  # noqa: E402


def main() -> None:
    d = os.path.abspath(os.path.join("submissions", sys.argv[1]))
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("core", os.path.join(d, "core.py"))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    with open(os.path.join(d, "name_date_prior.json")) as f:
        pr = json.load(f)
    assert core.NAME_PRIOR == pr

    # 1. date parsing
    dates = pd.read_parquet("data/irt/subjects.parquet").release_date.unique().tolist()
    dates += ["2024-01-01", "2023-05", "2025", " 2025-04-05 ", "2024-08-09T12:00:00Z", "20240809", "2024-02-30",
              "2024-13", "unknown", "", "May 2024", "2024/05/01"]
    for s in dates:
        assert core._release_years(s) == release_years(s), s
    expect = {"2024-01-01": 0.0, "2023-05": (pd.Timestamp("2023-05-15") - pd.Timestamp("2024-01-01")).days / 365.25,
              "2025": (pd.Timestamp("2025-07-01") - pd.Timestamp("2024-01-01")).days / 365.25,
              "20240809": None, "2024-02-30": None, "2024-13": None, "unknown": None, "": None}
    for s, v in expect.items():
        assert core._release_years(s) == v, (s, core._release_years(s), v)
    print(f"date parsing identical on {len(dates)} strings; edge cases as expected")

    # 2 + 3. theta
    base = {"provider": "openai", "access_date": "", "harness": "", "reasoning_effort": "high",
            "harness_version": "", "subject_features_extra": ""}
    known = next(iter(core.THETA["name"]))
    old = lambda s: sum(core.THETA[p].get(k, 0.0) for p, k in
                        (("name", s["normalized_name"]), ("provider", "openai"), ("effort", "high")))
    s = {**base, "normalized_name": known, "release_date": "2026-03-10"}
    assert core.subject_theta(s) == old(s), "known name changed"
    for date, years in (("2026-03-10", 2.0), ("2023-03-10", None), ("2019-01-01", -1.0), ("", "none"), ("n/a", "none")):
        s = {**base, "normalized_name": "Totally New Model 9000", "release_date": date}
        y = release_years(date) if years is None else years
        want = old(s) + (0.0 if y == "none" else pr["a"] + pr["b"] * y)
        got = core.subject_theta(s)
        assert abs(got - want) < 1e-12, (date, got, want)
        print(f"unseen name, release_date {date!r:14}: theta {got:+.4f} (known parts {old(s):+.4f})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
