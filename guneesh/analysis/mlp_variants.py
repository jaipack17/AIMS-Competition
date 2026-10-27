"""Build the label-update submissions with the MLP difficulty model (data/mlp_b) in place of the ridge, and score each
one with analysis/simulate_alc.py against its ridge twin built from the same (current) code.

Variants (the settings of the uploaded ridge zips; the stretch1 ones, which need a within-benchmark ridge, are left
out):  label_update, label_update_date, label_update_stretch2, label_update_stretch2_date,
       label_update_3improvements, label_update_3improvements_tuned, sim_stretch2.
--model ridge builds the ridge twins as regress_<variant> (a regression check: their ALC must equal the earlier
local results), --model mlp builds <variant>_mlp.  Every zip then goes through the kit's check_submission_zip.py and
run_smoke_test.py and the simulation (150 draws, the configuration that matches its switches).
Run from the project root:  python analysis/mlp_variants.py --model ridge|mlp [--only a,b] [--jobs 2]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

RUN = "versionB_4parts_logitnormal"
NP = ["--name_prior", f"data/irt/{RUN}/final/name_date_prior.json"]
ALL3 = ["--item_shifts", "--average", "--sigma_item_scale", "0.6"]
#  variant: (build options without the difficulty model, uses stretch, simulate_alc configuration)
VARIANTS = {
    "label_update":                     ([], False, "label_update"),
    "label_update_date":                (NP, False, "+date"),
    "label_update_stretch2":            ([], True, "label_update"),
    "label_update_stretch2_date":       (NP, True, "+date"),
    "label_update_3improvements":       (NP + ALL3, False, "all3"),
    "label_update_3improvements_tuned": (NP + ALL3 + ["--sigma_delta_scale", "2.0"], False, "all3"),
    "sim_stretch2":                     (NP + ALL3, True, "all3"),
}
KIT = "Starting Kit/tools"


def run(cmd: list[str], log: str) -> None:
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(cmd)}  -> see {log}")


def one(variant: str, model: str) -> str:
    opts, stretch, config = VARIANTS[variant]
    d = "data/ridge" if model == "ridge" else "data/mlp_b"
    name = f"regress_{variant}" if model == "ridge" else f"{variant}_mlp"
    log = os.path.join("logs", f"variant_{name}.log")
    open(log, "w").close()
    build = [sys.executable, "pipeline/build_submission.py", "--variant", "label_update", "--name", name,
             "--ridge_dir", d] + opts + (["--stretch", f"{d}/{RUN}/stretch.json"] if stretch else [])
    run(build, log)
    run([sys.executable, f"{KIT}/check_submission_zip.py", f"submissions/{name}.zip"], log)
    run([sys.executable, f"{KIT}/run_smoke_test.py", f"submissions/{name}"], log)
    run([sys.executable, "-u", "analysis/simulate_alc.py", "--sub", name, "--draws", "30", "--configs", config], log)
    return f"{name}: done ({config})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ridge", "mlp"], required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(VARIANTS)
    with ThreadPoolExecutor(args.jobs) as ex:
        for msg in ex.map(lambda v: one(v, args.model), names):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
