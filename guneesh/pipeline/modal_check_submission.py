"""End-to-end check of a built submission folder on a Modal GPU, the way the platform runs it.

Inside the container (same image as modal_embed.py, Qwen3-VL-8B from the paiec-hf-cache volume, hub offline):
  1. import model.py as a prediction worker (PAIEC_PREDICTION_WORKER_ID=0), so the model loads at import;
  2. call predict() on real training items (data/submission_check_items.parquet, built with their raw
     item content and one real subject per benchmark);
  3. compare each item's b_hat with the ridge prediction made from the stored training embedding
     (data/ridge/<run>/final/pred.parquet, or data/submission_check_items_<variant>.parquet for other ridges) and each probability with sigmoid(theta - b_pred);
  4. call predict() again for every item: the second pass must come from the on-disk b_hat cache.
HF_HOME points at the folder that holds models--Qwen--..., as the starting-kit template expects (cache_dir=HF_HOME).

label_update additionally: model loads at import in the parent (no worker id), labels move the prediction the
right way, and labeling.py returns a decision.

Run from the project root:  modal run pipeline/modal_check_submission.py --variant ap_pure|label_update
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
CHECK_ITEMS = ROOT / "data" / "submission_check_items.parquet"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", "transformers==5.12.1", "accelerate==1.15.0",
                 "numpy==2.4.3", "pandas==3.0.5", "pyarrow==25.0.1", "sentencepiece", "pillow")
    .env({"HF_HOME": "/hf/hub", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(ROOT / "submissions"), "/sub", ignore=["*.zip", "**/bhat_cache/**", "**/__pycache__/**"])
    .add_local_file(str(CHECK_ITEMS), "/check/items.parquet")
)
for _extra in sorted((ROOT / "data").glob("submission_check_items_*.parquet")):  # expected b_hat per variant
    image = image.add_local_file(str(_extra), f"/check/{_extra.name}")
hf_vol = modal.Volume.from_name("paiec-hf-cache")
app = modal.App("paiec-check-submission", image=image)


@app.function(gpu="A100-40GB", timeout=1800, volumes={"/hf": hf_vol})
def check(variant: str) -> dict:
    import importlib.util
    import math
    import os
    import shutil
    import sys
    import time

    import pandas as pd

    work = Path("/tmp/sub")  # writable copy, like the platform's submission directory
    shutil.copytree(f"/sub/{variant}", work)
    has_hook = (Path(f"/sub/{variant}") / "labeling.py").exists()
    if not has_hook:  # without a hook the model loads in prediction workers only; with one, in every process
        os.environ["PAIEC_PREDICTION_WORKER_ID"] = "0"
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("participant_model", work / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    load_s = time.time() - t0
    core = sys.modules["core"]
    assert core._STATE is not None, "model did not load at import"

    per_variant = Path(f"/check/submission_check_items_{variant}.parquet")
    T = pd.read_parquet(per_variant if per_variant.exists() else "/check/items.parquet")
    fields = ["normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort",
              "harness_version", "subject_features_extra"]
    rows, t1 = [], time.time()
    for r in T.to_dict("records"):
        subject = {k: r[k] for k in fields}
        item = {"item_content": r["content"], "item_features": "", "interactors": "", "benchmark_id": "benchmark_x"}
        p = model.predict([subject, item], labeled=[])
        b = core.item_b_hat(item)
        th = core.subject_theta(subject)
        z = th - r["b_pred"]
        if getattr(model, "AVERAGE", False):  # no labels: delta, zeta and the item shift are all at their priors
            var = model.SIGMA_DELTA ** 2 + model.SIGMA_ZETA ** 2 + core.subject_theta_var(subject) + model.SIGMA_ITEM ** 2
            z /= math.sqrt(1 + math.pi * var / 8)
        p_expected = 1 / (1 + math.exp(-z))
        rows.append({"item": r["item"], "b_hat": b, "b_pred_train": r["b_pred"], "p": p, "p_expected": p_expected})
    first_s = time.time() - t1
    n_cache = len(list((work / "bhat_cache").glob("*.txt")))
    t2 = time.time()
    again = [model.predict([{k: r[k] for k in fields},
                            {"item_content": r["content"], "item_features": "", "interactors": "",
                             "benchmark_id": "benchmark_x"}], labeled=[]) for r in T.to_dict("records")]
    cached_s = time.time() - t2
    R = pd.DataFrame(rows)
    extra = {}
    if has_hook:
        # labels on one benchmark: 6 other items of it answered correctly by the first subject of that benchmark
        lab_mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("lab", work / "labeling.py"))
        lab_mod.__spec__.loader.exec_module(lab_mod)
        recs = T.to_dict("records")
        bench = recs[0]["benchmark"]
        subj = {k: recs[0][k] for k in fields}
        same = [r for r in recs if r["benchmark"] == bench]
        target = {"item_content": same[0]["content"], "item_features": "", "interactors": "", "benchmark_id": bench}
        labeled = [[[subj, {"item_content": r["content"], "item_features": "", "interactors": "",
                            "benchmark_id": bench}], 1] for r in same[1:]]
        p0 = model.predict([subj, target], labeled=[])
        p_up = model.predict([subj, target], labeled=labeled)
        p_down = model.predict([subj, target], labeled=[[x, 0] for x, _ in labeled])
        hook = lab_mod.acquisition_function([subj, target], prediction=p_up, labeled=labeled,
                                            context={"labels_remaining": 29, "labels_acquired": 2, "max_labels": 31,
                                                     "items_remaining": 40, "subject_id": "s", "benchmark_id": bench})
        extra = {"p_no_labels": p0, "p_after_correct_labels": p_up, "p_after_wrong_labels": p_down,
                 "hook_decision": hook, "labels_move_prediction_right_way": bool(p_down < p0 < p_up)}
    import torch
    import transformers
    return {
        "gpu": torch.cuda.get_device_name(0), "transformers": transformers.__version__,
        "load_seconds": round(load_s, 1), "first_pass_seconds_per_item": round(first_s / len(T), 3),
        "cached_pass_seconds_per_item": round(cached_s / len(T), 5), "cache_files": n_cache, "n_items": len(T),
        "max_abs_diff_b": float((R.b_hat - R.b_pred_train).abs().max()),
        "corr_b": float(R.b_hat.corr(R.b_pred_train)),
        "max_abs_diff_p": float((R.p - R.p_expected).abs().max()),
        "second_pass_identical": bool(all(a == b for a, b in zip(again, R.p))),
        **extra,
        "table": R.round(5).to_dict("records"),
    }


@app.local_entrypoint()
def main(variant: str = "ap_pure"):
    out = check.remote(variant)
    table = out.pop("table")
    for k, v in out.items():
        print(f"{k}: {v}")
    print("\nitem | b_hat (submission) | b_pred (training embedding) | p | p expected")
    for r in table[:12]:
        print(f"{r['item'][:45]:45s} {r['b_hat']: .5f} {r['b_pred_train']: .5f} {r['p']:.5f} {r['p_expected']:.5f}")
    ok = (out["max_abs_diff_b"] < 0.01 and out["max_abs_diff_p"] < 0.005 and out["second_pass_identical"]
          and out.get("labels_move_prediction_right_way", True))
    print("\nRESULT:", "SUBMISSION PIPELINE OK" if ok else "MISMATCH")
