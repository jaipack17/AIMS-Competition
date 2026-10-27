"""Checks of the MLP path of the submission (submission_src/common/core.py with mlp.npz bundled).

1. core.mlp_b (numpy) equals pipeline/mlp_b_train.Net in eval mode (torch) for both designs, on nets with
   non-trivial LayerNorm parameters, packed with net_weights + pack_ensemble exactly as the training script does.
2. In smoke-test mode item_params returns the bundled centre (mean training b), as it returns the ridge intercept.
3. If the final fit exists (data/mlp_b/<run>/final/mlp.npz): core.mlp_b on real embeddings equals the training
   script's own predictions in pred.parquet.
Run from the project root:  python analysis/test_mlp_submission.py
"""
import glob
import importlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import mlp_b_train as T  # noqa: E402

FINAL = os.path.join(ROOT, "data", "mlp_b", T.RUN, "final")


def load_core(npz: dict, smoke: bool):
    """Import a fresh copy of core.py from a temporary submission folder holding only mlp.npz and a theta.json."""
    d = tempfile.mkdtemp()
    for f in ("core.py", "ap_embed_core.py"):
        shutil.copy(os.path.join(ROOT, "submission_src", "common", f), d)
    np.savez(os.path.join(d, "mlp.npz"), **npz)
    with open(os.path.join(d, "theta.json"), "w") as f:
        json.dump({p: {} for p in ("name", "harness", "provider", "effort")}, f)
    os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1" if smoke else "0"
    sys.path.insert(0, d)
    for m in ("core", "ap_embed_core"):
        sys.modules.pop(m, None)
    core = importlib.import_module("core")
    sys.path.remove(d)
    assert not os.path.exists(os.path.join(d, "ridge.npz")) and core.MLP is not None
    return core


# 1. numpy forward = torch forward
rng = np.random.default_rng(0)
dim = 4096
mean, scale = rng.normal(size=dim), rng.uniform(0.5, 2.0, size=dim)
torch.manual_seed(0)
nets = [T.Net(h, mean, scale) for h in [(64,), (256, 256), (64,)]]
for net in nets:
    with torch.no_grad():
        for m in net.body:
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.normal_(1.0, 0.3)
                m.bias.normal_(0.0, 0.3)
            if isinstance(m, torch.nn.Linear):
                m.bias.normal_(0.0, 0.1)
ym, ysd = 0.37, 1.9
core = load_core(T.pack_ensemble([T.net_weights(n) for n in nets], mean, scale, ym, ysd, center=0.21), smoke=False)
X = (mean + scale * rng.normal(size=(50, dim))).astype(np.float32)
Xt = torch.from_numpy(X)
ref = np.mean([T.predict(n, Xt, np.arange(len(X))) for n in nets], axis=0) * ysd + ym
ours = np.array([core.mlp_b(x.astype(np.float64)) for x in X])
err = np.abs(ours - ref).max()
assert err < 1e-4 and np.std(ref) > 0.1, (err, np.std(ref))
assert all(not n.training for n in nets)  # predict() put the nets in eval mode: no dropout, no noise
print(f"1. numpy MLP = torch Net (eval) on 50 inputs, 3 nets of 2 designs: max diff {err:.1e} (b sd {np.std(ref):.2f})")

# 2. smoke-test mode: the centre, as the ridge intercept
core_s = load_core(T.pack_ensemble([T.net_weights(n) for n in nets], mean, scale, ym, ysd, center=0.21), smoke=True)
assert core_s.item_params({"item_content": "x"}).tolist() == [0.21] and core_s.item_b_hat({"item_content": "x"}) == 0.21
print("2. smoke-test b_hat = bundled centre")

# 3. the real final ensemble reproduces the training script's predictions
if os.path.exists(os.path.join(FINAL, "mlp.npz")):
    with np.load(os.path.join(FINAL, "mlp.npz")) as z:
        packed = {k: z[k] for k in z.files}
    core_f = load_core(packed, smoke=False)
    pred = pd.read_parquet(os.path.join(FINAL, "pred.parquet")).set_index("item").b_pred
    checked, worst = 0, 0.0
    for p in sorted(glob.glob(os.path.join(T.EMB_DIR, "*.npz")))[::15]:      # every 15th benchmark, 40 items each
        b = os.path.basename(p)[:-4]
        with np.load(p, allow_pickle=True) as z:
            Xb, ids = z["X"][:40], z["task_ids"][:40].astype(str)
        for x, i in zip(Xb, ids):
            worst = max(worst, abs(core_f.mlp_b(x.astype(np.float64)) - pred[f"{b}::{i}"]))
            checked += 1
    assert worst < 1e-3, worst
    print(f"3. final mlp.npz on {checked} real embeddings = pred.parquet: max diff {worst:.1e}")
else:
    print("3. skipped: no final fit yet")
print("ALL CHECKS PASSED")
