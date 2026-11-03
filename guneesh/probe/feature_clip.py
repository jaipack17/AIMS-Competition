"""Per-dimension clipping bounds for a probe feature (CPU): <emb_dir>/clip.json with lo, hi.

Why: at layer 20 the last token of the rate prompt sometimes carries a "massive activation" in one dimension
(Qwen3-VL-8B: dim 2276, 400-6000 instead of ~0-60; normal vectors have norm ~90). Whether it fires for an item flips
with bf16 rounding (batching, GPU, kernel): in the full-run vs all-items comparison it appears in either run alone or
in both with different sizes, on the same tokens. After standardisation such a spike is dozens of SDs, so a test item
that spikes by chance would get an extreme b_hat (then stretched x4). Clipping every dimension to
median +- K * 1.4826 * MAD (K robust SDs, default 8) bounds that, and changes almost nothing else.

The bounds use no labels. They are applied identically in training (pipeline/ridge_train.py and stretch_factor.py
--clip <emb_dir>/clip.json) and at test time (build_submission.py bundles them as feature_clip.json next to
embed_spec.json; core.py clips before the ridge).

  python probe/feature_clip.py data/emb/qwen3vl8b_rate_L20 [--k 8] [--max_rows 100000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("emb_dir")
    pa.add_argument("--k", type=float, default=8.0, help="bounds = median +- k robust SDs (1.4826 MAD)")
    pa.add_argument("--max_rows", type=int, default=100_000, help="rows used for the median / MAD (random subset)")
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()

    files = sorted(glob.glob(os.path.join(args.emb_dir, "*.npz")))
    X = np.concatenate([np.load(f, allow_pickle=True)["X"] for f in files])
    print(f"{X.shape[0]:,} vectors x {X.shape[1]} from {len(files)} files", flush=True)
    rng = np.random.default_rng(args.seed)
    S = X[rng.choice(len(X), min(args.max_rows, len(X)), replace=False)]
    med = np.median(S, axis=0)
    mad = np.median(np.abs(S - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-3 * np.maximum(np.abs(med), 1.0))   # constant dimensions: avoid a zero-width range
    lo, hi = (med - args.k * mad).astype(np.float32), (med + args.k * mad).astype(np.float32)

    # statistics in chunks (X is ~4 GB)
    n_out_dim, n_out_vals, any_out, norm, normc = np.zeros(X.shape[1]), 0, [], [], []
    for i in range(0, len(X), 20000):
        x = X[i:i + 20000]
        o = (x < lo) | (x > hi)
        n_out_dim += o.sum(0)
        n_out_vals += int(o.sum())
        any_out.append(o.any(1))
        norm.append(np.linalg.norm(x, axis=1))
        normc.append(np.linalg.norm(np.clip(x, lo, hi), axis=1))
    any_out, norm, normc = np.concatenate(any_out), np.concatenate(norm), np.concatenate(normc)
    big = norm > 3 * np.median(norm)
    print(f"clipped values: {n_out_vals / X.size:.2e} of all; rows with any clipped value: {any_out.mean():.2%}; "
          f"rows with norm > 3x median (spikes): {big.sum():,} ({big.mean():.2%})")
    worst = np.argsort(-n_out_dim)[:5]
    print("most clipped dimensions: " + ", ".join(
        f"{d} ({n_out_dim[d] / len(X):.2%}, range {lo[d]:.1f}..{hi[d]:.1f}, max |x| {np.abs(X[:, d]).max():.0f})"
        for d in worst))
    if big.any():
        print(f"norm of spike rows: median {np.median(norm[big]):.0f} -> {np.median(normc[big]):.0f} after clipping "
              f"(all rows: {np.median(norm):.0f} -> {np.median(normc):.0f})")
    path = os.path.join(args.emb_dir, "clip.json")   # not .npz: every .npz in emb_dir is read as a benchmark
    with open(path, "w") as f:
        json.dump({"lo": lo.tolist(), "hi": hi.tolist(), "k": args.k, "n_rows": int(len(S))}, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
