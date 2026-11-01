"""Read the files written by probe/extract_activations.py.

Hidden states are stored as uint16 arrays holding the raw bfloat16 bits (lossless); bf16_to_f32 decodes them.
Array names in each <benchmark>.npz:
  item, b, parity_maxabs, skipped, layers (the layer numbers stored in the hidden-state arrays)
  <prompt>__last, <prompt>__item_last, <prompt>__item_mean   (n, n_saved_layers, hidden) uint16 (item_* for the first prompt)
  <prompt>__digit_probs (n, n_layers, 10), __digit_mass (n, n_layers), __entropy (n, n_layers),
  <prompt>__top5_ids (n, 5), __top5_probs (n, 5), __n_tokens (n,), __truncated (n,)

Example:
  from acts_io import load_layer, load_lens
  X, meta = load_layer("probe/acts/pilot", "ap__last", layer=20)       # real layer number (must be a saved one)
  L, meta = load_lens("probe/acts/pilot", "rate__digit_probs")          # (n, n_layers, 10)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def benchmark_files(run_dir) -> list[Path]:
    return sorted(p for p in Path(run_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))


def config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run_config.json").read_text())


def _meta(z, path: Path) -> pd.DataFrame:
    return pd.DataFrame({"item": z["item"].astype(str), "b": z["b"], "benchmark": path.stem})


def saved_layers(run_dir) -> list[int]:
    """Layer numbers stored in the hidden-state arrays, in order."""
    cfg = config(run_dir)
    return cfg.get("layers_saved") or list(range(cfg["n_layer_outputs"]))


def _slot(z, layer: int) -> int:
    layers = list(z["layers"]) if "layers" in z.files else None
    if layers is None:
        return layer
    if layer not in layers:
        raise KeyError(f"layer {layer} was not saved; saved layers: {layers}")
    return layers.index(layer)


def load_layer(run_dir, key: str, layer: int, benchmarks=None):
    """Hidden states of one key (e.g. "ap__last") at one layer (real layer number), for all (or given) benchmarks."""
    Xs, metas = [], []
    for p in benchmark_files(run_dir):
        if benchmarks is not None and p.stem not in benchmarks:
            continue
        with np.load(p) as z:
            Xs.append(bf16_to_f32(z[key][:, _slot(z, layer), :]))
            metas.append(_meta(z, p))
    return np.concatenate(Xs), pd.concat(metas, ignore_index=True)


def load_all_layers(run_dir, key: str, benchmarks=None):
    """All saved layers of one key: (n, n_saved_layers, hidden) float32, in the order of saved_layers().
    Pilot-sized runs only (memory)."""
    Xs, metas = [], []
    for p in benchmark_files(run_dir):
        if benchmarks is not None and p.stem not in benchmarks:
            continue
        with np.load(p) as z:
            Xs.append(bf16_to_f32(z[key]))
            metas.append(_meta(z, p))
    return np.concatenate(Xs), pd.concat(metas, ignore_index=True)


def load_lens(run_dir, key: str, benchmarks=None):
    """A non-activation array (digit_probs, digit_mass, entropy, top5_*, n_tokens, ...) for all benchmarks."""
    Xs, metas = [], []
    for p in benchmark_files(run_dir):
        if benchmarks is not None and p.stem not in benchmarks:
            continue
        with np.load(p) as z:
            Xs.append(np.asarray(z[key]))
            metas.append(_meta(z, p))
    return np.concatenate(Xs), pd.concat(metas, ignore_index=True)


def consolidate(run_dir, key: str, out_path=None) -> Path:
    """Write one (n, n_saved_layers, hidden) uint16 .npy for `key` that can be memory-mapped (for the full run, where
    loading every benchmark once per layer is slow). Also writes <key>_index.parquet with item, b, benchmark."""
    files = benchmark_files(run_dir)
    shapes = []
    for p in files:
        with np.load(p) as z:
            shapes.append(z[key].shape)
    n = sum(s[0] for s in shapes)
    out_path = Path(out_path or Path(run_dir) / f"{key}.npy")
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(n, *shapes[0][1:]))
    metas, i = [], 0
    for p, s in zip(files, shapes):
        with np.load(p) as z:
            mm[i:i + s[0]] = z[key]
            metas.append(_meta(z, p))
        i += s[0]
    mm.flush()
    pd.concat(metas, ignore_index=True).to_parquet(out_path.with_name(f"{key}_index.parquet"), index=False)
    return out_path


def load_consolidated_layer(run_dir, key: str, layer: int):
    mm = np.load(Path(run_dir) / f"{key}.npy", mmap_mode="r")
    slot = saved_layers(run_dir).index(layer)
    return bf16_to_f32(mm[:, slot, :]), pd.read_parquet(Path(run_dir) / f"{key}_index.parquet")
