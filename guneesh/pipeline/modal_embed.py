"""Run pipeline/embed_manifest.py on a Modal A100, keep the embeddings safe, and copy them to this computer.

Where the embeddings are stored (two copies):
  1. Modal volume "paiec-emb" (persistent cloud disk), folder <model>/ : one .npz per benchmark.
     The volume is committed every 5 minutes and when the run ends, so a crash loses at most the
     benchmark that was being embedded. Rerunning `full` skips benchmarks already on the volume.
  2. This computer: data/emb/<model>/ , filled by the `download` mode. Each file is written to a
     .part file, size-checked, and opened with numpy before it gets its final name.

Commands (run from the project root, AIMS_project/):
  modal run pipeline/modal_embed.py --model 8b --mode parity            # ~5 min incl. model download
  modal run pipeline/modal_embed.py --model 8b --mode pilot             # timing on 300 texts
  modal run --detach pipeline/modal_embed.py --model 8b --mode full     # returns at once; job runs on Modal alone
  modal run pipeline/modal_embed.py --model 8b --mode download          # copy volume -> data/emb/qwen3vl8b
  modal run pipeline/modal_embed.py --model 8b --mode status            # what is on the volume so far
Add --gpu A100-80GB for the 14B model if 40 GB runs out of memory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_LOCAL = ROOT / "data" / "embed_manifest" / "binary_cap5000.parquet"
MANIFEST_REMOTE = "/root/data/embed_manifest/binary_cap5000.parquet"
BACKBONES = {
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3vl8b"),
    "14b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "r1qwen14b"),
}
EMB_MOUNT = "/emb"
HF_MOUNT = "/hf"

# same versions as the local CPU machine where parity and the transformers-version check were verified
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", "transformers==5.12.1", "accelerate==1.15.0",
                 "numpy==2.4.3", "pandas==3.0.5", "pyarrow==25.0.1", "tqdm", "sentencepiece", "pillow")
    .env({"HF_HOME": HF_MOUNT, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(ROOT / "pipeline"), "/root/pipeline", ignore=["__pycache__"])
    .add_local_file(str(MANIFEST_LOCAL), MANIFEST_REMOTE)
)
emb_vol = modal.Volume.from_name("paiec-emb", create_if_missing=True)
hf_vol = modal.Volume.from_name("paiec-hf-cache", create_if_missing=True)
app = modal.App("paiec-embed", image=image)


@app.function(gpu="A100-40GB", timeout=24 * 3600, volumes={EMB_MOUNT: emb_vol, HF_MOUNT: hf_vol})
def embed(model: str, mode: str, max_hours: float) -> None:
    backbone, sub = BACKBONES[model]
    cmd = [sys.executable, "/root/pipeline/embed_manifest.py", "--manifest", MANIFEST_REMOTE,
           "--backbone", backbone, "--max_length", "8192", "--batch_size", "1", "--torch_dtype", "bfloat16",
           "--device_map", "auto", "--embedding_layer", "-1", "--seed", "0"]
    if mode == "pilot":
        cmd += ["--pilot", "300"]
    elif mode == "parity":
        cmd += ["--parity", "20"]
    else:
        cmd += ["--out_dir", f"{EMB_MOUNT}/{sub}", "--max_hours", str(max_hours)]
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    proc = subprocess.Popen(cmd, cwd="/root/pipeline")
    last = time.time()
    while proc.poll() is None:
        time.sleep(10)
        if time.time() - last > 300:  # save finished .npz files to the volume every 5 minutes
            emb_vol.commit()
            hf_vol.commit()
            last = time.time()
    emb_vol.commit()
    hf_vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"embed_manifest.py exited with code {proc.returncode}")


def _list(sub: str):
    try:
        return [e for e in emb_vol.listdir(sub) if e.path.endswith((".npz", ".csv")) and ".tmp." not in e.path]
    except Exception:  # folder not created yet
        return []


def download(sub: str) -> None:
    import numpy as np

    out = ROOT / "data" / "emb" / sub
    out.mkdir(parents=True, exist_ok=True)
    got = skipped = 0
    for e in _list(sub):
        dst = out / Path(e.path).name
        if dst.suffix == ".npz" and dst.exists() and dst.stat().st_size == e.size:
            skipped += 1
            continue
        part = dst.with_name(dst.name + ".part")
        with open(part, "wb") as f:
            emb_vol.read_file_into_fileobj(e.path, f)
        if part.stat().st_size != e.size:
            raise RuntimeError(f"size mismatch for {e.path}: {part.stat().st_size} vs {e.size}")
        if dst.suffix == ".npz":
            with np.load(part, allow_pickle=True) as z:  # proves the file is complete and readable
                assert z["X"].shape[0] == len(z["task_ids"])
        os.replace(part, dst)
        got += 1
    n = len(list(out.glob("*.npz")))
    print(f"downloaded {got}, already present {skipped}; {out} now holds {n} benchmark files")


@app.local_entrypoint()
def main(model: str = "8b", mode: str = "pilot", gpu: str = "", max_hours: float = 22.0):
    if model not in BACKBONES:
        raise SystemExit("--model must be 8b or 14b")
    sub = BACKBONES[model][1]
    if mode == "download":
        download(sub)
        return
    if mode == "status":
        files = _list(sub)
        print(f"volume paiec-emb/{sub}: {sum(f.path.endswith('.npz') for f in files)} of 115 benchmarks, "
              f"{sum(f.size for f in files) / 1e9:.2f} GB")
        return
    if mode not in ("pilot", "parity", "full"):
        raise SystemExit("--mode must be parity, pilot, full, download or status")
    fn = embed.with_options(gpu=gpu) if gpu else embed
    if mode == "full":
        # fire-and-forget: with `modal run --detach` the job keeps running after this command exits
        call = fn.spawn(model, mode, max_hours)
        print(f"started full run, call id {call.object_id}; check with --mode status, then --mode download")
        return
    fn.remote(model, mode, max_hours)
