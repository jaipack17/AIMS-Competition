"""Run pipeline/mirt_fit.py on Modal A100 GPUs, keep every result safe, and copy the results to this computer.

Where results are stored (two copies):
  1. Modal volume "paiec-mirt": /out/mirt_<form>_K<K>/<fit>/ . The fit writes a checkpoint every 200 epochs and the
     volume is committed every 2 minutes and when the fit ends, so a crash loses at most a few minutes; a rerun of
     the same fit resumes from its checkpoint, and finished fits (DONE) are skipped.
  2. This computer: data/irt/mirt_<form>_K<K>/<fit>/ , filled by the `download` mode (only finished fits; every
     file is written to a .part file, size-checked, then renamed).

Inputs baked into the image: data/irt/cells_binary.parquet, cells_cont.parquet, subjects.parquet, folds.csv and the
1D warm-start fits data/irt/versionB_4parts_logitnormal/<fit>/ (items, theta_*, zeta).

Commands (from the project root):
  modal run pipeline/modal_mirt.py --mode pilot --form nested --k 2            # 100 epochs of fold 0: time per epoch
  modal run --detach pipeline/modal_mirt.py --mode fit --form nested --k 2 --folds 0,1,2,3,4
  modal run --detach pipeline/modal_mirt.py --mode grid --forms nested,m2pl --ks 1,2,3,5,8,10,15 --folds 0,1,2,3,4
  python pipeline/modal_mirt.py status          # plain Python: reads the volume, starts no app, needs no credit
  python pipeline/modal_mirt.py download
One container per (form, K, fold), run in parallel.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
IRT = ROOT / "data" / "irt"
OUT = "/out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "pyro-ppl==1.9.2", "numpy==2.4.3", "pandas==3.0.5", "pyarrow==25.0.1",
                 "scipy==1.17.1")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(IRT / "cells_binary.parquet"), "/root/data/irt/cells_binary.parquet", copy=True)
    .add_local_file(str(IRT / "cells_cont.parquet"), "/root/data/irt/cells_cont.parquet", copy=True)
    .add_local_file(str(IRT / "subjects.parquet"), "/root/data/irt/subjects.parquet", copy=True)
    .add_local_file(str(IRT / "folds.csv"), "/root/data/irt/folds.csv", copy=True)
    .add_local_dir(str(IRT / "versionB_4parts_logitnormal"), "/root/data/irt/versionB_4parts_logitnormal",
                   ignore=["**/losses.csv", "**/ckpt_*"], copy=True)
    .add_local_dir(str(ROOT / "pipeline"), "/root/pipeline", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("paiec-mirt", create_if_missing=True)
app = modal.App("paiec-mirt", image=image)


# A100-40GB first; if Modal has no capacity for it, the next GPU in the list is used (all have >= 40 GB)
@app.function(gpu=["A100-40GB", "A100-80GB", "L40S"], timeout=12 * 3600, volumes={OUT: vol})
def fit(form: str, k: int, fold: str, epochs: int = 2000, out_sub: str = "") -> str:
    out_root = f"{OUT}/{out_sub}" if out_sub else OUT
    cmd = [sys.executable, "/root/pipeline/mirt_fit.py", "--irt_dir", "/root/data/irt", "--out_root", out_root,
           "--form", form, "--K", str(k), "--fold", fold, "--epochs", str(epochs), "--device", "cuda"]
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], check=False)
    vol.reload()                                   # see checkpoints left by an earlier, interrupted run
    proc = subprocess.Popen(cmd, cwd="/root")
    last = time.time()
    while proc.poll() is None:
        time.sleep(10)
        if time.time() - last > 120:              # checkpoints and outputs reach the volume every 2 minutes
            vol.commit()
            last = time.time()
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"mirt_fit.py {form} K={k} fold {fold} exited with code {proc.returncode}")
    return f"{form} K={k} fold {fold} done"


def _walk(prefix: str):
    try:
        entries = vol.listdir(prefix, recursive=True)
    except Exception:
        return []
    return [e for e in entries if e.type == modal.volume.FileEntryType.FILE]


def download() -> None:
    files = _walk("/")
    done_dirs = {os.path.dirname(e.path) for e in files if e.path.endswith("/DONE") and not e.path.startswith("pilot")}
    got = skipped = 0
    for e in files:
        d = os.path.dirname(e.path)
        if d not in done_dirs or os.path.basename(e.path).startswith("ckpt_"):
            continue
        dst = IRT / e.path
        if dst.exists() and dst.stat().st_size == e.size:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        with open(part, "wb") as f:
            vol.read_file_into_fileobj(e.path, f)
        if part.stat().st_size != e.size:
            raise RuntimeError(f"size mismatch for {e.path}")
        os.replace(part, dst)
        got += 1
    print(f"downloaded {got} files, {skipped} already present; finished fits on the volume: {len(done_dirs)}")
    for d in sorted(done_dirs):
        print("  ", d)


def status() -> None:
    files = _walk("/")
    fits = sorted({os.path.dirname(e.path) for e in files if e.path.count("/") >= 1})
    counts = {}
    for d in fits:
        names = {os.path.basename(e.path) for e in files if os.path.dirname(e.path) == d}
        state = "DONE" if "DONE" in names else ("checkpointed" if "ckpt_params.pt" in names else "started")
        counts[state] = counts.get(state, 0) + (not d.startswith("pilot"))
        print(f"{d:45s} {state}")
    print("fits (pilot excluded):", counts)


@app.local_entrypoint()
def main(mode: str = "status", form: str = "nested", k: int = 2, folds: str = "0", forms: str = "nested,m2pl",
         ks: str = "1,2,3,5,8,10,15", epochs: int = 2000):
    if mode == "status":
        status()
    elif mode == "download":
        download()
    elif mode == "pilot":
        print(fit.remote(form, k, "0", 100, "pilot"))
        status()
    elif mode in ("fit", "grid"):
        jobs = ([(form, k, f) for f in folds.split(",")] if mode == "fit" else
                [(fm, int(kk), f) for fm in forms.split(",") for kk in ks.split(",") for f in folds.split(",")])
        calls = [fit.spawn(fm, kk, f, epochs) for fm, kk, f in jobs]
        print(f"started {len(calls)} fits: {jobs}")
        print("check with --mode status; copy finished fits with --mode download")
    else:
        raise SystemExit("--mode must be pilot, fit, grid, status or download")


if __name__ == "__main__":       # status / download without starting a Modal app (no GPU, no credit needed)
    if len(sys.argv) != 2 or sys.argv[1] not in ("status", "download"):
        raise SystemExit("usage: python pipeline/modal_mirt.py status|download")
    status() if sys.argv[1] == "status" else download()
