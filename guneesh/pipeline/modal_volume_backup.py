"""Mirror a Modal volume to this computer: data/modal_backup/<profile>/<volume>/<path>.

Every file is written to a .part file, size-checked, then renamed; files already present with the same size are
skipped, so the command can be rerun any time to resume. It reads the volume only (no app, no GPU, no credit).

Run:  python pipeline/modal_volume_backup.py <volume> [<volume> ...]      (uses the active Modal profile, or
      MODAL_PROFILE=<profile> python pipeline/modal_volume_backup.py ...)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent


def profile() -> str:
    if os.environ.get("MODAL_PROFILE"):
        return os.environ["MODAL_PROFILE"]
    try:
        import tomllib
        cfg = tomllib.loads((Path.home() / ".modal.toml").read_text())
        return next(k for k, v in cfg.items() if isinstance(v, dict) and v.get("active"))
    except Exception:
        return "default"


def mirror(name: str, prof: str) -> None:
    vol = modal.Volume.from_name(name)
    files = [e for e in vol.listdir("/", recursive=True) if e.type == modal.volume.FileEntryType.FILE]
    files.sort(key=lambda e: e.size)                    # small result files first, big checkpoints last
    dest = ROOT / "data" / "modal_backup" / prof / name
    total, got, skipped, t0 = sum(e.size for e in files), 0, 0, time.time()
    print(f"[{prof}/{name}] {len(files)} files, {total / 1e9:.1f} GB -> {dest}", flush=True)
    for i, e in enumerate(files, 1):
        dst = dest / e.path
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
        if i % 20 == 0 or e.size > 1e8:
            print(f"  {i}/{len(files)} {e.path} ({e.size / 1e6:.0f} MB)  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[{prof}/{name}] done: downloaded {got}, already present {skipped}  ({(time.time() - t0) / 60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python pipeline/modal_volume_backup.py <volume> [<volume> ...]")
    for v in sys.argv[1:]:
        mirror(v, profile())
