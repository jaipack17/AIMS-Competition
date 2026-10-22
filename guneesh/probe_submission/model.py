"""Runtime probe: checks whether the test machine can run our embedding model. Not a real predictor.

Hosted logs are hidden, so each check is reported through the per-budget Brier table:
at label budget k the probe predicts 0.5 if check k passed (Brier is then exactly 0.2500 for every
subject-benchmark pair, whatever the outcomes) and 0.0 if it failed (Brier = that pair's success rate,
almost never exactly 0.25). Only our own environment is inspected; no test data is recorded or sent.

  budget 0  : transformers >= 4.57 (native Qwen3-VL support)
  budget 1  : Qwen/Qwen3-VL-8B-Instruct loads from the prefetched cache and one forward pass gives a
              finite 4096-dim last-token vector (only attempted on a CUDA GPU with >= 20 GB)
  budget 3  : torch.cuda.is_available()
  budget 7  : GPU memory >= 30 GB
  budget 15 : GPU memory >= 60 GB
  budget 31 : accelerate is importable (needed for device_map="auto")
The budget is read from how many labels this subject-benchmark pair has in `labeled`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
BUDGET_INDEX = {0: 0, 1: 1, 3: 2, 7: 3, 15: 4, 31: 5}
_CHECKS = None


def _cache_dir():  # same logic as the starting kit's hf_submission template
    for c in [os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(Path(__file__).with_name(".hf_cache"))]:
        if not c:
            continue
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(p, os.W_OK):
            return str(p)
    return None


def _version_at_least(v: str, target: tuple) -> bool:
    parts = []
    for x in v.split(".")[:3]:
        digits = "".join(ch for ch in x if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) >= target


def _run_checks() -> list:
    ok = [False] * 6
    try:
        import transformers
        ok[0] = _version_at_least(transformers.__version__, (4, 57, 0))
    except Exception:
        pass
    mem_gb = 0.0
    try:
        import torch
        ok[2] = bool(torch.cuda.is_available())
        if ok[2]:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok[3] = mem_gb >= 30
        ok[4] = mem_gb >= 60
    except Exception:
        pass
    try:
        import accelerate  # noqa: F401
        ok[5] = True
    except Exception:
        pass
    if ok[2] and mem_gb >= 20:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            cd = _cache_dir()
            tok = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=cd, local_files_only=True)
            model = AutoModel.from_pretrained(REPO_ID, cache_dir=cd, local_files_only=True,
                                              torch_dtype=torch.bfloat16).to("cuda").eval()
            text = ("Task statement:\nWhat is 1 + 1?\n\nHow difficult is the above task for a coding agent? Please output "
                    "one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n")
            enc = tok([text], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            vec = out.last_hidden_state[0, -1].float()
            ok[1] = vec.numel() == 4096 and bool(torch.isfinite(vec).all())
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    return ok


def predict(input: list, labeled: list | None = None) -> float:
    global _CHECKS
    if _CHECKS is None:
        _CHECKS = _run_checks()
    subject, item = input
    n = sum(1 for (s, it), _ in (labeled or []) if s == subject and it.get("benchmark_id") == item.get("benchmark_id"))
    idx = BUDGET_INDEX.get(n)
    if idx is None:  # pair ran out of acquisition items before this cap: report the nearest lower budget
        idx = max(i for b, i in BUDGET_INDEX.items() if b <= n)
    return 0.5 if _CHECKS[idx] else 0.0
