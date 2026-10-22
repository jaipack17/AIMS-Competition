"""Shared code of our PAIEC submissions: subject -> theta, item -> b_hat. Imported by model.py.

theta   = theta_name + theta_harness + theta_provider + theta_effort, from our version-B IRT fit on all 152
          training benchmarks ("final" fit, theta.json). Fields are normalised exactly as in training
          (pipeline/irt_build_data.py); a value never seen in training contributes 0, except a name when
          name_date_prior.json is bundled: then an unseen name contributes a + b * (release date in years since
          2024, clipped), or 0 if the date is blank or unreadable.
b_hat   = ridge(StandardScaler(embedding)), trained on that fit's item difficulties (ridge.npz).
          embedding = Qwen3-VL-8B-Instruct, last layer, last token, of AP's no-solution prompt
          "Task statement:\n{item_content}\n\n{instruction}", 8192 tokens with left truncation, bfloat16:
          the same code path (ap_embed_core.py, verbatim AP code) that made the training embeddings.
  * load_model() raises if the model cannot be loaded (no silent fallback).
  * With PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1 (set by the kit's local check tools while importing) no model is
    loaded and b_hat is the ridge intercept, i.e. the average training difficulty. Local tools only.
  * If bhat_stretch.json is bundled, b_hat is stretched around the training mean by its factor k.
  * b_hat is cached in the submission directory, one small file per item text (sha256 name), because
    workers restart at every label budget and the evaluation items are the same at every budget.
"""
from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
import os
import re
from pathlib import Path

import numpy as np

import ap_embed_core as ap  # verbatim AP embedding functions, bundled next to this file

HERE = Path(__file__).resolve().parent

LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
# read once at import: the kit's local tools set this variable only while importing model.py
SMOKE_TEST = os.environ.get(LOCAL_SMOKE_TEST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
MAX_LENGTH = 8192
CACHE_DIR = HERE / "bhat_cache"

with open(HERE / "theta.json", encoding="utf-8") as _f:
    THETA = json.load(_f)                      # {"name": {key: theta}, "harness": {...}, ...}
with np.load(HERE / "ridge.npz") as _r:
    RIDGE_MEAN, RIDGE_SCALE, RIDGE_COEF = _r["mean"], _r["scale"], _r["coef"]
    RIDGE_INTERCEPT = float(_r["intercept"])
# optional stretch of b_hat around the training mean, b = center + k * (b_hat - center) (pipeline/stretch_factor.py);
# without the file b_hat is used as the ridge gives it
_STRETCH_PATH = HERE / "bhat_stretch.json"
if _STRETCH_PATH.exists():
    with open(_STRETCH_PATH, encoding="utf-8") as _f:
        _st = json.load(_f)
    STRETCH_CENTER, STRETCH_K = float(_st["center"]), float(_st["k"])
else:
    STRETCH_CENTER, STRETCH_K = RIDGE_INTERCEPT, 1.0
# optional prior for a name never seen in training, theta_name = a + b * years since 2024 of release_date
# (pipeline/name_date_prior.py); without the file such a name contributes 0
_NAME_PRIOR_PATH = HERE / "name_date_prior.json"
NAME_PRIOR: dict | None = None
if _NAME_PRIOR_PATH.exists():
    with open(_NAME_PRIOR_PATH, encoding="utf-8") as _f:
        NAME_PRIOR = json.load(_f)
_STATE: dict | None = None


# --------------------------------------------------------------------------- subject -> theta
HARNESS_MERGE = {"openai-codex": "codex", "codex-cli": "codex"}


def _norm_harness(h: str) -> str:  # = irt_build_data.norm_harness
    h = re.sub(r"[\s_]+", "-", h.strip().lower())
    h = re.sub(r"-v?\d+(\.\d+)+$", "", h)
    return HARNESS_MERGE.get(h, h)


def _lower(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip().lower())


def subject_theta(subject: dict) -> float:
    provider = _lower(str(subject.get("provider") or ""))
    keys = {
        "name": str(subject.get("normalized_name") or "").strip(),
        "harness": _norm_harness(str(subject.get("harness") or "")),
        "provider": "" if provider == "unknown" else provider,
        "effort": _lower(str(subject.get("reasoning_effort") or "")),
    }
    theta = float(sum(THETA[part].get(key, 0.0) for part, key in keys.items() if key))
    if NAME_PRIOR is not None and keys["name"] not in THETA["name"]:
        years = _release_years(str(subject.get("release_date") or ""))
        if years is not None:
            years = min(max(years, NAME_PRIOR["years_min"]), NAME_PRIOR["years_max"])
            theta += NAME_PRIOR["a"] + NAME_PRIOR["b"] * years
    return theta


def subject_theta_var(subject: dict) -> float:
    """Extra uncertainty of theta from an unseen name: the leave-one-name-out mean squared error of the date line
    (or of 0 when the date is blank / unreadable). 0 for a known name, or when name_date_prior.json is absent."""
    if NAME_PRIOR is None or str(subject.get("normalized_name") or "").strip() in THETA["name"]:
        return 0.0
    dated = _release_years(str(subject.get("release_date") or "")) is not None
    return float(NAME_PRIOR["loo_rmse_line"] if dated else NAME_PRIOR["loo_rmse_zero"]) ** 2


_DATE = re.compile(r"\s*(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?!\d)")


def _release_years(release_date: str) -> float | None:  # = pipeline/name_date_prior.release_years
    """'YYYY-MM-DD' -> years since 2024-01-01; 'YYYY-MM' -> the 15th of that month; 'YYYY' -> 1 July; else None."""
    m = _DATE.match(release_date)
    if m is None:
        return None
    month = int(m[2]) if m[2] else 7
    day = int(m[3]) if m[3] else (15 if m[2] else 1)
    try:
        d = datetime.date(int(m[1]), month, day)
    except ValueError:
        return None
    return (d - datetime.date(2024, 1, 1)).days / 365.25


# --------------------------------------------------------------------------- item -> b_hat
def _hf_cache_dir() -> str | None:  # = starting-kit hf_submission template
    for candidate in [os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(HERE / ".hf_cache")]:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


def load_model() -> None:
    """pipeline/embed_manifest.load_backbone (AP embed_items setup), reading only the local HF cache.

    AP's loader tries AutoModelForImageTextToText first, which is the class that loads Qwen3-VL; it is called here
    directly with the literal repo ID declared in models.txt and without remote code (platform rule).
    """
    import torch
    from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU: Qwen3-VL-8B embeddings need a GPU")
    ap.seed_everything(0, deterministic=True)
    cache_dir = _hf_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", cache_dir=cache_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    fp_params = inspect.signature(AutoModel.from_pretrained).parameters
    kwargs = {"dtype" if "dtype" in fp_params else "torch_dtype": torch.bfloat16,
              "cache_dir": cache_dir, "local_files_only": True}
    try:
        import accelerate  # noqa: F401  (device_map needs it)
        kwargs["device_map"] = "auto"
    except ImportError:
        pass
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", **kwargs)
    model.eval()
    if "device_map" not in kwargs:
        model.to("cuda")
    text_model = ap._select_text_submodel(model)
    for module in (model, text_model):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    global _STATE
    _STATE = {"torch": torch, "tokenizer": tokenizer, "text_model": text_model,
              "device": text_model.get_input_embeddings().weight.device}


def _embed(text: str) -> np.ndarray:
    """pipeline/embed_manifest.embed_batch (AP embed_items flush) for one text."""
    if _STATE is None:
        load_model()
    s = _STATE
    enc = s["tokenizer"]([text], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    ids, mask = enc["input_ids"].to(s["device"]), enc["attention_mask"].to(s["device"])
    kwargs = {"input_ids": ids, "attention_mask": mask, "return_dict": True, "output_hidden_states": False}
    if "use_cache" in inspect.signature(s["text_model"].forward).parameters:
        kwargs["use_cache"] = False
    with s["torch"].inference_mode():
        hidden = ap._extract_hidden_state(s["text_model"](**kwargs), embedding_layer=-1)
        pooled = ap.last_token_pool(hidden, mask)
    return pooled.float().cpu().numpy()[0]


def item_b_hat(item: dict) -> float:
    if SMOKE_TEST:
        return RIDGE_INTERCEPT
    text = ap.format_qs_instruction(question_statement=str(item.get("item_content") or ""),
                                    instruction=ap.DIFFICULTY_INSTRUCTION)
    path = CACHE_DIR / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".txt")
    if path.exists():
        return float(path.read_text())
    x = _embed(text).astype(np.float64)
    if x.shape != RIDGE_COEF.shape or not np.isfinite(x).all():
        raise RuntimeError(f"bad embedding: shape {x.shape}")
    b = float(((x - RIDGE_MEAN) / RIDGE_SCALE) @ RIDGE_COEF + RIDGE_INTERCEPT)
    b = STRETCH_CENTER + STRETCH_K * (b - STRETCH_CENTER)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # unique name, then atomic rename
    tmp.write_text(repr(b))
    os.replace(tmp, path)
    return b


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
