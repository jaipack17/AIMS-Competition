"""Shared code of our PAIEC submissions: subject -> theta, item -> b_hat. Imported by model.py.

theta   = theta_name + theta_harness + theta_provider + theta_effort, from our version-B IRT fit on all 152
          training benchmarks ("final" fit, theta.json). Fields are normalised exactly as in training
          (pipeline/irt_build_data.py); a value never seen in training contributes 0, except a name when
          name_date_prior.json is bundled: then an unseen name contributes a + b * (release date in years since
          2024, clipped), or 0 if the date is blank or unreadable.
b_hat   = ridge(StandardScaler(embedding)), trained on that fit's item difficulties (ridge.npz), or, when mlp.npz is
          bundled instead, the average of small MLPs (pipeline/mlp_b_train.py, final fit), run here in numpy.
MIRT    (only when mirt.json is bundled; pipeline/mirt_fit.py, pipeline/mirt_ridge.py): ridge.npz then has one row per
          item parameter [b, w1..wK] and the logit of base_logit() becomes
            nested:  theta + u(subject) . w_hat(item) - b_hat        (u = sum of the parts' K-vectors in mirt.json)
            m2pl:    theta_vec(subject) . w_hat(item) - b_hat         (theta_vec = sum of the parts' K-vectors)
          Without mirt.json, base_logit() = theta - b_hat exactly as before.
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
# difficulty model: mlp.npz (an MLP ensemble, 1D only) if bundled, else ridge.npz
_MLP_PATH = HERE / "mlp.npz"
MLP: dict | None = None
if _MLP_PATH.exists():
    with np.load(_MLP_PATH) as _m:
        MLP = {k: _m[k] for k in _m.files}
    EMB_DIM = MLP["mean"].shape[0]
    # mean training b: plays the ridge intercept's roles (smoke-test b_hat, default stretch centre)
    RIDGE_INTERCEPTS = np.array([float(MLP["center"])])
else:
    with np.load(HERE / "ridge.npz") as _r:
        RIDGE_MEAN, RIDGE_SCALE = _r["mean"], _r["scale"]
        RIDGE_COEF = np.atleast_2d(_r["coef"])           # (T, 4096): T = 1 (b) or 1 + K (b, w1..wK) for MIRT
        RIDGE_INTERCEPTS = np.atleast_1d(_r["intercept"]).astype(np.float64)
    EMB_DIM = RIDGE_COEF.shape[1]
RIDGE_INTERCEPT = float(RIDGE_INTERCEPTS[0])             # of b
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
# optional MIRT subject vectors {"form": "nested" | "m2pl", "K": K, "subject": {part: {key: [K floats]}}};
# without the file the model is the 1D one
_MIRT_PATH = HERE / "mirt.json"
MIRT: dict | None = None
if _MIRT_PATH.exists():
    with open(_MIRT_PATH, encoding="utf-8") as _f:
        MIRT = json.load(_f)
    assert MLP is None, "the MLP predicts b only: no MIRT with mlp.npz"
    assert RIDGE_COEF.shape[0] == 1 + MIRT["K"], "ridge.npz must have one row per MIRT item parameter"
_STATE: dict | None = None


# --------------------------------------------------------------------------- subject -> theta
HARNESS_MERGE = {"openai-codex": "codex", "codex-cli": "codex"}


def _norm_harness(h: str) -> str:  # = irt_build_data.norm_harness
    h = re.sub(r"[\s_]+", "-", h.strip().lower())
    h = re.sub(r"-v?\d+(\.\d+)+$", "", h)
    return HARNESS_MERGE.get(h, h)


def _lower(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip().lower())


def _subject_keys(subject: dict) -> dict:
    provider = _lower(str(subject.get("provider") or ""))
    return {
        "name": str(subject.get("normalized_name") or "").strip(),
        "harness": _norm_harness(str(subject.get("harness") or "")),
        "provider": "" if provider == "unknown" else provider,
        "effort": _lower(str(subject.get("reasoning_effort") or "")),
    }


def subject_vec(subject: dict) -> np.ndarray:
    """MIRT: sum over the parts of their K-vectors (u for nested, theta for m2pl); unseen or blank keys add 0."""
    v = np.zeros(MIRT["K"])
    for part, key in _subject_keys(subject).items():
        if key and key in MIRT["subject"].get(part, {}):
            v += np.asarray(MIRT["subject"][part][key], dtype=np.float64)
    return v


def subject_theta(subject: dict) -> float:
    keys = _subject_keys(subject)
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


_erf = np.frompyfunc(math.erf, 1, 1)


def mlp_b(x: np.ndarray) -> float:
    """b_hat of the MLP ensemble for one embedding, as pipeline/mlp_b_train.Net in eval mode (no dropout, no input
    noise): standardise; per hidden layer Linear -> LayerNorm (eps 1e-5) -> exact GELU; Linear; then
    * y_sd + y_mean, averaged over the nets."""
    z0 = (x - MLP["mean"]) / MLP["scale"]
    outs = []
    for i in range(int(MLP["n_nets"])):
        z, j = z0, 0
        while f"n{i}_W{j}" in MLP:
            h = MLP[f"n{i}_W{j}"] @ z + MLP[f"n{i}_b{j}"]
            h = (h - h.mean()) / np.sqrt(h.var() + 1e-5) * MLP[f"n{i}_g{j}"] + MLP[f"n{i}_beta{j}"]
            z = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)).astype(np.float64))
            j += 1
        outs.append(float((MLP[f"n{i}_Wout"] @ z + MLP[f"n{i}_bout"])[0]))
    return float(np.mean(outs)) * float(MLP["y_sd"]) + float(MLP["y_mean"])


def item_params(item: dict) -> np.ndarray:
    """Ridge (or MLP) predictions [b_hat] (1D) or [b_hat, w1_hat..wK_hat] (MIRT); the stretch applies to b_hat
    only."""
    if SMOKE_TEST:
        return RIDGE_INTERCEPTS.copy()
    text = ap.format_qs_instruction(question_statement=str(item.get("item_content") or ""),
                                    instruction=ap.DIFFICULTY_INSTRUCTION)
    path = CACHE_DIR / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".txt")
    if path.exists():
        return np.array(path.read_text().split(), dtype=np.float64)
    x = _embed(text).astype(np.float64)
    if x.shape != (EMB_DIM,) or not np.isfinite(x).all():
        raise RuntimeError(f"bad embedding: shape {x.shape}")
    if MLP is not None:
        out = np.array([mlp_b(x)])
    else:
        out = RIDGE_COEF @ ((x - RIDGE_MEAN) / RIDGE_SCALE) + RIDGE_INTERCEPTS
    out[0] = STRETCH_CENTER + STRETCH_K * (out[0] - STRETCH_CENTER)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # unique name, then atomic rename
    tmp.write_text(" ".join(repr(float(v)) for v in out))
    os.replace(tmp, path)
    return out


def item_b_hat(item: dict) -> float:
    return float(item_params(item)[0])


def base_logit(subject: dict, item: dict) -> float:
    """Logit before any label-based shift: theta - b_hat (1D), or the MIRT logit when mirt.json is bundled."""
    if MIRT is None:
        return subject_theta(subject) - item_b_hat(item)
    p = item_params(item)
    dot = float(subject_vec(subject) @ p[1:])
    if MIRT["form"] == "nested":
        return subject_theta(subject) + dot - float(p[0])
    return dot - float(p[0])


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
