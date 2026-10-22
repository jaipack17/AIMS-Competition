"""Embedding code copied verbatim from Agent Psychometrics (Ge, Kryvosheieva, Fried, Girit, Hariharan, COLM 2026).

Source: https://github.com/dariakryvosheieva/agent-psychometrics, file utils/embeddings.py,
commit 1c34d34a0ee82c92b1c63deb834495c5440bd362. Copied line ranges: 3-47, 52-140, 175-179, 259-547, 639-678.
Nothing in those ranges is edited. Left out: the SWE-bench/GSO dataset readers, cache-path helpers and the CLI.

MIT License, Copyright (c) 2026 Fulcrum. Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - embedding generation will fail loudly if needed.
    torch = None  # type: ignore[assignment]
try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]
try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast
except Exception:  # pragma: no cover
    AutoConfig = AutoModel = AutoTokenizer = PreTrainedTokenizerFast = None  # type: ignore[assignment]


DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one "
    "floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)

EMBEDDING_TEXT_FORMAT_WITH_SOLUTION = "qs_solution_instruction_v1"
EMBEDDING_TEXT_FORMAT_WITHOUT_SOLUTION = "qs_instruction_v1"

def _require_embedding_dependency(name: str, obj: Any) -> None:
    if obj is None:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install the embedding-generation dependencies "
            "before generating embedding caches."
        )


def seed_everything(seed: int, *, deterministic: bool) -> None:
    _require_embedding_dependency("torch", torch)
    s = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(s))
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(s)
    except Exception:
        pass

    if deterministic:
        set_torch_determinism(True)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def set_torch_determinism(enabled: bool) -> None:
    _require_embedding_dependency("torch", torch)
    on = bool(enabled)
    try:
        torch.use_deterministic_algorithms(on, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(on)
        except Exception:
            pass
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = on
        torch.backends.cudnn.benchmark = not on
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def prompt_signature(instruction: str, *, include_solution: bool = True) -> str:
    h = hashlib.sha1(str(instruction).encode("utf-8")).hexdigest()[:8]
    prefix = "qs_sol_instr" if bool(include_solution) else "qs_instr"
    return f"{prefix}_{h}"


def _sanitize_text(s: str) -> str:
    return "".join((" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch) for ch in (s or ""))


def format_qs_solution_instruction(*, question_statement: str, solution: str, instruction: str) -> str:
    qs = _sanitize_text(str(question_statement or "")).strip()
    sol = _sanitize_text(str(solution or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\nSolution:\n{sol}\n\n{instr}".strip()


def format_qs_instruction(*, question_statement: str, instruction: str) -> str:
    qs = _sanitize_text(str(question_statement or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\n{instr}".strip()


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    question_statement: str
    solution: str


def _load_tokenizer(backbone: str, *, trust_remote_code: bool) -> Any:
    _require_embedding_dependency("transformers", AutoTokenizer)
    try:
        return AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise

        _require_embedding_dependency("huggingface_hub", hf_hub_download)
        tok_json = hf_hub_download(repo_id=backbone, filename="tokenizer.json")
        tok_cfg_path = None
        try:
            tok_cfg_path = hf_hub_download(repo_id=backbone, filename="tokenizer_config.json")
        except Exception:
            pass

        tok_kwargs: Dict[str, Any] = {"tokenizer_file": tok_json}
        extra_special_tokens: Optional[List[str]] = None
        if tok_cfg_path is not None and os.path.exists(tok_cfg_path):
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
                if isinstance(cfg.get(key), str) and cfg.get(key):
                    tok_kwargs[key] = cfg[key]
            if isinstance(cfg.get("model_max_length"), int):
                tok_kwargs["model_max_length"] = cfg["model_max_length"]
            if isinstance(cfg.get("extra_special_tokens"), list):
                extra_special_tokens = [str(x) for x in cfg["extra_special_tokens"]]

        tokenizer = PreTrainedTokenizerFast(**tok_kwargs)
        if extra_special_tokens:
            tokenizer.additional_special_tokens = extra_special_tokens
        return tokenizer


def _try_load_model_class(backbone: str, *, trust_remote_code: bool, model_kwargs: dict) -> torch.nn.Module:
    _require_embedding_dependency("torch", torch)
    _require_embedding_dependency("transformers", AutoModel)
    try:
        import transformers.activations as activations

        if not hasattr(activations, "PytorchGELUTanh") and hasattr(activations, "GELUTanh"):
            activations.PytorchGELUTanh = activations.GELUTanh
    except Exception:
        pass

    errors = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
        except Exception as exc:
            errors.append((class_name, exc))

    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as exc:
        errors.append(("AutoModelForCausalLM", exc))

    try:
        return AutoModel.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as exc:
        errors.append(("AutoModel", exc))

    message = "Failed to load model with any supported auto class:\n" + "\n".join(
        f"- {name}: {type(err).__name__}: {err}" for name, err in errors
    )
    raise RuntimeError(message)


def _select_text_submodel(model: torch.nn.Module) -> torch.nn.Module:
    for attr in ("language_model", "text_model"):
        submodel = getattr(model, attr, None)
        if isinstance(submodel, torch.nn.Module):
            return submodel

    submodel = getattr(model, "model", None)
    if isinstance(submodel, torch.nn.Module) and hasattr(submodel, "get_input_embeddings"):
        return submodel
    return model


def _get_hidden_states_tuple(outputs: Any) -> Any:
    for attr in ("hidden_states", "encoder_hidden_states", "decoder_hidden_states"):
        if hasattr(outputs, attr):
            hidden_states = getattr(outputs, attr)
            if hidden_states is not None:
                return hidden_states
    return None


def _extract_hidden_state(outputs: Any, *, embedding_layer: int) -> torch.Tensor:
    layer = int(embedding_layer)

    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    hidden_states = _get_hidden_states_tuple(outputs)
    if hidden_states is not None:
        try:
            return hidden_states[layer]
        except Exception as exc:
            raise RuntimeError(
                f"Requested embedding_layer={layer}, but model returned {len(hidden_states)} hidden_states entries. "
                f"Try a value in [-{len(hidden_states)}, {len(hidden_states) - 1}] or use --embedding_layer -1."
            ) from exc

    raise RuntimeError(
        "Model outputs did not expose hidden states. Try --embedding_layer -1 and ensure the model supports "
        "output_hidden_states=True."
    )


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)


def _torch_dtype_arg(torch_dtype: str) -> Any:
    _require_embedding_dependency("torch", torch)
    if torch_dtype == "auto":
        return "auto"
    if torch_dtype in ("float16", "fp16"):
        return torch.float16
    if torch_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if torch_dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {torch_dtype}")


def embed_items(
    *,
    items: List[ItemRecord],
    backbone: str,
    trust_remote_code: bool,
    max_length: int,
    batch_size: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    instruction: str,
    embedding_layer: int,
    include_solution: bool = True,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int], int]:
    _require_embedding_dependency("torch", torch)
    _require_embedding_dependency("transformers", AutoModel)
    _require_embedding_dependency("tqdm", tqdm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    dtype_arg = _torch_dtype_arg(torch_dtype)
    fp_params = inspect.signature(AutoModel.from_pretrained).parameters
    model_kwargs: Dict[str, Any] = {"dtype" if "dtype" in fp_params else "torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        cfg = AutoConfig.from_pretrained(backbone, trust_remote_code=trust_remote_code)
        quantization_config = getattr(cfg, "quantization_config", None)
        if isinstance(quantization_config, dict) and str(quantization_config.get("quant_method", "")).lower() == "fp8":
            try:
                from transformers import FineGrainedFP8Config

                model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            except Exception:
                pass
    except Exception:
        pass

    model = _try_load_model_class(backbone, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)
    model.eval()
    if device_map in ("", "none", None):
        model.to(device)

    text_model = _select_text_submodel(model)
    for module in (model, text_model):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = False
            except Exception:
                pass

    try:
        embed_device = text_model.get_input_embeddings().weight.device
    except Exception:
        embed_device = device

    per_id: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    embedding_dim = 0

    def flush() -> None:
        nonlocal batch_ids, batch_texts, embedding_dim
        if not batch_texts:
            return

        pairs = [(rid, txt) for rid, txt in zip(batch_ids, batch_texts) if str(txt).strip()]
        batch_ids = []
        batch_texts = []
        if not pairs:
            return

        ids = [rid for rid, _ in pairs]
        texts = [txt for _, txt in pairs]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if int(input_ids.shape[1]) == 0:
            return

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        want_hidden_states = int(embedding_layer) != -1

        with torch.inference_mode():
            fwd_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
                "output_hidden_states": bool(want_hidden_states),
            }
            try:
                sig = inspect.signature(text_model.forward)
                if "use_cache" in sig.parameters:
                    fwd_kwargs["use_cache"] = False
            except Exception:
                fwd_kwargs["use_cache"] = False

            outputs = text_model(**fwd_kwargs)
            try:
                hidden = _extract_hidden_state(outputs, embedding_layer=int(embedding_layer))
            except RuntimeError:
                if want_hidden_states:
                    raise
                fwd_kwargs["output_hidden_states"] = True
                outputs = text_model(**fwd_kwargs)
                hidden = _extract_hidden_state(outputs, embedding_layer=int(embedding_layer))

            pooled = last_token_pool(hidden, attention_mask).detach().float().cpu().numpy()

        embedding_dim = int(pooled.shape[1])
        for item_id, vec, text in zip(ids, pooled, texts):
            per_id[str(item_id)] = vec.astype(np.float32, copy=False)
            counts[str(item_id)] = int(len(str(text)))

    for rec in tqdm(items, desc="embed_items"):
        if bool(include_solution):
            text = format_qs_solution_instruction(
                question_statement=rec.question_statement,
                solution=rec.solution,
                instruction=instruction,
            )
        else:
            text = format_qs_instruction(
                question_statement=rec.question_statement,
                instruction=instruction,
            )
        if not text.strip():
            continue
        batch_ids.append(rec.item_id)
        batch_texts.append(text)
        if len(batch_texts) >= int(batch_size):
            flush()
    flush()

    ids_sorted = sorted(per_id.keys())
    return ids_sorted, per_id, counts, int(embedding_dim)


def save_embeddings_cache(
    *,
    path: str,
    task_ids: Sequence[str],
    embeddings_by_id: Dict[str, np.ndarray],
    counts_by_id: Dict[str, int],
    embedding_dim: int,
    dataset_sources: str,
    split: str,
    dataset_path: str,
    instruction: str,
    backbone: str,
    max_length: int,
    embedding_layer: int,
    include_solution: bool = True,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ids_sorted = list(task_ids)
    X = np.stack([embeddings_by_id[item_id] for item_id in ids_sorted], axis=0).astype(np.float32)
    counts_arr = np.array([int(counts_by_id.get(item_id, 0)) for item_id in ids_sorted], dtype=np.int64)
    text_format = EMBEDDING_TEXT_FORMAT_WITH_SOLUTION if bool(include_solution) else EMBEDDING_TEXT_FORMAT_WITHOUT_SOLUTION
    np.savez_compressed(
        path,
        task_ids=np.array(ids_sorted, dtype=object),
        X=X,
        counts_kind=np.array(["text_len_chars"], dtype=object),
        counts=counts_arr,
        dataset_name=np.array([str(dataset_sources)], dtype=object),
        split=np.array([str(split)], dtype=object),
        dataset_path=np.array([str(dataset_path)], dtype=object),
        n_items=np.array([int(len(ids_sorted))], dtype=np.int64),
        instruction=np.array([str(instruction)], dtype=object),
        instruction_signature=np.array([str(prompt_signature(instruction, include_solution=bool(include_solution)))], dtype=object),
        text_format=np.array([str(text_format)], dtype=object),
        includes_solution=np.array([bool(include_solution)], dtype=np.bool_),
        backbone=np.array([str(backbone)], dtype=object),
        max_length=np.array([int(max_length)], dtype=np.int64),
        embedding_dim=np.array([int(embedding_dim)], dtype=np.int64),
        embedding_layer=np.array([int(embedding_layer)], dtype=np.int64),
    )
