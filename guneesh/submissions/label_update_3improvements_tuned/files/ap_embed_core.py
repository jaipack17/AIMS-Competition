"""The parts of Agent Psychometrics' embedding code that our submission calls, copied verbatim.

Source: https://github.com/dariakryvosheieva/agent-psychometrics, utils/embeddings.py, commit
1c34d34a0ee82c92b1c63deb834495c5440bd362 (via pipeline/ap_embeddings.py, which produced the training embeddings).
Only these definitions are kept: DIFFICULTY_INSTRUCTION, _require_embedding_dependency, seed_everything,
set_torch_determinism, _sanitize_text, format_qs_instruction, _select_text_submodel, _get_hidden_states_tuple,
_extract_hidden_state, last_token_pool. Their bodies are unchanged. The model/tokenizer loaders are left out:
core.py loads the declared repo itself (the platform only accepts loading calls with a literal repo ID).

MIT License, Copyright (c) 2026 Fulcrum. Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - embedding generation will fail loudly if needed.
    torch = None  # type: ignore[assignment]

DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one "
    "floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)


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


def _sanitize_text(s: str) -> str:
    return "".join((" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch) for ch in (s or ""))


def format_qs_instruction(*, question_statement: str, instruction: str) -> str:
    qs = _sanitize_text(str(question_statement or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\n{instr}".strip()


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
