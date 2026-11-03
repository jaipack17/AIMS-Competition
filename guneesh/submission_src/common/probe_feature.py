"""Probe feature: a middle-layer hidden state of Qwen3-VL-8B at the last token of a difficulty prompt.

Used by BOTH the training extraction (probe/extract_feature.py) and the submission (core.py, when embed_spec.json is
bundled), so the vector the ridge was trained on and the vector it is applied to come from the same code.

  spec = {"prompt": "rate", "layer": 20, "position": "last", "max_length": 8192}

  prompt    "rate": "Task statement:\\n<item>\\n\\nHow difficult is the above task for an AI model? Answer with a single
            digit from 0 (very easy) to 9 (very hard).\\nDifficulty: "  (as probe/extract_activations.py; the trailing
            space is kept, so the next token the model would write is the digit)
            "ap": Agent Psychometrics' prompt, as the existing embeddings (ap_embed_core.format_qs_instruction)
  layer     hidden_states[layer] in the transformers convention: 0 = token embeddings, i = output of decoder layer i
            (before the final norm). 1 <= layer < number of decoder layers; the last layer (after the final norm)
            is the existing embedding and does not need this module.
  position  "last": last real token (right padding, as ap_embed_core.last_token_pool)
  max_length tokens, left truncation (the tokenizer must have truncation_side "left", padding_side "right")

The forward pass stops right after decoder layer `layer` (a forward hook raises), so the remaining layers are not
computed. Selected by probe/analyze_probes.py + probe/compare_production.py on the full run (probe/README.md).
"""
from __future__ import annotations

import inspect

import ap_embed_core as ap

HEAD = "Task statement:\n"
RATE_INSTRUCTION = ("How difficult is the above task for an AI model? Answer with a single digit from "
                    "0 (very easy) to 9 (very hard).\nDifficulty: ")
DEFAULT_SPEC = {"prompt": "rate", "layer": 20, "position": "last", "max_length": 8192}


def check_spec(spec: dict) -> dict:
    s = {**DEFAULT_SPEC, **spec}
    if s["prompt"] not in ("rate", "ap"):
        raise ValueError(f"unknown prompt {s['prompt']!r}")
    if s["position"] != "last":
        raise ValueError(f"unknown position {s['position']!r}")
    s["layer"], s["max_length"] = int(s["layer"]), int(s["max_length"])
    return s


def build_text(content, prompt: str) -> str:
    """Model input for one item; = probe/extract_activations.build_text."""
    if prompt == "ap":
        return ap.format_qs_instruction(question_statement=str(content or ""), instruction=ap.DIFFICULTY_INSTRUCTION)
    qs = ap._sanitize_text(str(content or "")).strip()
    return f"{HEAD}{qs}\n\n{RATE_INSTRUCTION}"


def decoder_layers(text_model):
    """The ModuleList of decoder layers: text_model.layers (text models) or text_model.language_model.layers
    (Qwen3-VL in recent transformers, where the text submodel is Qwen3VLModel)."""
    import torch
    for owner in (text_model, getattr(text_model, "language_model", None)):
        layers = getattr(owner, "layers", None) if owner is not None else None
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    raise RuntimeError(f"cannot find the decoder layers of {type(text_model).__name__}")


class _Stop(Exception):
    pass


def hidden_at_layer(text_model, input_ids, attention_mask, layer: int):
    """hidden_states[layer] (B, T, H) of one forward pass that stops after decoder layer `layer`."""
    layers = decoder_layers(text_model)
    if not 1 <= layer < len(layers):
        raise ValueError(f"layer {layer} not in 1..{len(layers) - 1}")
    box = {}

    def hook(_module, _inp, out):
        box["h"] = out[0] if isinstance(out, (tuple, list)) else out
        raise _Stop

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
    params = inspect.signature(text_model.forward).parameters
    if "use_cache" in params:
        kwargs["use_cache"] = False
    handle = layers[layer - 1].register_forward_hook(hook)
    try:
        text_model(**kwargs)
    except _Stop:
        pass
    finally:
        handle.remove()
    if "h" not in box:
        raise RuntimeError("the forward pass did not reach the requested layer")
    return box["h"]


def encode(tokenizer, texts: list[str], max_length: int):
    if tokenizer.padding_side != "right" or tokenizer.truncation_side != "left":
        raise RuntimeError("tokenizer must pad on the right and truncate on the left")
    return tokenizer(texts, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")


def embed_texts(tokenizer, text_model, device, texts: list[str], spec: dict):
    """Feature vectors (B, H) as float32 torch tensors on the CPU for full input texts (build_text output)."""
    import torch
    enc = encode(tokenizer, texts, spec["max_length"])
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    with torch.inference_mode():
        h = hidden_at_layer(text_model, ids, mask, spec["layer"])
        pooled = ap.last_token_pool(h, mask.to(h.device))
    return pooled.float().cpu()
