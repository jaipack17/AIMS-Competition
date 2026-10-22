"""GPU-hour estimates for embedding the training items with the Agent Psychometrics recipe.

Token counts: per benchmark, unique-text bytes from parquet footers (item_text_sizes.csv),
/4.3 bytes per token (measured with the DeepSeek-R1-Distill-Qwen tokenizer on 8 benchmarks),
+40 template tokens per item, each item capped at 8192 tokens (left truncation),
at most CAP sampled items per benchmark. Dropped (wrong-direction) benchmarks excluded.
Compute: FLOPs/seq = 2*P*L + 2*layers*d*L^2 (causal attention), at an assumed achieved throughput.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DROP = {"wmt_mqm", "llmail_inject", "harmbench", "emoji_attack", "crow", "jailbreakbench", "advprompter",
        "alignment_faking", "tensortrust", "safeagentbench", "braveguard", "dbpa", "os_harm",
        "helm_bold", "helm_real_toxicity_prompts", "oasst", "data_juicer2", "engibench",
        "classroom_ai", "das_med_hallucination", "preference_dissection", "xstest"}
BYTES_PER_TOKEN, TEMPLATE_TOKENS, MAXLEN = 4.3, 40, 8192
# name, params (B), layers, hidden, paper Table 8 AUC
MODELS = [("Qwen3-VL-4B-Instruct", 4.4, 36, 2560, 0.814), ("gemma-3-4b-it", 3.9, 34, 2560, 0.792),
          ("DeepSeek-R1-Distill-Qwen-7B", 7.6, 28, 3584, 0.801), ("Qwen3-VL-8B-Instruct", 8.2, 36, 4096, 0.816),
          ("DeepSeek-R1-Distill-Llama-8B", 8.0, 32, 4096, 0.807), ("gemma-3-12b-it", 11.8, 48, 3840, 0.812),
          ("DeepSeek-R1-Distill-Qwen-14B", 14.8, 48, 5120, 0.821), ("gemma-3-27b-it", 27.0, 62, 5376, 0.815),
          ("DeepSeek-R1-Distill-Qwen-32B", 32.8, 64, 5120, 0.824)]
# assumed ACHIEVED bf16 TFLOPS for batch-1 HF forward passes, and memory (GB)
GPUS = {"A100 40GB": (125.0, 40), "A10G 24GB": (35.0, 24), "L4 24GB": (35.0, 24)}


def items_tokens(cap):
    d = pd.read_csv(os.path.join(HERE, "item_text_sizes.csv"))
    d = d[(d.n_items > 0) & ~d.bench.isin(DROP)]
    n = np.minimum(d.n_items, cap)
    per_item = np.minimum(d.content_bytes / d.n_items.clip(lower=1) / BYTES_PER_TOKEN + TEMPLATE_TOKENS, MAXLEN)
    return d.bench.values, n.values, per_item.values


if __name__ == "__main__":
    for cap in (2000, 1000, 500):
        b, n, L = items_tokens(cap)
        print(f"\nCAP {cap}/benchmark: {len(b)} benchmarks, {int(n.sum()):,} items, "
              f"{(n * L).sum() / 1e6:,.0f}M tokens, mean {(n * L).sum() / n.sum():.0f} tokens/item")
        rows = []
        for name, P, nl, dm, auc in MODELS:
            flops = (n * (2 * P * 1e9 * L + 2 * nl * dm * L ** 2)).sum()
            mem = 2 * P + 4
            row = {"model": name, "params_B": P, "AUC(paper)": auc, "bf16_GB": round(mem)}
            for g, (tf, gb) in GPUS.items():
                row[g] = f"{flops / (tf * 1e12) / 3600:.0f} h" if mem <= gb else "does not fit"
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
