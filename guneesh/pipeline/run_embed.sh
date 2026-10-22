#!/usr/bin/env bash
# Embed the training items on a GPU machine with the Agent Psychometrics recipe.
#
#   bash pipeline/run_embed.sh 8b   pilot|parity|full
#   bash pipeline/run_embed.sh 14b  pilot|parity|full
#
# Copy to the GPU machine: pipeline/  and  data/embed_manifest/binary_cap5000.parquet  (no HF token needed;
# both backbones are public). "full" is resumable: rerun the same command after a crash or a time limit and
# finished benchmarks are skipped. Outputs: data/emb/<model>/<benchmark>.npz and _progress.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  8b)  BACKBONE="Qwen/Qwen3-VL-8B-Instruct";            OUT="data/emb/qwen3vl8b" ;;
  14b) BACKBONE="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"; OUT="data/emb/r1qwen14b" ;;
  *) echo "first argument must be 8b or 14b"; exit 1 ;;
esac
MANIFEST="data/embed_manifest/binary_cap5000.parquet"
MAX_HOURS="${MAX_HOURS:-11.5}"   # stop starting new benchmarks after this many hours

# deterministic cuBLAS, as requested by torch.use_deterministic_algorithms in seed_everything()
export CUBLAS_WORKSPACE_CONFIG=:4096:8
COMMON=(--manifest "$MANIFEST" --backbone "$BACKBONE" --max_length 8192 --batch_size 1
        --torch_dtype bfloat16 --device_map auto --embedding_layer -1 --seed 0)

case "${2:-}" in
  pilot)  python pipeline/embed_manifest.py "${COMMON[@]}" --pilot 300 ;;
  parity) python pipeline/embed_manifest.py "${COMMON[@]}" --parity 20 ;;
  full)   python pipeline/embed_manifest.py "${COMMON[@]}" --out_dir "$OUT" --max_hours "$MAX_HOURS" ;;
  *) echo "second argument must be pilot, parity or full"; exit 1 ;;
esac
