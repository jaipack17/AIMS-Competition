#!/usr/bin/env bash
# Start probe/extract_activations.py in the background so it survives a dropped SSH connection.
#
#   bash probe/run_extract.sh pilot                 # 30 benchmarks x 100 items
#   bash probe/run_extract.sh full --max_hours 10   # any extra arguments are passed on
#   GPU=1 bash probe/run_extract.sh pilot           # choose the GPU (default 0)
#
# Rerunning the same command resumes: finished benchmarks are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."           # project root (guneesh/)

MODE="${1:-pilot}"; shift || true
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p probe/logs
LOG="probe/logs/extract_${MODE}_$(date +%Y%m%d_%H%M%S).log"
nohup python -u probe/extract_activations.py --mode "$MODE" "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "probe/logs/extract_${MODE}.pid"
echo "started PID $PID on GPU $CUDA_VISIBLE_DEVICES"
echo "log:    tail -f $LOG"
echo "stop:   kill $PID"
