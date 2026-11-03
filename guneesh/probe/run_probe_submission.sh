#!/usr/bin/env bash
# Build label_update_stretch2_date with the probe feature in place of the last-layer AP embedding, end to end,
# in the background (survives a dropped SSH connection). Every step skips work already done, so rerunning resumes.
#
#   bash probe/run_probe_submission.sh                  # rate prompt, layer 20
#   LAYER=22 bash probe/run_probe_submission.sh         # another layer (own folders)
#   GPU=1 bash probe/run_probe_submission.sh            # choose the GPU (default 0)
#
# Steps (log: probe/logs/probe_sub_L<layer>_<time>.log):
#   0. data from Hugging Face: every IRT fit's items.parquet (training targets) and cells_binary.parquet (simulation)
#   1. probe/extract_feature.py         -> data/emb/qwen3vl8b_rate_L<layer>/   (GPU, the long step)
#   2. pipeline/ridge_train.py          -> data/ridge_rate_L<layer>/            (CPU, fold0..4 + final)
#   3. pipeline/stretch_factor.py       -> data/ridge_rate_L<layer>/<run>/stretch.json
#   4. pipeline/build_submission.py     -> submissions/label_update_stretch2_date_probe_L<layer>(.zip)
#   5. kit checks (if "Starting Kit/tools" exists), then analysis/simulate_alc.py (150 draws, +date) and
#      analysis/paired_alc.py against the ridge (0.2273) and MLP (0.2234) versions on the same draws
set -euo pipefail
cd "$(dirname "$0")/.."           # project root (guneesh/)

if [ "${1:-}" != "--inner" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    mkdir -p probe/logs
    LOG="probe/logs/probe_sub_L${LAYER:-20}_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$0" --inner > "$LOG" 2>&1 &
    echo "$!" > "probe/logs/probe_sub.pid"
    echo "started PID $! on GPU $CUDA_VISIBLE_DEVICES"
    echo "log:    tail -f $LOG"
    echo "stop:   kill $!   (then also: pkill -f extract_feature.py)"
    exit 0
fi

LAYER="${LAYER:-20}"
RUN=versionB_4parts_logitnormal
EMB="data/emb/qwen3vl8b_rate_L${LAYER}"
RIDGE="data/ridge_rate_L${LAYER}"
NAME="label_update_stretch2_date_probe_L${LAYER}"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "0. data"
for f in fold0 fold1 fold2 fold3 fold4 final; do
    [ -f "data/irt/$RUN/$f/items.parquet" ] || NEED=1
done
[ -f data/irt/cells_binary.parquet ] || NEED=1
if [ -n "${NEED:-}" ]; then
    hf download GunGG4/paiec-Qwen8Binstruct-guneesh-data --repo-type dataset --local-dir data \
        --include "irt/$RUN/*" --include "irt/cells_binary.parquet"
fi
ls -l "data/irt/$RUN"/*/items.parquet data/irt/cells_binary.parquet

step "1. extract the feature (rate prompt, layer $LAYER) for every embedded item"
python -u probe/extract_feature.py --prompt rate --layer "$LAYER" --out_dir "$EMB"
[ "$(ls "$EMB"/*.npz | wc -l)" -eq "$(ls data/emb/qwen3vl8b/*.npz | wc -l)" ] || { echo "extraction incomplete"; exit 1; }

step "2. ridge (fold0..4 + final)"
python -u pipeline/ridge_train.py --emb_dir "$EMB" --out_dir "$RIDGE" --runs "$RUN"

step "3. stretch factor"
python -u pipeline/stretch_factor.py --emb_dir "$EMB" --ridge_dir "$RIDGE"

step "4. build the submission"
python pipeline/build_submission.py --variant label_update --name "$NAME" --ridge_dir "$RIDGE" \
    --stretch "$RIDGE/$RUN/stretch.json" --name_prior "data/irt/$RUN/final/name_date_prior.json" \
    --embed_spec "$EMB/spec.json"

step "5. checks and local ALC"
KIT="Starting Kit/tools"
if [ -d "$KIT" ]; then
    python "$KIT/check_submission_zip.py" "submissions/$NAME.zip"
    python "$KIT/run_smoke_test.py" "submissions/$NAME"
else
    echo "($KIT not found: kit checks skipped; run them where the starting kit is)"
fi
python -u analysis/simulate_alc.py --sub "$NAME" --draws 30 --configs +date
python analysis/paired_alc.py "$NAME" regress_label_update_stretch2_date label_update_stretch2_date_mlp
step "ALL DONE: submissions/$NAME.zip"
