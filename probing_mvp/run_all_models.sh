#!/usr/bin/env bash
# Dispatcher: run probing for many models in parallel, one model per GPU.
#
# Usage:
#   ./probing_mvp/run_all_models.sh model1 [model2 ...]
#
# Examples:
#   ./probing_mvp/run_all_models.sh \
#       /opt/tiger/ouro2/Qwen3-0.5B \
#       /opt/tiger/ouro2/Qwen3-1.7B \
#       /opt/tiger/ouro2/Qwen3-4B \
#       /opt/tiger/ouro2/Qwen3-8B \
#       /opt/tiger/ouro2/Qwen3-8B-Base
#
# Each model runs ./probing_mvp/run_probing.sh with CUDA_VISIBLE_DEVICES
# pinned to a single GPU id, in the background. Output goes to
# logs/<run_name>.log.
#
# GPU assignment: model i goes to GPU (i % NGPU). Default NGPU=8. Override
# with the GPUS env var (e.g. GPUS="0,2,4,6"). If you have more models than
# GPUs, the extra models queue up on the same GPU (sequentially-ish; PyTorch
# will share the device).
#
# GrabGPU: this dispatcher does NOT touch the GrabGPU keepalive logic inside
# run_probing.sh. It does pass GRABGPU_ENABLE=0 to children when run in
# parallel, because the keepalive arg "0,1,2,3,4,5,6,7" assumes the script
# owns all 8 cards — that's wrong when 8 children share the box. Keep
# GrabGPU outside the dispatcher (start it before, kill it after).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 model_path [model_path ...]" >&2
  exit 1
fi

GPUS_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS_ARR <<< "$GPUS_LIST"
NGPU="${#GPUS_ARR[@]}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "=== model dispatcher ==="
echo "models:    $#"
echo "GPUs:      $GPUS_LIST  (count=$NGPU)"
echo "logs:      $LOG_DIR/"
echo

declare -a PIDS=()
declare -a NAMES=()

for ((i=0; i<$#; i++)); do
  MODEL="${@:$((i+1)):1}"
  RUN_NAME="$(basename "$MODEL")"
  GPU_ID="${GPUS_ARR[$((i % NGPU))]}"
  LOG="$LOG_DIR/${RUN_NAME}.log"

  echo "[dispatch] model=$RUN_NAME  gpu=$GPU_ID  log=$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  GRABGPU_ENABLE=0 \
  nohup ./probing_mvp/run_probing.sh "$MODEL" "$RUN_NAME" \
    > "$LOG" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$RUN_NAME")
done

echo
echo "=== all dispatched, waiting... ==="
echo "(tail -f $LOG_DIR/<name>.log to follow any one)"
echo

FAILED=()
for ((i=0; i<${#PIDS[@]}; i++)); do
  PID="${PIDS[$i]}"
  NAME="${NAMES[$i]}"
  if wait "$PID"; then
    echo "[done] $NAME  exit=0"
  else
    code=$?
    echo "[FAIL] $NAME  exit=$code  (see $LOG_DIR/${NAME}.log)"
    FAILED+=("$NAME")
  fi
done

echo
echo "=== summary ==="
echo "completed: ${#PIDS[@]}"
echo "failed:    ${#FAILED[@]}  ${FAILED[*]:-}"
echo
echo "summary files to push back:"
for NAME in "${NAMES[@]}"; do
  if [[ -f "runs/${NAME}/summary.json" ]]; then
    echo "  runs/${NAME}/summary.json"
  fi
done
