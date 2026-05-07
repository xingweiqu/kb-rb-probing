#!/usr/bin/env bash
# End-to-end probing pipeline.
#
# Usage:
#   ./probing_mvp/run_probing.sh /path/to/model [run_name]
#
# Examples:
#   ./probing_mvp/run_probing.sh /opt/tiger/Flame/Qwen3-8B
#   ./probing_mvp/run_probing.sh /opt/tiger/Flame/Qwen3-8B-Base qwen3_8b_base
#
# Inputs:
#   - dataset at runs/full_25/output/dataset.jsonl (committed)
# Outputs (under runs/<run_name>/):
#   hidden_*.npy / item_index.jsonl / model_outputs.jsonl   (extract step)
#   capability_labels.json                                    (label derivation)
#   linear_probe.json / lora_probe.json                       (probes)
#   probe_geometry.json + .md                                 (geometry)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/model [run_name]" >&2
  exit 1
fi

MODEL="$1"
RUN_NAME="${2:-$(basename "$MODEL")}"
DATASET="${DATASET:-runs/full_25/output/dataset.jsonl}"
OUT="runs/${RUN_NAME}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
COT_STATES="${COT_STATES:-no_cot with_cot}"
POOLS="${POOLS:-last mean}"

# Optional: keep GPUs occupied between runs via GrabGPU.
# Per request: after each probing run, start gg; before each probing run, kill gg.
GRABGPU_DIR="${GRABGPU_DIR:-/opt/tiger/ouro2/GrabGPU}"
GRABGPU_ARGS="${GRABGPU_ARGS:-76 1024 0,1,2,3,4,5,6,7}"
GRABGPU_ENABLE="${GRABGPU_ENABLE:-1}"

kill_grabgpu() {
  if [[ "$GRABGPU_ENABLE" != "1" ]]; then
    return 0
  fi
  # 1) kill previously recorded pid (if any)
  if [[ -f "$OUT/gg.pid" ]]; then
    local pid
    pid="$(cat "$OUT/gg.pid" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[GrabGPU] killing previous gg pid=$pid"
      kill "$pid" 2>/dev/null || true
      # give it a moment, then hard-kill if needed
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  # 2) best-effort cleanup: kill any gg started from GRABGPU_DIR by current user
  if [[ -x "$GRABGPU_DIR/gg" ]]; then
    if pgrep -u "${USER:-$(id -un)}" -f "$GRABGPU_DIR/gg" >/dev/null 2>&1; then
      echo "[GrabGPU] pkill gg under $GRABGPU_DIR"
      pkill -u "${USER:-$(id -un)}" -f "$GRABGPU_DIR/gg" || true
    fi
  fi
}

start_grabgpu() {
  if [[ "$GRABGPU_ENABLE" != "1" ]]; then
    return 0
  fi
  if [[ ! -x "$GRABGPU_DIR/gg" ]]; then
    echo "[GrabGPU] skip: $GRABGPU_DIR/gg not found/executable" >&2
    return 0
  fi
  echo "[GrabGPU] starting: (cd $GRABGPU_DIR && ./gg $GRABGPU_ARGS)"
  # Do not block the probing script.
  (cd "$GRABGPU_DIR" && nohup ./gg $GRABGPU_ARGS >/tmp/gg_${RUN_NAME}.log 2>&1 & echo $! > "$OUT/gg.pid")
  echo "[GrabGPU] pid=$(cat "$OUT/gg.pid" 2>/dev/null || true) log=/tmp/gg_${RUN_NAME}.log"
}

mkdir -p "$OUT"
echo "=== probing pipeline ==="
echo "model:    $MODEL"
echo "dataset:  $DATASET"
echo "out_dir:  $OUT"
echo "device:   $DEVICE / $DTYPE"
echo "cot:      $COT_STATES"
echo "pools:    $POOLS"
echo

echo "[1/4] extracting hidden states..."
kill_grabgpu
python -m probing_mvp.extract_hidden_states \
  --model_name "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUT" \
  --cot_states $COT_STATES \
  --pools $POOLS \
  --device "$DEVICE" \
  --dtype "$DTYPE"

echo "[2/4] deriving capability labels..."
python -m probing_mvp.derive_labels \
  --dataset "$DATASET" \
  --model_outputs "$OUT/model_outputs.jsonl" \
  --output "$OUT/capability_labels.json"

echo "[3/4] linear probes..."
python -m probing_mvp.linear_probe \
  --hidden_dir "$OUT" \
  --labels "$OUT/capability_labels.json" \
  --output "$OUT/linear_probe.json" \
  --cot_states $COT_STATES \
  --pools $POOLS

echo "[3b/4] LoRA probes (rank=16, robustness check)..."
python -m probing_mvp.lora_probe \
  --hidden_dir "$OUT" \
  --labels "$OUT/capability_labels.json" \
  --output "$OUT/lora_probe.json" \
  --cot_states $COT_STATES \
  --pools $POOLS \
  --device "$DEVICE"

echo "[4/4] geometry analysis..."
python -m probing_mvp.geometry_analysis \
  --hidden_dir "$OUT" \
  --output "$OUT/probe_geometry.json" \
  --cot_states $COT_STATES \
  --pools $POOLS

echo "[5/5] writing summary.json..."
python -m probing_mvp.make_summary \
  --run_dir "$OUT" \
  --model_path "$MODEL"

echo
echo "=== done ==="
echo "results in $OUT/"
echo "summary file (commit this back to the repo): $OUT/summary.json"
ls -la "$OUT" | sed 's/^/  /'

start_grabgpu
