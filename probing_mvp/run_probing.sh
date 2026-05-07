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

echo
echo "=== done ==="
echo "results in $OUT/"
ls -la "$OUT" | sed 's/^/  /'
