#!/usr/bin/env bash
# One-click Figure 2 replication.
#
# Usage:
#   ./run.sh <model_path_or_hf_id> [output_dir]
#
# Example (local weights):
#   ./run.sh /opt/tiger/Flame/Qwen3-8B
#
# Example (HuggingFace ID -- auto-downloads):
#   ./run.sh Qwen/Qwen3-8B
#
# What it does:
#   1. Extract per-item gold log-probabilities + hidden states from the model
#      on the 650-item atomic-capacity dataset (no-CoT + CoT).
#   2. Derive capacity labels (per (family_id, capability, cot) judgments).
#   3. Roll up per-cell |mean Delta log p / token| stats into summary.json.
#   4. Plot the 3 x 9 atomic-capacity matrix (Figure 2 in the paper).
#
# Output:
#   runs/<model_name>/model_outputs.jsonl
#   runs/<model_name>/capability_labels.json
#   runs/<model_name>/summary.json
#   figures/<model_name>/fig_capacity_matrix.png
#
# Hardware: 1 GPU (H100 / A100 recommended; smaller models fit on 24GB).
# Time:    ~5 min for 8B-class models.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path_or_hf_id> [output_dir]"
    exit 1
fi

MODEL_PATH="$1"
MODEL_NAME="$(basename "$MODEL_PATH")"
OUT_DIR="${2:-runs/${MODEL_NAME}}"
DATASET="runs/full_25/output/dataset.jsonl"

mkdir -p "$OUT_DIR" figures

echo "=== [1/4] Extract hidden states + log-probs for ${MODEL_NAME} ==="
python -m probing_mvp.extract_hidden_states \
    --model_name "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUT_DIR" \
    --device cuda \
    --dtype float16 \
    --max_length 1024

echo ""
echo "=== [2/4] Derive capacity labels ==="
python -m probing_mvp.derive_labels \
    --dataset "$DATASET" \
    --model_outputs "${OUT_DIR}/model_outputs.jsonl" \
    --output "${OUT_DIR}/capability_labels.json" \
    --mode natural

echo ""
echo "=== [3/4] Build summary.json ==="
python -m probing_mvp.make_summary \
    --run_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH"

echo ""
echo "=== [4/4] Render Figure 2 (3 x 9 atomic-capacity matrix) ==="
python -m probing_mvp.make_plots \
    "${OUT_DIR}/summary.json" \
    --output-dir figures

echo ""
echo "Done."
echo "  summary:  ${OUT_DIR}/summary.json"
echo "  figure:   figures/${MODEL_NAME}/fig_capacity_matrix.png"
