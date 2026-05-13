#!/usr/bin/env bash
# One-click Figure 2 replication.
#
# Usage:
#   ./run.sh <model_path_or_hf_id> [output_dir]
#
# Example (local weights):
#   ./run.sh /opt/tiger/Flame/Qwen3-8B
#
# Example (HuggingFace ID — auto-downloads):
#   ./run.sh Qwen/Qwen3-8B
#
# What it does:
#   1. Extract hidden states + per-item gold log-probabilities from the model
#      on the 650-item atomic-capacity dataset.
#   2. Summarise per-cell |mean Δlog p / token| into summary.json.
#   3. Plot the 3×9 atomic-capacity matrix (Figure 2 in the paper).
#
# Output:
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
FIG_DIR="figures/${MODEL_NAME}"

mkdir -p "$OUT_DIR" "$FIG_DIR"

echo "=== [1/3] Extracting hidden states + log-probs for ${MODEL_NAME} ==="
python -m probing_mvp.extract_hidden_states \
    --model_name "$MODEL_PATH" \
    --dataset runs/full_25/output/dataset.jsonl \
    --output_dir "$OUT_DIR" \
    --device cuda \
    --dtype float16 \
    --max_length 1024

echo ""
echo "=== [2/3] Building summary.json ==="
python -m probing_mvp.make_summary "$OUT_DIR"

echo ""
echo "=== [3/3] Rendering Figure 2 (atomic-capacity matrix) ==="
python -m probing_mvp.make_plots "${OUT_DIR}/summary.json" --output-dir figures

echo ""
echo "Done."
echo "  summary:  ${OUT_DIR}/summary.json"
echo "  figure:   ${FIG_DIR}/fig_capacity_matrix.png"
