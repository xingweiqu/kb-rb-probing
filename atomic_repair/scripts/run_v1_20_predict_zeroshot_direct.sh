#!/usr/bin/env bash
# A zero-shot direct
set -euo pipefail
LF_DIR="${LF_DIR:-$HOME/LLaMA-Factory}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$REPO_DIR/configs/v1/qwen3_8b_zeroshot_direct_predict.yaml"
[ -d "$LF_DIR" ] || { echo "LLaMA-Factory not found at $LF_DIR; set LF_DIR" >&2; exit 1; }
cd "$REPO_DIR"
llamafactory-cli train "$CFG"
