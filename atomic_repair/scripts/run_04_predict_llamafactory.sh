#!/usr/bin/env bash
# Run prediction on the eval set with the LoRA adapter from run_03.
set -euo pipefail

LF_DIR="${LF_DIR:-$HOME/LLaMA-Factory}"
CFG="$(cd "$(dirname "$0")/.." && pwd)/configs/qwen3_8b_repair_lora_predict.yaml"

if [ ! -d "$LF_DIR" ]; then
  echo "LLaMA-Factory not found at $LF_DIR; set LF_DIR or clone first" >&2
  exit 1
fi

cd "$LF_DIR"
llamafactory-cli train "$CFG"
