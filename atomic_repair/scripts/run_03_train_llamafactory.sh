#!/usr/bin/env bash
# Launch LLaMA-Factory SFT for atomic-repair on a server with GPUs.
# Run this AFTER cloning LLaMA-Factory and installing its deps. See README.
set -euo pipefail

LF_DIR="${LF_DIR:-$HOME/LLaMA-Factory}"
CFG="$(cd "$(dirname "$0")/.." && pwd)/configs/qwen3_8b_repair_lora_sft.yaml"

if [ ! -d "$LF_DIR" ]; then
  echo "LLaMA-Factory not found at $LF_DIR; set LF_DIR or clone first" >&2
  exit 1
fi

cd "$LF_DIR"
llamafactory-cli train "$CFG"
