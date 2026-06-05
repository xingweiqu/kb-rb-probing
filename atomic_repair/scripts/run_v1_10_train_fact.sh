#!/usr/bin/env bash
# B: Fact-only SFT from Instruct -> output_v1/fact_only
set -euo pipefail
LF_DIR="${LF_DIR:-$HOME/LLaMA-Factory}"
NPROC="${NPROC:-8}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$REPO_DIR/configs/v1/qwen3_8b_fact_sft.yaml"
[ -d "$LF_DIR" ] || { echo "LLaMA-Factory not found at $LF_DIR; set LF_DIR" >&2; exit 1; }
cd "$REPO_DIR"
FORCE_TORCHRUN=1 NPROC_PER_NODE="$NPROC" llamafactory-cli train "$CFG"
