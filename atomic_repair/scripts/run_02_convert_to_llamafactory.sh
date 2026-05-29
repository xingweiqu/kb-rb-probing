#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python convert_to_llamafactory.py \
  --train   data/repair_raw_train.jsonl \
  --eval    data/repair_raw_eval.jsonl \
  --out_dir data
