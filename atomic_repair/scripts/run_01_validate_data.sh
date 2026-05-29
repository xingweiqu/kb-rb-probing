#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python validate_repair_data.py \
  --train data/repair_raw_train.jsonl \
  --eval  data/repair_raw_eval.jsonl \
  --out   data/data_sanity_report.json
