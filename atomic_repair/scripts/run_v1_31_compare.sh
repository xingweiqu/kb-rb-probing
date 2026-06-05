#!/usr/bin/env bash
# Build the final comparison report from the per-condition eval reports.
set -euo pipefail
cd "$(dirname "$0")/.."
R=data_v1/reports
python compare_conditions.py --out data_v1/comparison_v1 \
  --fact-sanity-base "$R/fact_sanity_base.json" \
  --fact-gate "$R/fact_gate.json" \
  --zeroshot-direct "$R/zeroshot_direct.json" \
  --zeroshot-cot "$R/zeroshot_cot.json" \
  --zeroshot-skillcot "$R/zeroshot_skillcot.json" \
  --factonly-repair "$R/factonly_repair.json" \
  --fact-then-cot "$R/fact_then_cot.json" \
  --fact-then-skillcot "$R/fact_then_skillcot.json"
echo "comparison -> data_v1/comparison_v1.md"
