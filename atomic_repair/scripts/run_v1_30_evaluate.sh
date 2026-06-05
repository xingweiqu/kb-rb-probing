#!/usr/bin/env bash
# Local scoring. Run after pulling the prediction files back from the server into
# ./output_v1/predict_*/generated_predictions.jsonl. Produces one eval report each.
set -euo pipefail
cd "$(dirname "$0")/.."

P=output_v1
R=data_v1/reports
mkdir -p "$R"

pred() { echo "$P/$1/generated_predictions.jsonl"; }

# --- gates (fact mode) ---
python evaluate_v1.py --mode fact   --pred "$(pred predict_base_fact_sanity)" \
  --eval-source data_v1/fact_eval.jsonl --out "$R/fact_sanity_base.json" --sanity --sanity-thresh 0.05 || true
python evaluate_v1.py --mode fact   --pred "$(pred predict_fact_gate)" \
  --eval-source data_v1/fact_eval.jsonl --out "$R/fact_gate.json" || true

# --- repair (repair mode) ---
python evaluate_v1.py --mode repair --pred "$(pred predict_zeroshot_direct)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/zeroshot_direct.json" || true
python evaluate_v1.py --mode repair --pred "$(pred predict_zeroshot_cot)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/zeroshot_cot.json" || true
python evaluate_v1.py --mode repair --pred "$(pred predict_zeroshot_skillcot)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/zeroshot_skillcot.json" || true
python evaluate_v1.py --mode repair --pred "$(pred predict_factonly_repair)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/factonly_repair.json" || true
python evaluate_v1.py --mode repair --pred "$(pred predict_fact_then_cot)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/fact_then_cot.json" \
  --report "$R/fact_then_cot.md" || true
python evaluate_v1.py --mode repair --pred "$(pred predict_fact_then_skillcot)" \
  --eval-source data_v1/repair_eval.jsonl --out "$R/fact_then_skillcot.json" \
  --report "$R/fact_then_skillcot.md" || true

echo "eval reports -> $R"
