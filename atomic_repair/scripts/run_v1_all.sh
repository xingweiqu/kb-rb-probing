#!/usr/bin/env bash
# v1 one-button orchestration.
#   Local steps (data): generate -> validate -> convert -> scriptcheck.
#   Server steps (training/predict): printed as a checklist (this script does NOT ssh).
#   Local scoring (after pulling predictions): evaluate -> compare.
#
# Usage:
#   bash scripts/run_v1_all.sh            # local data steps + print server checklist
#   bash scripts/run_v1_all.sh --dry-run  # same (local-only is the only local capability)
#   bash scripts/run_v1_all.sh --score    # run local scoring (needs pulled predictions)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"

if [ "$MODE" = "--score" ]; then
  bash scripts/run_v1_30_evaluate.sh
  bash scripts/run_v1_31_compare.sh
  echo "Done. See data_v1/comparison_v1.md"
  exit 0
fi

echo "############ LOCAL: data pipeline ############"
bash scripts/run_v1_00_generate.sh
bash scripts/run_v1_01_validate.sh
bash scripts/run_v1_02_convert.sh
bash scripts/run_v1_03_scriptcheck.sh

cat <<'EOF'

############ SERVER: run these on the GPU box ############
# Prereqs: clone LLaMA-Factory, pip install -e ".[torch,metrics,deepspeed]",
#          set LF_DIR, edit model_name_or_path in configs/v1/*.yaml to your
#          Qwen3-8B-Instruct path, confirm template name (qwen).

# 0) cleanliness gate (BEFORE training): base Instruct must score ~0 on fact-QA
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_09_sanity_base.sh

# 1) knowledge stage
LF_DIR=~/LLaMA-Factory NPROC=8 bash scripts/run_v1_10_train_fact.sh

# 2) relay both trajectory branches from fact_only (after step 1)
LF_DIR=~/LLaMA-Factory NPROC=8 bash scripts/run_v1_11_train_fact_then_cot.sh
LF_DIR=~/LLaMA-Factory NPROC=8 bash scripts/run_v1_12_train_fact_then_skillcot.sh

# 3) predictions
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_20_predict_zeroshot_direct.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_20_predict_zeroshot_cot.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_20_predict_zeroshot_skillcot.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_21_predict_fact_gate.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_22_predict_factonly_repair.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_23_predict_fact_then_cot.sh
LF_DIR=~/LLaMA-Factory bash scripts/run_v1_24_predict_fact_then_skillcot.sh

# 4) pull ./output_v1/predict_*/generated_predictions.jsonl back to this repo, then:
bash scripts/run_v1_all.sh --score
EOF
