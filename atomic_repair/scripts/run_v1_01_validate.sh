#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python validate_v1.py --data_dir data_v1 --out data_v1/data_sanity_report_v1.json
