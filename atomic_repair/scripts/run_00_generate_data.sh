#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python generate_repair_data.py --out_dir data --seed 42
