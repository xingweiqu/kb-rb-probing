#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python generate_v1.py --out_dir data_v1 --seed 42
