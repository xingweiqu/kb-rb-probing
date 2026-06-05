#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python convert_v1.py --data_dir data_v1
