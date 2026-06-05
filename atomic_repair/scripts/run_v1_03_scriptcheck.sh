#!/usr/bin/env bash
# Local pre-flight: every v1 shell script parses, every v1 yaml loads.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== bash -n =="
for s in scripts/run_v1_*.sh; do bash -n "$s" && echo "ok: $s"; done
echo "== yaml load =="
python - <<'PY'
import glob, sys
try:
    import yaml
except ImportError:
    print("PyYAML not installed; skipping yaml load (install pyyaml on the server)")
    sys.exit(0)
bad = 0
for f in sorted(glob.glob("configs/v1/*.yaml")):
    try:
        yaml.safe_load(open(f)); print("ok:", f)
    except Exception as e:
        print("FAIL:", f, e); bad += 1
sys.exit(1 if bad else 0)
PY
echo "scriptcheck PASS"
