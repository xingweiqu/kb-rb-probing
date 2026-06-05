#!/usr/bin/env bash
set -euo pipefail
BRANCH="${1:-atomic-repair-v1}"
cd "$(git rev-parse --show-toplevel)"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add atomic_repair/
git status
git commit -m "Add atomic-repair v1 (known-component repair: facts seen, forms unseen)" \
  || echo "No changes to commit"
git push -u origin "$BRANCH"
