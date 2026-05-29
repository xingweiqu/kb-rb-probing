#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-atomic-repair-data-v0}"

# Move to repo root (one level above this scripts dir's parent).
cd "$(git rev-parse --show-toplevel)"

git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add atomic_repair/
git status
git commit -m "Add LLM-Physics-style atomic repair data v0" || echo "No changes to commit"
git push -u origin "$BRANCH"
