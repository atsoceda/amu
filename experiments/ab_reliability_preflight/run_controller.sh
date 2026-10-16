#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
exec python experiments/ab_reliability_preflight/run_preflight.py \
  --levels 0.5 \
  --max-variants 2 \
  --inference-mode direct \
  --batch-size 4
