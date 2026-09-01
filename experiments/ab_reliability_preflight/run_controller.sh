#!/bin/zsh
set -euo pipefail

cd /Users/anthony/repos/amu
export PYTHONPATH=/Users/anthony/repos/amu
exec /Users/anthony/miniconda3/bin/python experiments/ab_reliability_preflight/run_preflight.py \
  --levels 0.5 \
  --max-variants 2 \
  --inference-mode direct \
  --batch-size 4
