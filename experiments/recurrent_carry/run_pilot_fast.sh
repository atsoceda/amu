#!/usr/bin/env bash
# Fast recurrent-carry pilot: screen N couplets, then the pilot cells using the screen's own
# greedy line 2 (no separate step 2). Run from ~/amu_jobs/bundle: bash .../run_pilot_fast.sh <model> [N]
M=${1:-Qwen3.5-27B}; N=${2:-24}; P=~/amu_jobs/.venv-next/bin/python; C=experiments/couplet_routes
while pgrep -f "hf download Qwen/$M" >/dev/null; do sleep 20; done
set -e
$P -u $C/screen_rhymes.py $M --n $N
$P -u experiments/recurrent_carry/pilot.py $M --from-screen
