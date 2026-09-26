#!/usr/bin/env bash
# Mac Studio job: recurrent-carry pilot on a hybrid model (Python 3.12 env ~/amu_jobs/.venv-next,
# needed for the qwen3_5 architecture). Run from ~/amu_jobs/bundle: bash experiments/recurrent_carry/run_pilot.sh <model>
M=${1:-Qwen3.5-4B}; P=~/amu_jobs/.venv-next/bin/python; C=experiments/couplet_routes
while pgrep -f "hf download Qwen/$M" >/dev/null; do sleep 30; done
set -e
$P -u $C/screen_rhymes.py $M --n 40
$P -u $C/step2_state_edit.py $M
$P -u experiments/recurrent_carry/pilot.py $M
