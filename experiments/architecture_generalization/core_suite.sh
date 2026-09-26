#!/usr/bin/env bash
# Core suite for one model (Python 3.12 env). Resumable: a step is skipped when its output exists.
# Run from ~/amu_jobs/bundle: bash experiments/architecture_generalization/core_suite.sh <model>
M=$1; P=~/amu_jobs/.venv-next/bin/python; C=experiments/couplet_routes; R=$C/results/$M
H=experiments/hidden_choice; HR=$H/results/$M; I=experiments/relay_positive_control
step() { out=$1; shift; if [ -e "$out" ]; then echo "[skip] $out exists"; else echo "[$(date +%T)] $M: $*"; "$@" || { echo "FAILED $M: $*"; exit 1; }; fi; }
step $R/subset.csv $P -u $C/screen_rhymes.py $M --n 100
step $R/step2_rows.json $P -u $C/step2_state_edit.py $M
step $R/step34_summary.json $P -u $C/step34_routes.py $M
step $R/relay_positions_summary.json $P -u $C/relay_positions.py $M
step $R/relay_necessity_summary.json $P -u $C/relay_necessity.py $M
step $I/results/$M/induction_summary.json $P -u $I/induction.py $M
step $HR/choice_outlist_summary.json $P -u $H/choice_outlist.py $M
step $HR/choice_replicate_animals_summary.json $P -u $H/choice_replicate.py $M --domain animals
step $HR/choice_summary.json $P -u $H/choice_routes.py $M
echo "[$(date +%T)] $M: core suite done"
