#!/usr/bin/env bash
# Trimmed core suite (2026-09-26): the parts the surprise-led framing rests on. Resumable by step.
#   bash experiments/architecture_generalization/core_suite_lite.sh <model> [--recurrent]
# hidden choice (out-of-list specificity, animal replication, forced route split), induction
# control, and with --recurrent the fast recurrent-carry pilot (24 couplets).
M=$1; P=~/amu_jobs/.venv-next/bin/python; C=experiments/couplet_routes
H=experiments/hidden_choice; HR=$H/results/$M; I=experiments/relay_positive_control
while pgrep -f "hf download .*/$M" >/dev/null; do sleep 20; done
step() { out=$1; shift; if [ -e "$out" ]; then echo "[skip] $out exists"; else echo "[$(date +%T)] $M: $*"; "$@" || { echo "FAILED $M: $*"; exit 1; }; fi; }
if [ "$2" = "--recurrent" ]; then
  step $C/results/$M/subset.csv $P -u $C/screen_rhymes.py $M --n 24
  step experiments/recurrent_carry/results/$M/pilot_summary.json $P -u experiments/recurrent_carry/pilot.py $M --from-screen
fi
step $HR/choice_outlist_summary.json $P -u $H/choice_outlist.py $M
step $HR/choice_replicate_animals_summary.json $P -u $H/choice_replicate.py $M --domain animals
step $HR/choice_summary.json $P -u $H/choice_routes.py $M
step $I/results/$M/induction_summary.json $P -u $I/induction.py $M
echo "[$(date +%T)] $M: trimmed core suite done"
