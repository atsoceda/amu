#!/usr/bin/env bash
# Couplet chain for one model, from a given step: chain.sh <model> [first step]
# Steps: screen step2 step34 positions anchor step2null step34null positionsnull
# (screen only for models without an authors' subset: Gemma and -plain runs).
# Run from ~/amu_jobs/bundle (bash 3.2 compatible).
M=$1; FROM=${2:-screen}; E=experiments/couplet_routes
cmd() {
  case $1 in
    screen) echo "$E/screen_rhymes.py $M" ;;
    step2) echo "$E/step2_state_edit.py $M" ;;
    step34) echo "$E/step34_routes.py $M" ;;
    positions) echo "$E/relay_positions.py $M" ;;
    anchor) echo "$E/anchor_specificity.py $M" ;;
    step2null) echo "$E/step2_state_edit.py $M --control same_rhyme" ;;
    step34null) echo "$E/step34_routes.py $M --control same_rhyme" ;;
    positionsnull) echo "$E/relay_positions.py $M --control same_rhyme" ;;
  esac
}
on=0
for s in screen step2 step34 positions anchor step2null step34null positionsnull; do
  [ "$s" = "$FROM" ] && on=1
  [ "$on" = 1 ] || continue
  echo "[$(date +%T)] $M: $s"
  python -u $(cmd $s) || { echo "FAILED $M at $s"; exit 1; }
done
