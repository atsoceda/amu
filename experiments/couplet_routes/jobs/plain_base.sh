#!/usr/bin/env bash
# Mac Studio job: plain-format couplet chain on base checkpoints (Qwen3 Base, Gemma 3 PT),
# gated on free memory; models over 50 GB take ~/amu_jobs/biglock. Run from ~/amu_jobs/bundle.
freegb() { vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages speculative/{s=$3}END{print int((f+i+s)*16384/1e9)}'; }
gate() { while :; do if [ $(freegb) -ge $1 ]; then if [ "$2" = 1 ]; then mkdir ~/amu_jobs/biglock 2>/dev/null && break; else break; fi; fi; sleep 60; done; echo "[gate $(date +%T) $(freegb) GB free] $3"; }
release() { rmdir ~/amu_jobs/biglock 2>/dev/null; true; }
E=experiments/couplet_routes
chain() { python -u $E/screen_rhymes.py $1 && python -u $E/step2_state_edit.py $1 && python -u $E/step34_routes.py $1 && python -u $E/relay_positions.py $1 && python -u $E/anchor_specificity.py $1 && python -u $E/step2_state_edit.py $1 --control same_rhyme && python -u $E/step34_routes.py $1 --control same_rhyme && python -u $E/relay_positions.py $1 --control same_rhyme || echo FAILED $1; }
while pgrep -f "hf download Qwen/Qwen3" >/dev/null; do sleep 60; done
for spec in Qwen3-8B-Base-plain:20:0 Qwen3-14B-Base-plain:35:0 gemma-3-12b-pt-plain:30:0 gemma-3-27b-pt-plain:60:1; do
  IFS=: read M need big <<< "$spec"; gate $need $big $M; chain $M; [ $big = 1 ] && release
done
true
