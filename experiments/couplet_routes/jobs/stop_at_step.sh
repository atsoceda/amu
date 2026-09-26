#!/usr/bin/env bash
# Stop a running chain.sh job cleanly at a step boundary and requeue it to resume there:
#   stop_at_step.sh <job name> <model> <step to resume from>
# Waits until the chain logs the start of <step> (the previous step has saved its output),
# stops the job, and puts "<job> <mem> chain.sh <model> <step>" at the end of the queue.
cd ~/amu_jobs || exit 1
J=$1; M=$2; S=$3
until grep -q "] $M: $S" logs/$J.log 2>/dev/null; do sleep 10; done
p=$(cat running/$J.pid); mem=$(cat running/$J.mem)
for c in $(pgrep -P "$p"); do pkill -P "$c" 2>/dev/null; kill "$c" 2>/dev/null; done; kill "$p" 2>/dev/null
echo "$J-resume $mem bash experiments/couplet_routes/jobs/chain.sh $M $S" >> queue.txt
echo "[$(date +%T)] stopped $J at $S; requeued"
