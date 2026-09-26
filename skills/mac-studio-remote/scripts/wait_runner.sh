#!/usr/bin/env bash
# Wait until the next job finishes under the Mac Studio scheduler
# (experiments/couplet_routes/jobs/runner.sh), then sync all experiment results back
# (excluding transcoder feature caches) and print the scheduler's new log lines.
#   wait_runner.sh            # run in the background; re-launch after each notification
source "$(dirname "$0")/_common.sh"
poll="${POLL:-120}"
count() { "${SSH[@]}" "grep -cE 'finished|queue empty' ~/$REMOTE_ROOT/logs/runner.log 2>/dev/null || echo 0"; }
seen=$(count)   # baseline at launch; every notification syncs all results, so nothing is missed
while :; do
  n=$(count)
  [ "$n" -gt "$seen" ] && break
  "${SSH[@]}" "kill -0 \$(cat ~/$REMOTE_ROOT/logs/runner.pid) 2>/dev/null" || { echo "runner not running"; break; }
  sleep "$poll"
done
cd "$REPO"
for d in experiments/couplet_routes/results experiments/derived_value_carry/results experiments/relay_positive_control/results experiments/hidden_choice/results experiments/recurrent_carry/results; do
  # a results folder may not exist remotely yet (new experiment): skip it rather than fail
  "${SSH[@]}" "test -d ~/$REMOTE_ROOT/bundle/$d" || continue
  mkdir -p "$d"
  rsync -az --exclude 'features/' --exclude '*.partial.jsonl' "$HOST:$REMOTE_ROOT/bundle/$d/" "$d/"
done
"${SSH[@]}" "tail -8 ~/$REMOTE_ROOT/logs/runner.log; echo; echo queued:; cat ~/$REMOTE_ROOT/queue.txt | cut -d' ' -f1,2"
