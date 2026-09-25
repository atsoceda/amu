#!/usr/bin/env bash
# Wait until the next job finishes under the Mac Studio scheduler
# (experiments/couplet_routes/jobs/runner.sh), then sync all experiment results back
# (excluding transcoder feature caches) and print the scheduler's new log lines.
#   wait_runner.sh            # run in the background; re-launch after each notification
source "$(dirname "$0")/_common.sh"
poll="${POLL:-120}"
state="$REPO/.git/amu_runner_seen"   # local, untracked: number of runner log lines already seen
seen=$(cat "$state" 2>/dev/null || echo 0)
while :; do
  n=$("${SSH[@]}" "grep -cE 'finished|queue empty' ~/$REMOTE_ROOT/logs/runner.log 2>/dev/null || echo 0")
  [ "$n" -gt "$seen" ] && break
  "${SSH[@]}" "kill -0 \$(cat ~/$REMOTE_ROOT/logs/runner.pid) 2>/dev/null" || { echo "runner not running"; break; }
  sleep "$poll"
done
cd "$REPO"
for d in experiments/couplet_routes/results experiments/derived_value_carry/results; do
  rsync -az --exclude 'features/' "$HOST:$REMOTE_ROOT/bundle/$d/" "$d/"
done
echo "$n" > "$state"
"${SSH[@]}" "tail -8 ~/$REMOTE_ROOT/logs/runner.log; echo; echo queued:; cat ~/$REMOTE_ROOT/queue.txt | cut -d' ' -f1,2"
