#!/usr/bin/env bash
# Mac Studio job scheduler with a declared memory budget (replaces free-memory gates,
# which let jobs overlap and thrash: models load gradually, so free memory at start
# time says little). Run once, detached, from ~/amu_jobs/bundle:
#   nohup bash experiments/couplet_routes/jobs/runner.sh > ~/amu_jobs/logs/runner.log 2>&1 &
# Queue: ~/amu_jobs/queue.txt, one job per line: <name> <mem_gb> <shell command...>
#   (appending lines while the runner runs is fine; it re-reads the file each cycle).
# Running jobs: ~/amu_jobs/running/<name>.pid and <name>.mem (external processes can
# be registered the same way so their memory is counted). Logs: ~/amu_jobs/logs/<name>.log.
# Starts the first queued job that fits in BUDGET_GB minus the memory of running jobs
# (later jobs may start first if they fit: backfill). Exits when queue and running are empty.
set -u
ROOT=$HOME/amu_jobs
BUDGET_GB=${BUDGET_GB:-108}
Q=$ROOT/queue.txt
RUN=$ROOT/running
mkdir -p "$RUN" "$ROOT/logs"
touch "$Q"
cd "$ROOT/bundle" || exit 1
export HF_HOME=$ROOT/hf_cache PATH=$ROOT/.venv/bin:$PATH
while :; do
  used=0
  for f in "$RUN"/*.pid; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .pid)
    if kill -0 "$(cat "$f")" 2>/dev/null; then
      used=$((used + $(cat "$RUN/$n.mem")))
    else
      echo "[$(date +%T)] finished $n"
      rm -f "$f" "$RUN/$n.mem"
    fi
  done
  started=0
  if [ -s "$Q" ]; then
    lineno=0
    while IFS= read -r line; do
      lineno=$((lineno + 1))
      [ -z "$line" ] && continue
      name=${line%% *}; rest=${line#* }; mem=${rest%% *}; cmd=${rest#* }
      if [ $((used + mem)) -le "$BUDGET_GB" ]; then
        nohup bash -c "echo \"[start \$(date '+%F %T')] $cmd\"; $cmd; echo \"[exit \$? \$(date '+%F %T')]\"" \
          > "$ROOT/logs/$name.log" 2>&1 < /dev/null &
        echo $! > "$RUN/$name.pid"; echo "$mem" > "$RUN/$name.mem"
        echo "[$(date +%T)] started $name ($mem GB; $((used + mem))/$BUDGET_GB in use)"
        sed -i '' "${lineno}d" "$Q"
        started=1
        break
      fi
    done < "$Q"
  fi
  [ "$started" = 1 ] && continue
  if [ ! -s "$Q" ] && ! ls "$RUN"/*.pid >/dev/null 2>&1; then echo "[$(date +%T)] queue empty; exiting"; exit 0; fi
  sleep 30
done
