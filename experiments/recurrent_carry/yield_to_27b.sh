#!/usr/bin/env bash
# Give the Qwen3.5-27B pilot the whole GPU without losing work: once it starts computing
# (its screen begins), PAUSE the smaller pilots (SIGSTOP; they keep their memory and state)
# and RESUME them (SIGCONT) when the 27B pilot finishes. Run detached on the Mac Studio.
cd ~/amu_jobs || exit 1
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
until pgrep -f "screen_rhymes.py Qwen3.5-27B" >/dev/null; do sleep 20; done
paused=""
for n in rc-pilot-9b rc-pilot-4b; do
  [ -f running/$n.pid ] || continue
  for p in $(tree "$(cat running/$n.pid)"); do kill -STOP "$p" 2>/dev/null && paused="$paused $p"; done
  echo "[$(date +%T)] paused $n"
done
while [ -f running/rc-pilot-27b.pid ] && kill -0 "$(cat running/rc-pilot-27b.pid)" 2>/dev/null; do sleep 30; done
for p in $paused; do kill -CONT "$p" 2>/dev/null; done
echo "[$(date +%T)] resumed:$paused"
