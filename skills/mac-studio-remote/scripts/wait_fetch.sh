#!/usr/bin/env bash
# Wait for a detached Mac Studio job to exit, then fetch its results: wait_fetch.sh <name> <repo-relative path>...
# Polls every POLL seconds (default 300). Prints the log tail and exits 0 once fetched, non-zero if the job failed.
# Run it in the background locally so the agent is notified when results are back.
source "$(dirname "$0")/_common.sh"
name="${1:?usage: wait_fetch.sh <name> <path>...}"; shift
poll="${POLL:-300}"
while "$(dirname "$0")/status.sh" "$name" 0 2>/dev/null | grep -q '^running'; do sleep "$poll"; done
log=$("${SSH[@]}" "tail -5 ~/$REMOTE_ROOT/logs/$name.log")
echo "$log"
code=$(echo "$log" | sed -n 's/^\[exit \([0-9]*\) .*/\1/p' | tail -1)
[ "$#" -gt 0 ] && "$(dirname "$0")/fetch.sh" "$@"
[ "${code:-1}" = 0 ] || die "job $name exited with ${code:-unknown}"
