#!/usr/bin/env bash
# Wait until a result file exists on the Mac Studio (or its job has ended), then fetch paths:
#   wait_file.sh <job name> <repo-relative marker file> <repo-relative path to fetch>...
# Use it to be notified per step of a long multi-step job. Polls every POLL seconds (default 300).
source "$(dirname "$0")/_common.sh"
name="${1:?usage: wait_file.sh <job> <marker> <path>...}"; marker="${2:?marker}"; shift 2
poll="${POLL:-300}"
until "${SSH[@]}" "test -f ~/$REMOTE_ROOT/bundle/$marker"; do
  "$(dirname "$0")/status.sh" "$name" 0 | grep -q '^running' || { echo "job $name ended before $marker appeared"; "$(dirname "$0")/status.sh" "$name" 6; break; }
  sleep "$poll"
done
[ "$#" -gt 0 ] && "$(dirname "$0")/fetch.sh" "$@"
"$(dirname "$0")/status.sh" "$name" 2
