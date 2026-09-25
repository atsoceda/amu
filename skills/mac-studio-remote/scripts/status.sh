#!/usr/bin/env bash
# Show whether a job is running and the last lines of its log: status.sh <name> [lines]
source "$(dirname "$0")/_common.sh"
name="${1:?usage: status.sh <name> [lines]}"; n="${2:-5}"
"${SSH[@]}" "cd ~/$REMOTE_ROOT/logs 2>/dev/null || { echo 'no logs yet'; exit 0; };
  if [ -f $name.pid ] && kill -0 \$(cat $name.pid) 2>/dev/null; then echo 'running (pid '\$(cat $name.pid)')'; else echo 'not running'; fi;
  [ -f $name.log ] && grep -v -i 'warn' $name.log | tail -$n || echo 'no log: $name'"
