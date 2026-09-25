#!/usr/bin/env bash
# Launch a detached job on the Mac Studio: run.sh <name> <command...>
# Runs in ~/amu_jobs/bundle with the venv active and HF_HOME=~/amu_jobs/hf_cache; logs to ~/amu_jobs/logs/<name>.log.
source "$(dirname "$0")/_common.sh"
name="${1:?usage: run.sh <name> <command...>}"; shift
[ $# -gt 0 ] || die "no command given"
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "name may contain only letters, digits, . _ -"
cmd=$(printf '%q ' "$@")
"${SSH[@]}" "cd ~/$REMOTE_ROOT/bundle || exit 1; [ -x ../.venv/bin/python ] || { echo 'venv missing: run setup_env.sh (with user approval)'; exit 1; };
  if [ -f ../logs/$name.pid ] && kill -0 \$(cat ../logs/$name.pid) 2>/dev/null; then echo 'already running: $name (pid '\$(cat ../logs/$name.pid)')'; exit 0; fi;
  nohup env HF_HOME=\$HOME/$REMOTE_ROOT/hf_cache PATH=\$HOME/$REMOTE_ROOT/.venv/bin:\$PATH bash -c 'echo \"[start \$(date +%F\ %T)] $cmd\"; $cmd; echo \"[exit \$? \$(date +%F\ %T)]\"' > ../logs/$name.log 2>&1 < /dev/null &
  echo \$! > ../logs/$name.pid; echo \"started $name (pid \$!)\""
