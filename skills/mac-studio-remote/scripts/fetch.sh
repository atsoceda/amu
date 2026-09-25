#!/usr/bin/env bash
# Copy results back from ~/amu_jobs/bundle/<path> into the same repository-relative path here.
# fetch.sh [--allow-large] <path>...   (refuses files over 50 MB unless --allow-large)
source "$(dirname "$0")/_common.sh"
large=0; [ "${1:-}" = "--allow-large" ] && { large=1; shift; }
[ $# -gt 0 ] || die "usage: fetch.sh [--allow-large] <repo-relative path>..."
cd "$REPO"
for p in "$@"; do
  case "$p" in /*|*..*) die "path must be repository-relative without '..': $p" ;; esac
  if [ "$large" = 0 ]; then
    big=$("${SSH[@]}" "cd ~/$REMOTE_ROOT/bundle && find '$p' -type f -size +50M 2>/dev/null" || true)
    [ -z "$big" ] || die "files over 50 MB (keep them remote or pass --allow-large):"$'\n'"$big"
  fi
  mkdir -p "$(dirname "$p")"
  rsync -az "$HOST:$REMOTE_ROOT/bundle/$p" "$(dirname "$p")/"
  echo "fetched: $p"
done
