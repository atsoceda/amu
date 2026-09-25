#!/usr/bin/env bash
# Send an allowlisted bundle (repository-relative paths, one per line) to ~/amu_jobs/bundle.
# Refuses anything that looks like the codebase or private material.
source "$(dirname "$0")/_common.sh"
list="${1:?usage: push_bundle.sh <allowlist.txt>}"
[ -f "$list" ] || die "allowlist not found: $list"
cd "$REPO"
files=()
while IFS= read -r p; do
  p="${p%%#*}"; p="$(echo "$p" | xargs)"; [ -z "$p" ] && continue
  case "$p" in
    /*|*..*) die "allowlist paths must be repository-relative without '..': $p" ;;
    .git*|paper.qmd|manuscript/*|docs/*|AGENTS.md|CLAUDE.md|submissions/*|dist/*|drive-sync-amu/*|*.env|*token*|*secret*|*.ovpn|*id_ed25519*)
      die "forbidden path for the Mac Studio: $p" ;;
  esac
  [ -e "$p" ] || die "missing: $p"
  # A directory entry never sends results: pushing local copies of results/ overwrites newer
  # remote results (this destroyed a rerun on 2026-09-26). List a result file by name to send it.
  while IFS= read -r f; do files+=("$f"); done < <(if [ -d "$p" ]; then find "$p" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path '*/results/*'; else echo "$p"; fi)
done < "$list"
[ ${#files[@]} -gt 0 ] || die "allowlist is empty"
bytes=$(du -ck "${files[@]}" | tail -1 | awk '{print $1*1024}')
printf '%s\n' "${files[@]}" | sed 's/^/send: /'
echo "total: ${#files[@]} files, $((bytes/1024)) KB"
[ "$bytes" -lt $((100*1024*1024)) ] || [ "${ALLOW_LARGE:-0}" = 1 ] || die "bundle over 100 MB; the link is slow. Let the Mac Studio download large data itself, or set ALLOW_LARGE=1."
"${SSH[@]}" "mkdir -p ~/$REMOTE_ROOT/bundle ~/$REMOTE_ROOT/logs"
printf '%s\n' "${files[@]}" | rsync -az --files-from=- ./ "$HOST:$REMOTE_ROOT/bundle/"
echo "sent to $HOST:~/$REMOTE_ROOT/bundle"
