#!/usr/bin/env bash
# Read-only inventory of everything under the remote job folder, for accounting:
#   inventory.sh [out.tsv]
# Records running and queued jobs, every result file (size, mtime, SHA-256 when under
# 50 MB), every log with its last line, cached Hugging Face revisions, both Python
# environments and disk use. Changes nothing on either machine except the local output
# file. Compare with the repository using compare_inventory.py.
source "$(dirname "$0")/_common.sh"
out="${1:-/dev/stdout}"
"${SSH[@]}" "bash -s" > "$out" <<EOF
set -u
cd ~/$REMOTE_ROOT 2>/dev/null || { echo "no ~/$REMOTE_ROOT"; exit 0; }
echo "#section	host"
echo "date	\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -h ~ | tail -1 | awk '{print "disk_free\t" \$4 "\t" \$5 " used"}'
du -sh . 2>/dev/null | awk '{print "jobs_folder\t" \$1}'
echo "#section	running"
for f in running/*.pid logs/*.pid; do
  [ -f "\$f" ] || continue
  pid=\$(cat "\$f"); state=dead; kill -0 "\$pid" 2>/dev/null && state=alive
  echo "\$f	\$pid	\$state"
done
ps -axo pid,etime,command | grep -E "amu_jobs|experiments/" | grep -v grep | sed 's/\t/ /g' | awk '{print "proc\t" \$0}'
echo "#section	queues"
for q in queue.txt queue.later.txt; do [ -f "\$q" ] && sed "s/^/\$q	/" "\$q"; done
echo "#section	results"
find bundle -type f \( -path "*/results/*" -o -name "*_rows*.json*" -o -name "*_summary*.json" \) 2>/dev/null | sort | while read -r f; do
  size=\$(stat -f %z "\$f"); mtime=\$(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "\$f")
  if [ "\$size" -lt 52428800 ]; then sha=\$(shasum -a 256 "\$f" | cut -d' ' -f1); else sha=large; fi
  echo "\${f#bundle/}	\$size	\$mtime	\$sha"
done
echo "#section	logs"
for f in logs/*.log; do
  [ -f "\$f" ] || continue
  echo "\$f	\$(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "\$f")	\$(tail -1 "\$f" | tr '\t' ' ' | cut -c1-160)"
done
echo "#section	hf_revisions"
for d in hf_cache/hub/models--*; do
  [ -d "\$d" ] || continue
  repo=\$(basename "\$d" | sed 's/^models--//; s/--/\//')
  echo "\$repo	\$(cat "\$d/refs/main" 2>/dev/null || echo none)	\$(ls "\$d/snapshots" 2>/dev/null | tr '\n' ' ')	\$(du -sh "\$d" | cut -f1)"
done
echo "#section	envs"
for v in .venv .venv-next; do
  [ -x "\$v/bin/python" ] || continue
  echo "\$v	python	\$(\$v/bin/python --version 2>&1)"
  \$v/bin/python -m pip freeze 2>/dev/null | sed "s/^/\$v	pkg	/"
done
EOF
