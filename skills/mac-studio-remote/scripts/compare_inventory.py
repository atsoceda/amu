#!/usr/bin/env python3
"""Compare a remote inventory (inventory.sh output) with this repository.

Usage: compare_inventory.py <inventory.tsv>

Every remote result file is classed as:
  same          committed here with identical content
  differs       present here with different content (newer remote rerun, or an older copy)
  remote-only   not in the repository: fetch it if the experiment README reports it
  large         50 MB or more (intermediates such as captured activations; normally stay remote)
and every running process, queued job and log is listed, so nothing is left unaccounted.
Standard library only; reads files, changes nothing.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sections(path):
    out, cur = defaultdict(list), None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#section\t"):
            cur = line.split("\t", 1)[1]
        elif cur and line.strip():
            out[cur].append(line.split("\t"))
    return out


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tracked(rel):
    return subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
                          capture_output=True).returncode == 0


def main(inv):
    s = sections(inv)
    for k, v in s.get("host", []):
        print(f"{k}: {v}")
    groups = defaultdict(list)
    for rel, size, mtime, digest in s.get("results", []):
        local = REPO / rel
        if digest == "large":
            groups["large"].append((rel, size, mtime, ""))
        elif not local.exists():
            groups["remote-only"].append((rel, size, mtime, ""))
        elif sha(local) == digest:
            groups["same"].append((rel, size, mtime, "tracked" if tracked(rel) else "UNTRACKED here"))
        else:
            local_mtime = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cs", "--", rel],
                                         capture_output=True, text=True).stdout.strip()
            groups["differs"].append((rel, size, mtime, f"committed {local_mtime or 'never'}"))
    for g in ("differs", "remote-only", "large", "same"):
        items = groups[g]
        print(f"\n== {g}: {len(items)}")
        by_dir = defaultdict(list)
        for it in items:
            by_dir[str(Path(it[0]).parent)].append(it)
        for d in sorted(by_dir):
            files = by_dir[d]
            if g == "same":
                untracked = [f for f in files if f[3] != "tracked"]
                print(f"  {d}: {len(files)} files" + (f" ({len(untracked)} untracked here)" if untracked else ""))
            else:
                for rel, size, mtime, note in files:
                    print(f"  {rel}  {int(size):,} B  remote {mtime}  {note}")
    print("\n== running / pid files")
    for row in s.get("running", []):
        print("  " + "  ".join(row))
    print("\n== queues")
    for row in s.get("queues", []) or [["(empty)"]]:
        print("  " + "  ".join(row))
    print(f"\n== logs: {len(s.get('logs', []))} (last line of each in the inventory file)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
