"""Resume support for long jobs (added 2026-09-26).

Finished rows are appended to <final>.partial.jsonl as they complete. A restarted run
loads them, skips items already done and continues; when the final JSON is written the
partial file is removed. A job stopped at any point therefore loses at most the item
(or batch) in progress.
"""
from __future__ import annotations

import json
from pathlib import Path


class Checkpoint:
    def __init__(self, final_path, key):
        self.final = Path(final_path)
        self.path = self.final.with_name(self.final.stem + ".partial.jsonl")
        self.key = key
        self.rows = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))
        self.done = {self.key(r) for r in self.rows}
        if self.rows:
            print(f"resuming {self.final.name}: {len(self.rows)} rows already done", flush=True)

    def has(self, k) -> bool:
        return k in self.done

    def add(self, row) -> None:
        self.rows.append(row)
        self.done.add(self.key(row))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def finish(self, rows=None) -> None:
        self.final.write_text(json.dumps(self.rows if rows is None else rows, indent=1))
        if self.path.exists():
            self.path.unlink()
