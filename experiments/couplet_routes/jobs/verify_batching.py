#!/usr/bin/env python3
"""Check batched scripts against their one-at-a-time paths on a model no experiment uses
(Qwen3-0.6B), so no real result file is touched. Writes
experiments/couplet_routes/results/batching_check.json. Run from ~/amu_jobs/bundle."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

M = "Qwen3-0.6B"
B = Path.cwd()
checks = [
    ("chain", "experiments/derived_value_carry/chain_routes.py", ["--limit", "3", "--ks", "3"],
     "experiments/derived_value_carry/results/Qwen3-0.6B/chain_K3_rows.json"),
    ("free", "experiments/hidden_choice/choice_free.py", ["--pairs", "10"],
     "experiments/hidden_choice/results/Qwen3-0.6B/choice_free_rows.json"),
]


def run(script, args, batch):
    subprocess.run([sys.executable, "-u", script, M, *args, "--batch", str(batch)], check=True)


def flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{prefix}{k}.")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}{i}.")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix.rstrip("."), float(x)


out = {}
for name, script, args, rows_path in checks:
    run(script, args, 1)
    a = json.loads((B / rows_path).read_text())
    run(script, args, 8)
    b = json.loads((B / rows_path).read_text())
    if name == "free":  # match rows by pair: batched screening can pick a different pair at near-ties
        key = lambda r: (tuple(r["list"]), tuple(r["donor_list"]))  # noqa: E731
        bb = {key(r): r for r in b}
        matched = [(r, bb[key(r)]) for r in a if key(r) in bb]
        same = [(x, y) for x, y in matched if x["text0"] == y["text0"] and x["text1"] == y["text1"]]
        diffs = sorted(abs(x[k] - y[k]) for x, y in same for k in ("total", "emission", "persistence_t1", "persistence_t0"))
        out[name] = {"n_rows": [len(a), len(b)], "matched_pairs": len(matched), "matched_same_text": len(same),
                     "max_abs_diff_same_text": diffs[-1] if diffs else None,
                     "median_abs_diff_same_text": diffs[len(diffs) // 2] if diffs else None}
        print(name, out[name], flush=True)
        shutil.rmtree(B / Path(rows_path).parent)
        continue
    fa, fb = dict(flat(a)), dict(flat(b))
    common = [k for k in fa if k in fb]
    diffs = sorted((abs(fa[k] - fb[k]), k) for k in common)
    same_text = [all(x.get(t) == y.get(t) for t in ("text0", "text1", "gen_off", "gen_on") if t in x) for x, y in zip(a, b)]
    out[name] = {"n_rows": [len(a), len(b)], "n_numbers": len(common), "max_abs_diff": diffs[-1] if diffs else None,
                 "median_abs_diff": diffs[len(diffs) // 2][0] if diffs else None, "rows_with_same_text": sum(same_text)}
    print(name, out[name], flush=True)
    shutil.rmtree(B / Path(rows_path).parent)
(B / "experiments/couplet_routes/results/batching_check.json").write_text(json.dumps(out, indent=1))
