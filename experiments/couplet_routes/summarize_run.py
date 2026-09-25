#!/usr/bin/env python3
"""One-line-per-quantity summary of a couplet-route run directory (for logs and notes).

Usage: python summarize_run.py <model> [<model> ...]
"""
import json
import statistics as st
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def ci(v):
    return f"{v['mean']:+.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"


for m in sys.argv[1:]:
    d = R / m
    out = [m]
    if (d / "rhyme_screen.json").exists():
        sc = json.loads((d / "rhyme_screen.json").read_text())
        out.append(f"screen {sum(x['rhyme_success'] for x in sc)}/{len(sc)}")
    if (d / "step2_summary.json").exists():
        s2 = json.loads((d / "step2_summary.json").read_text())
        out.append(f"donor rhyme {s2['on_rhymes_donor']:.0%}")
    if (d / "step34_summary.json").exists():
        s = json.loads((d / "step34_summary.json").read_text())
        a = s["all"]
        rows = json.loads((d / "step34_rows.json").read_text())
        xs = [x["d_relay"] / x["d_persistence_p0"] for x in rows if abs(x["d_persistence_p0"]) > 1]
        out.append(f"total {ci(a['d_total'])} emission {ci(a['d_emission'])} persistence {ci(a['d_persistence_p0'])} "
                   f"retrieval {ci(a['d_retrieval'])} relay {ci(a['d_relay'])} gap {ci(s['additivity_gap_p0'])} "
                   f"median relay share {st.median(xs):.0%}")
    for tag in ("", "_same_rhyme"):
        f = d / f"relay_positions_summary{tag}.json"
        if f.exists():
            p = json.loads(f.read_text())["all"]
            out.append(("null " if tag else "") + "positions: " + " ".join(
                f"{k.replace('d_relay_', '')} {ci(v)}" for k, v in p.items() if k.startswith("d_relay")))
    if (d / "anchor_specificity_summary.json").exists():
        a = json.loads((d / "anchor_specificity_summary.json").read_text())["by_position"]
        out.append("anchor " + " ".join(f"{k} {v['persistence']['mean']:+.1f}" for k, v in a.items()))
    print("\n  ".join(out))
