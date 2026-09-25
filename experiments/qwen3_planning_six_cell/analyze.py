#!/usr/bin/env python3
"""Summarize six-cell rows: prompt-bootstrap means per model and condition."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
TASK = os.environ.get("AMU_TASK", "a_an")
RES = EXP / "results" if TASK == "a_an" else EXP / "results" / TASK
KEYS = ["tv_total", "tv_public", "tv_private", "tv_private_baseline_article",
        "tau1_tv_total", "tau1_tv_public", "tau1_tv_private",
        "tau1_planned_logp_total", "tau1_planned_logp_public", "tau1_planned_logp_private"]


def boot(xs, n=10000, seed=20260924):
    if not xs:
        return None
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return {"mean": m, "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n) - 1], "n": len(xs)}


def summarize(model: str) -> dict:
    rows = [json.loads(l) for l in (RES / model / "six_cell_rows.jsonl").read_text().splitlines()]
    if os.environ.get("AMU_SUBSET") == "first150":
        sys.path.insert(0, str(EXP))
        from select_features import prompts
        df = prompts(model)
        keep = set(df.index[df["source_index"] < 150])
        rows = [r for r in rows if r["prompt_index"] in keep]
    out = {"model": model, "conditions": {}}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        valid = [r for r in rs if r["off_greedy_is_article"] and r["on_greedy_is_article"]]
        c = {"n_prompts": len(rs), "n_greedy_article_support": len(valid),
             "article_switch_rate": sum(r["off_greedy_article"] != r["on_greedy_article"] for r in rs) / len(rs),
             "mean_abs_q_an_change": sum(abs(r["on_q_an"] - r["off_q_an"]) for r in rs) / len(rs)}
        for k in KEYS:
            src = valid if k.startswith("tv_") else rs
            c[k] = boot([r[k] for r in src])
        out["conditions"][cond] = c
    tag = "_first150" if os.environ.get("AMU_SUBSET") == "first150" else ""
    (RES / model / f"summary{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def fmt(b):
    return "--" if b is None else f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


if __name__ == "__main__":
    lines = ["# Six-cell summary", ""]
    for model in sys.argv[1:]:
        s = summarize(model)
        lines += [f"## {model}", "", "| Condition | n | Switch | TV total | TV public | TV private | TV private (baseline art.) | τ=1 Δlog p(planned) public | private |",
                  "|---|---:|---:|---|---|---|---|---|---|"]
        for cond, c in s["conditions"].items():
            lines.append(f"| {cond} | {c['n_prompts']} | {c['article_switch_rate']:.2f} | {fmt(c['tv_total'])} | {fmt(c['tv_public'])} | "
                         f"{fmt(c['tv_private'])} | {fmt(c['tv_private_baseline_article'])} | {fmt(c['tau1_planned_logp_public'])} | {fmt(c['tau1_planned_logp_private'])} |")
        lines.append("")
    tag = "_first150" if os.environ.get("AMU_SUBSET") == "first150" else ""
    (RES / f"tables{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
