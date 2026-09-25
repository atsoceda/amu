#!/usr/bin/env python3
"""Priority-1 hardening of Workstream A: ceiling and leverage stratification.

Per prompt, leverage L = log p(planned | correct article) - log p(planned | wrong
article) with the intervention off. Policy movement dq = change in the tau=1
probability of the correct article. Equation 1 predicts the public effect scales
with dq x L. We report, per leverage tertile and per baseline-probability
tertile, prompt-bootstrap means of the tau=1 public and private effects on the
planned word's log-probability, and correlations of the public effect with dq and
with dq x L. Uses only committed six-cell rows.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
CONDS = ["multiplied_5x", "zeroed", "random_multiplied_5x", "random_zeroed"]


def boot(xs, n=10000, seed=20260925):
    xs = list(xs)
    rng = random.Random(seed)
    bs = sorted(float(np.mean(rng.choices(xs, k=len(xs)))) for _ in range(n))
    return {"mean": float(np.mean(xs)), "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n) - 1], "n": len(xs)}


def correct(r, key_a, key_an):
    return (r[key_an], r[key_a]) if r["article"] == "an" else (r[key_a], r[key_an])


def main() -> None:
    out = {}
    for m in MODELS:
        rows = [json.loads(l) for l in (EXP / "results" / m / "six_cell_rows.jsonl").read_text().splitlines()]
        lev, basep = {}, {}
        for r in rows:
            pc, pw = correct(r, "off_a_planned_p", "off_an_planned_p")
            lev[r["prompt_index"]] = math.log(max(pc, 1e-12)) - math.log(max(pw, 1e-12))
            basep[r["prompt_index"]] = pc
        res = {"leverage_tertile_cuts": np.quantile(list(lev.values()), [1 / 3, 2 / 3]).tolist(),
               "baseline_tertile_cuts": np.quantile(list(basep.values()), [1 / 3, 2 / 3]).tolist(), "conditions": {}}
        for cond in CONDS:
            rs = [r for r in rows if r["condition"] == cond]
            dq = np.array([(r["on_q_an"] - r["off_q_an"]) * (1 if r["article"] == "an" else -1) for r in rs])
            L = np.array([lev[r["prompt_index"]] for r in rs])
            B = np.array([basep[r["prompt_index"]] for r in rs])
            pub = np.array([r["tau1_planned_logp_public"] for r in rs])
            prv = np.array([r["tau1_planned_logp_private"] for r in rs])
            c = {"corr_public_dq": float(np.corrcoef(dq, pub)[0, 1]), "corr_public_dq_x_L": float(np.corrcoef(dq * L, pub)[0, 1])}
            for name, var, cuts in (("leverage", L, res["leverage_tertile_cuts"]), ("baseline", B, res["baseline_tertile_cuts"])):
                strata = {}
                for lo, hi, tag in ((-1e18, cuts[0], "low"), (cuts[0], cuts[1], "mid"), (cuts[1], 1e18, "high")):
                    k = (var >= lo) & (var < hi)
                    strata[tag] = {"dq_correct": boot(dq[k]), "public": boot(pub[k]), "private": boot(prv[k])}
                c[f"by_{name}_tertile"] = strata
            res["conditions"][cond] = c
        out[m] = res
    (EXP / "results" / "leverage_analysis.json").write_text(json.dumps(out, indent=1))
    for m, res in out.items():
        print(f"== {m}")
        for cond in ("multiplied_5x", "zeroed"):
            c = res["conditions"][cond]
            line = "  ".join(f"{t}: pub {s['public']['mean']:+.3f} priv {s['private']['mean']:+.3f}" for t, s in c["by_leverage_tertile"].items())
            print(f"  {cond}: corr(pub,dq)={c['corr_public_dq']:.3f} corr(pub,dq*L)={c['corr_public_dq_x_L']:.3f} | {line}")


if __name__ == "__main__":
    main()
