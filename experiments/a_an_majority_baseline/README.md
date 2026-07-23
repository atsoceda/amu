# A/An Majority Baseline

This is a circuit-free behavioral replication of the `a/an` baseline from
*Latent Planning Emerges with Scale*. It tests whether Gemma 3 270M defaults to
the majority article `a` when a prompted future noun requires `an`.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/a_an_majority_baseline/run.py
```

The experiment evaluates every target after both an `a` and an `an` in-context
demonstration. Outputs are written to:

- `results/examples.jsonl`
- `results/summary.json`
- `results/report.md`
- `results/run.log`

To regenerate the experiment, prompt an agent with:

> Regenerate the `a_an_majority_baseline` results.

This baseline measures next-token article recall only. It does not establish
that Gemma planned the listed noun or that a competing grammar circuit concealed
such a plan.
