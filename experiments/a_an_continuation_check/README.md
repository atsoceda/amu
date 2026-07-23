# A/An Continuation Check

This circuit-free follow-up tests what Gemma 3 270M generates after the strong
`a` preference found by `a_an_majority_baseline`.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/a_an_continuation_check/run.py
```

It evaluates all vowel-initial targets from the baseline under both
demonstrations. For each target it records:

- unconstrained greedy continuation;
- continuation after forced `a`;
- continuation after forced `an`;
- log probability of the listed word after each forced article.

Outputs:

- `results/examples.jsonl`
- `results/summary.json`
- `results/report.md`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `a_an_continuation_check` results.
