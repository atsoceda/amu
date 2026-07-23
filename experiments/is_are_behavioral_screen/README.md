# Is/Are Behavioral Screen

This circuit-free experiment screens the full subtraction dataset released with
*Latent Planning Emerges with Scale* for Gemma continuations of the form
`are 1`.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/is_are_behavioral_screen/run.py
```

Outputs:

- `results/report.md`
- `results/summary.json`
- `results/examples.jsonl`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `is_are_behavioral_screen` results.
