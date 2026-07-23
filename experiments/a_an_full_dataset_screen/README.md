# Full A/An Dataset Screen

This circuit-free experiment extends the paper's behavioral evaluation by
generating continuations for every `an`-target prompt in its released combined
`a/an` dataset.

The source dataset is downloaded once from a pinned commit of
`hannamw/model-planning-public` and verified by SHA-256.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/a_an_full_dataset_screen/run.py
```

Outputs:

- `results/report.md`
- `results/summary.json`
- `results/examples.jsonl`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `a_an_full_dataset_screen` results.
