# planning dose response

E1–E6 protocol (complete; historical): [`../PAPER_EXPERIMENTS.md`](../PAPER_EXPERIMENTS.md). Current paper claim lives in `paper.qmd`.

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python experiments/planning_dose_response/run.py
```

Requires prior experiment outputs where noted in `config.json` (E1 selection/summary, etc.).

Stack: `google/gemma-3-270m` + PT `width_262k_l0_medium_affine`.
