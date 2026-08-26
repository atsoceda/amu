# Six-cell family sweep

Run the generated-token six-cell design on S1–S4, the S1 activation-matched
random control, and S1 at \(1.5\times\) and \(3\times\) gain.

The six cells are intervention off/on crossed with free generation, inserted
`a`, and inserted `an`. Intervention-off cells are shared across handles.
Attention is recomputed. Vector decomposition stores the additive probability
contrasts \(E_T = E_M + E_R\) before total-variation distances.

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/six_cell_family_sweep/run.py
```

Optional smoke test:

```bash
/Users/anthony/miniconda3/bin/python experiments/six_cell_family_sweep/run.py --max-prompts 1
```

Outputs:

- `results/rows.json`: compact per-prompt, per-handle cells and contrasts
- `results/summary.json`: bootstrap summaries by handle
- `results/report.md`
- `manuscript/figures/fig_six_cell_family.png` after `plot.py`
