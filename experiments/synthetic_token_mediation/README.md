# Synthetic token-mediation validation

Ground-truth check that the generated-token mediation estimands discriminate:

1. **Mediated mechanism** — intervention changes only the article; nouns depend on the article (generated-token relay).
2. **Direct mechanism** — intervention also injects a persistent plan bias that survives article / filler clamping.

## Run

```bash
/Users/anthony/miniconda3/bin/python -m experiments.synthetic_token_mediation.run
/Users/anthony/miniconda3/bin/python -m experiments.synthetic_token_mediation.plot
```

## Outputs

- `results/summary.json` — TE / article-only / residual / CDE interaction / k-curves
- `results/rows.json` — per-cue rows
- `results/report.md` — human-readable acceptance summary
- `manuscript/figures/fig_synthetic_mediation_validation.png`

## Acceptance

Configured thresholds in `config.json` require near-zero residual for the mediated model and large residual for the direct model, including the k-token residual-control curve.
