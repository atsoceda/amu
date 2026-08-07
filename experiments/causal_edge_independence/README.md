# Causal edge independence (N0–N5)

Corrected protocol for testing modular H1 edges vs packaged H2 trajectories.

## Critical fix vs Stage XV

Stage XV pasted the native article then generated the noun with interventions **off**.
That cannot falsify independent \(C \to c\). This experiment keeps content clamps
**on** at the original pre-article position during noun prediction while \(b\) is fixed.

## Stages

- **N0**: schedule smoke (content-off vs content-on under fixed \(b\))
- **N1**: within-class fixed-article \(C \to c\) (primary)
- **N2**: factorial \(C \to B\) at article step (interpreted only if N1 finds a dial)
- **N3**: selective \(B \to b\) (S2)
- **N4**: latent plan vs executed-token readouts
- **N5**: skipped unless N1 validates a content dial (no pure \(A\) set yet)

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python -m experiments.causal_edge_independence.run
```

Outputs: `results/summary.json`, `results/rows.json`, `results/report.md`, `results/run.log`.
