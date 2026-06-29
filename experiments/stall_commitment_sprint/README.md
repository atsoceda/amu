# Stall Commitment Sprint

This sprint tests whether an ambiguous France association prompt exposes a causal competition between a direct answer token and a stall/bridge token.

Primary prompt:

`The city people most strongly associate with France is`

Primary metric:

`commitment_margin = logit(" Paris") - logit(" the")`

A useful intervention increases the commitment margin, ideally by reducing the stall token, increasing `Paris`, or both. The validation prompts test whether any effect is specific to France/Paris commitment rather than a broad perturbation of answer format.

## Regenerate

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/stall_commitment_sprint/run.py
```

Outputs:

- `results/summary.json`
- `results/report.md`
- `results/graphs/france_association.pt`

The runner reuses model loading, local Hugging Face cache handling, graph attribution, and feature extraction utilities from `experiments/semantic_commitment_sprint/run.py`.
