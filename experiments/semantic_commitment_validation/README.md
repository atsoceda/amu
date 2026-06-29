# Semantic Commitment Validation

This repeatable sprint validates whether the strongest causal handles from `semantic_commitment_sprint` behave like a specific semantic-commitment intervention or like a broad distribution perturbation.

Regenerate results from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/semantic_commitment_validation/run.py
```

Primary outputs:

- `experiments/semantic_commitment_validation/results/summary.json`
- `experiments/semantic_commitment_validation/results/report.md`

The validation reuses selected intervention specs from:

- `experiments/semantic_commitment_sprint/results/summary.json`

It then applies those same interventions to near-match ambiguous prompts, France prompts, semantic competitor prompts, syntax controls, and a non-city control. The core question is whether ` Paris` rises selectively where a Paris-like semantic commitment is plausible, without similarly rising on competitor or syntax-control prompts.
