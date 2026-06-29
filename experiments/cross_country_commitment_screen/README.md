# Cross-Country Commitment Screen

This sprint tests whether answer-over-stall interventions behave like plausible commitment handles rather than Paris-specific or broad formatting handles.

Run:

```bash
/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/cross_country_commitment_screen/run.py
```

Outputs:

- `experiments/cross_country_commitment_screen/results/summary.json`
- `experiments/cross_country_commitment_screen/results/report.md`
- `experiments/cross_country_commitment_screen/results/graphs/*.pt`

The strict positive pattern is:

- France-derived interventions improve `Paris - the` on France prompts.
- Germany-derived interventions improve `Berlin - the` on Germany prompts.
- The same interventions do not strongly induce the wrong city in the other country family.
- Effects remain weak on syntax and unrelated-country controls.

This is intentionally a screen, not a final claim. A passing result would identify a candidate mechanism worth deeper representation validation.
