# Shared Stall Suppression

This experiment tests whether the overlapping `the`-supporting features from
`cross_country_commitment_screen` can be used as a fixed, reusable suppression
regimen.

Run:

```bash
/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/shared_stall_suppression/run.py
```

Outputs:

- `results/summary.json`
- `results/report.md`

The main comparison is between two intervention regimens:

- `shared_overlap_fixed_pos`: suppress the shared features at their discovered
  absolute token position.
- `shared_overlap_final_token`: suppress the same layer/feature coordinates at
  each prompt's final token position.
- `shared_overlap_top*_...` and `shared_single_final_token_*`: diagnostic
  subset screens that test whether the full shared bundle is over-suppressing
  useful semantic features.

The final-token regimen is the closer approximation to a handoff tool because it
does not require every prompt to have the same token length as the discovery
prompts.
