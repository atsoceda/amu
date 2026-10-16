# Paired Shared Feature Suppression

This experiment tests the narrow follow-up question from
`shared_stall_suppression`: do the two individually promising shared features
work better together than either one alone?

Run:

```bash
python experiments/paired_shared_feature_suppression/run.py
```

Outputs:

- `results/summary.json`
- `results/report.md`

The key regimen is `l7_f89_plus_l10_f9037`. It suppresses both features at the
final token position of each prompt. The report compares it against each single
feature and against `l7_f89_plus_l5_f383`, a higher-direct-effect comparison
bundle.
