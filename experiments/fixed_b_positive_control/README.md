# Fixed-b positive control + near-boundary screen

Validates whether the fixed-native-article protocol can transmit noun-token
effects via oracle MLP-in activation patching, and screens same-article-class
pairs for near-boundary baseline logit gaps.

## Run

```bash
/Users/anthony/miniconda3/bin/python experiments/fixed_b_positive_control/run.py
```

## Outputs

- `results/screens.json` — baseline gaps for all candidate pairs
- `results/oracle_rows.json` — activation-patch rows
- `results/random_rows.json` — matched-norm random controls
- `results/summary.json` / `results/report.md`
