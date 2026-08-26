# Six-cell family sweep

Generated: 2026-08-26T12:25:17.135172+00:00
Model: `google/gemma-3-270m`
Runtime: 214.5s

Each handle uses intervention off/on crossed with free generation,
inserted `a`, and inserted `an`. Intervention-off cells are shared.

| Handle | Free `an` | Noun changed | Treated-token top-1 match | Total TV | Token-substitution TV | Matched-prefix residual TV | Residual TV under `a` | Residual TV under `an` | Twin ΔΔ `a` | Twin ΔΔ `an` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1_5x` | 1.00 | 0.95 | 1.00 | 0.893 [0.782, 0.969] | 0.891 [0.781, 0.968] | 0.028 [0.021, 0.035] | 0.033 [0.027, 0.039] | 0.028 [0.021, 0.035] | 0.018 [-0.071, 0.107] | 0.089 [0.027, 0.152] |
| `S1_1.5x` | 0.10 | 0.05 | 1.00 | 0.065 [0.012, 0.164] | 0.049 [0.000, 0.148] | 0.016 [0.011, 0.021] | 0.017 [0.012, 0.022] | 0.015 [0.010, 0.020] | -0.009 [-0.062, 0.045] | 0.098 [0.027, 0.161] |
| `S1_3x` | 0.90 | 0.85 | 1.00 | 0.802 [0.637, 0.934] | 0.798 [0.634, 0.932] | 0.019 [0.015, 0.024] | 0.026 [0.021, 0.031] | 0.020 [0.015, 0.025] | 0.027 [-0.018, 0.071] | 0.071 [-0.009, 0.143] |
| `S2_5x` | 0.00 | 0.70 | 0.86 | 0.109 [0.064, 0.158] | 0.000 [0.000, 0.000] | 0.109 [0.065, 0.158] | 0.108 [0.083, 0.133] | 0.101 [0.078, 0.124] | 0.018 [-0.259, 0.277] | 0.830 [0.545, 1.125] |
| `S3_5x` | 0.05 | 0.00 | 1.00 | 0.025 [0.018, 0.032] | 0.000 [0.000, 0.000] | 0.025 [0.018, 0.033] | 0.027 [0.020, 0.034] | 0.021 [0.016, 0.025] | 0.062 [0.009, 0.107] | 0.018 [-0.036, 0.071] |
| `S4_5x` | 0.00 | 0.35 | 1.00 | 0.028 [0.022, 0.034] | 0.000 [0.000, 0.000] | 0.028 [0.022, 0.034] | 0.027 [0.021, 0.032] | 0.021 [0.016, 0.027] | -0.107 [-0.152, -0.045] | 0.080 [0.036, 0.125] |
| `S1_random_5x` | 0.05 | 0.05 | 1.00 | 0.019 [0.013, 0.024] | 0.000 [0.000, 0.000] | 0.019 [0.013, 0.024] | 0.019 [0.014, 0.024] | 0.020 [0.014, 0.026] | 0.000 [-0.071, 0.089] | 0.071 [0.027, 0.134] |

## Shared intervention-off article contrast

Total variation between inserted `a` and inserted `an` with the
intervention off, on the same 20 held-out prompts:
0.937 [0.891, 0.975].

## Notes

- Total / token-substitution / residual TV come from the additive
  probability-vector split and are reported only when both free articles
  are `a` or `an`.
- Matched-prefix residual TV under `a` and `an` uses all 20 prompts.
- Twin ΔΔ is target-minus-source first-token logits on the seven
  pre-specified pairs.
