# Six-cell family sweep

Generated: 2026-08-29T12:03:59.196117+00:00
Model: `google/gemma-3-1b-pt`
Runtime: 341.2s

Each handle uses intervention off/on crossed with free generation,
inserted `a`, and inserted `an`. Intervention-off cells are shared.

| Handle | Free `an` | Noun changed | Treated-token top-1 match | Total TV | Token-substitution TV | Matched-prefix residual TV | Residual TV under `a` | Residual TV under `an` | Twin ΔΔ `a` | Twin ΔΔ `an` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1_original_top4_5x` | 0.05 | 0.10 | 1.00 | 0.102 [0.018, 0.228] | 0.082 [0.000, 0.213] | 0.021 [0.014, 0.029] | 0.023 [0.016, 0.030] | 0.031 [0.018, 0.045] | -0.089 [-0.205, 0.027] | -0.036 [-0.143, 0.089] |
| `S1_margin_top24_20x` | 0.05 | 0.90 | 1.00 | 0.356 [0.025, 0.686] | 0.343 [0.000, 0.686] | 0.017 [0.009, 0.025] | 0.097 [0.069, 0.127] | 0.266 [0.147, 0.394] | 0.393 [0.089, 0.732] | -1.009 [-1.679, -0.321] |

## Shared intervention-off article contrast

Total variation between inserted `a` and inserted `an` with the
intervention off, on the same 20 held-out prompts:
0.881 [0.813, 0.937].

## Notes

- Total / token-substitution / residual TV come from the additive
  probability-vector split and are reported only when both free articles
  are `a` or `an`.
- Matched-prefix residual TV under `a` and `an` uses all 20 prompts.
- Twin ΔΔ is target-minus-source first-token logits on the seven
  pre-specified pairs.
