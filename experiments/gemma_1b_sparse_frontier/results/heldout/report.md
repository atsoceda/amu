# Six-cell family sweep

Generated: 2026-08-30T02:02:59.096165+00:00
Model: `google/gemma-3-1b-pt`
Runtime: 190.6s

Each handle uses intervention off/on crossed with free generation,
inserted `a`, and inserted `an`. Intervention-off cells are shared.

| Handle | Free `an` | Noun changed | Treated-token top-1 match | Total TV | Token-substitution TV | Matched-prefix residual TV | Residual TV under `a` | Residual TV under `an` | Twin ΔΔ `a` | Twin ΔΔ `an` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1_margin_frontier_top1_7.5x` | 0.10 | 0.05 | 1.00 | 0.046 [0.008, 0.114] | 0.033 [0.000, 0.100] | 0.012 [0.007, 0.019] | 0.014 [0.008, 0.021] | 0.016 [0.009, 0.024] | 0.027 [-0.045, 0.107] | -0.036 [-0.089, 0.036] |

## Shared intervention-off article contrast

Total variation between inserted `a` and inserted `an` with the
intervention off, on the same 20 held-out prompts:
0.881 [0.812, 0.938].

## Notes

- Total / token-substitution / residual TV come from the additive
  probability-vector split and are reported only when both free articles
  are `a` or `an`.
- Matched-prefix residual TV under `a` and `an` uses all 20 prompts.
- Twin ΔΔ is target-minus-source first-token logits on the seven
  pre-specified pairs.
