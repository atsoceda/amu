# Gemma 270M--1B residual comparison

Natural-strength target patch:

| Regime | Model | Layer (relative) | Target ΔΔ | Private TV | Article change | Target top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| between | gemma_270m | 12 (0.706) | 0.776 | 0.049 | 0.00 | 0.00 |
| between | gemma_1b | 14 (0.560) | 0.846 | 0.054 | 0.00 | 0.00 |
| within | gemma_270m | 12 (0.706) | 0.964 | 0.059 | 0.00 | 0.00 |
| within | gemma_1b | 14 (0.560) | 1.312 | 0.039 | 0.00 | 0.08 |

At natural strength neither model changes the article. The 1B target contrast is modestly larger, especially within class, but distributional private TV is not uniformly larger. This is stronger private target efficacy, not evidence of a public/private carrier reallocation.
