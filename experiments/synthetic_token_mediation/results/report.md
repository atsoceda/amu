# Synthetic token-mediation validation

Generated: 2026-08-25T06:27:05.160285+00:00
Runtime: 0.1s

Ground-truth mechanisms share vocabulary and cues. The assay must
return near-zero token-clamped residual control for the mediated
mechanism and large residual control for the direct-planning mechanism.

## Mechanism: `mediated`

- N cues: 8; article-switch rate: 1.00
- Total TV: 0.905 [0.905, 0.905]
- Article-only TV: 0.905 [0.905, 0.905]
- Residual TV: 0.000 [0.000, 0.000]
- Residual/total TV: 0.000 [0.000, 0.000]
- Article-only top-1 reproduction: 1.00
- CDE_a TV / Δ(target−source): 0.000 [0.000, 0.000] / 0.000 [0.000, 0.000]
- CDE_an TV / Δ(target−source): 0.000 [0.000, 0.000] / 0.000 [0.000, 0.000]
- Interaction (an−a) logit ΔΔ: 0.000 [0.000, 0.000]

Primary residual-control curve under `do(a)` (unsaturated article):

| k | Residual TV | Target−source ΔΔ |
| ---: | ---: | ---: |
| 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 2 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

Secondary curve under `do(an)`:

| k | Residual TV | Target−source ΔΔ |
| ---: | ---: | ---: |
| 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 2 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## Mechanism: `direct`

- N cues: 8; article-switch rate: 1.00
- Total TV: 0.995 [0.995, 0.995]
- Article-only TV: 0.905 [0.905, 0.905]
- Residual TV: 0.090 [0.090, 0.090]
- Residual/total TV: 0.090 [0.090, 0.090]
- Article-only top-1 reproduction: 1.00
- CDE_a TV / Δ(target−source): 0.667 [0.667, 0.667] / 6.250 [6.250, 6.250]
- CDE_an TV / Δ(target−source): 0.090 [0.090, 0.090] / 6.250 [6.250, 6.250]
- Interaction (an−a) logit ΔΔ: 0.000 [0.000, 0.000]

Primary residual-control curve under `do(a)` (unsaturated article):

| k | Residual TV | Target−source ΔΔ |
| ---: | ---: | ---: |
| 0 | 0.667 [0.667, 0.667] | 6.250 [6.250, 6.250] |
| 1 | 0.667 [0.667, 0.667] | 6.250 [6.250, 6.250] |
| 2 | 0.667 [0.667, 0.667] | 6.250 [6.250, 6.250] |
| 3 | 0.667 [0.667, 0.667] | 6.250 [6.250, 6.250] |
| 4 | 0.667 [0.667, 0.667] | 6.250 [6.250, 6.250] |

Secondary curve under `do(an)`:

| k | Residual TV | Target−source ΔΔ |
| ---: | ---: | ---: |
| 0 | 0.090 [0.090, 0.090] | 6.250 [6.250, 6.250] |
| 1 | 0.090 [0.090, 0.090] | 6.250 [6.250, 6.250] |
| 2 | 0.090 [0.090, 0.090] | 6.250 [6.250, 6.250] |
| 3 | 0.090 [0.090, 0.090] | 6.250 [6.250, 6.250] |
| 4 | 0.090 [0.090, 0.090] | 6.250 [6.250, 6.250] |

## Acceptance checks

- `mediated_residual_ratio_small`: **True**
- `direct_cde_a_tv_large`: **True**
- `mediated_k_curve_near_zero`: **True**
- `direct_force_a_k0_residual_large`: **True**
- `mediated_cde_near_zero`: **True**
- `mediated_top1_reproduction`: **True**
- all_passed: **True**

## Interpretation

- The mediated model is a positive control for generated-token relay:
  free-generation TE is large, article-only reproduces it, and the
  token-clamped residual / k-curve stay near zero.
- The direct model is a positive control for residual control: the
  same estimands remain large after article and filler clamping.
- The Gemma full-residual reference remains necessary for stack
  sensitivity; this synthetic check validates the estimands themselves.
