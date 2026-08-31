# Correctness-preserving a/an carrier assay

Model: `google/gemma-3-1b-pt`. Independent semantic families: 6 between article classes and 8 within `a`.

All leave-one-family-out folds selected layer 18/25. The earlier frozen layer-14 transfer attempt is preserved in `frozen_layer14/` and fails local efficacy.

| Native-strength result | Between article | Within article |
| --- | ---: | ---: |
| Fixed-target-article lexical ΔΔ | 0.250 [0.167, 0.375] | 0.227 [0.133, 0.328] |
| Public TV, τ=1 | 0.423 [0.352, 0.483] | 0.030 [0.017, 0.046] |
| Private TV, τ=1 | 0.030 [0.018, 0.046] | 0.038 [0.025, 0.053] |
| Public target alignment, τ=1 | 0.370 [0.212, 0.523] | -0.003 [-0.009, 0.002] |
| Private target alignment, τ=1 | 0.018 [0.008, 0.032] | 0.029 [0.008, 0.054] |

TV route interaction: 0.401 [0.335, 0.453]; exact semantic-family permutation p=0.000333.
Target-aligned route interaction: 0.384 [0.229, 0.529].

Interpretation: when the article distinguishes two correct synonyms, the public route dominates; when both synonyms require `a`, intended lexical movement survives primarily in the private component. This is a constructed full-residual intervention result, not evidence of spontaneous or sparse carrier selection.
