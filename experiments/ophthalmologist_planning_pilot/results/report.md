# Ophthalmologist Planning Pilot

Generated: 2026-07-23T08:54:37.034537+00:00
Model: `google/gemma-3-270m`

## Question

Before Gemma predicts the incorrect article `a`, are there active features that also causally support the future answer prefix ` ophthalm`?

## Exact Prompts

- Article decision: `Someone who studies living organisms is a biologist. Someone who treats eye diseases is`
- Future-word decision with the observed article fixed: `Someone who studies living organisms is a biologist. Someone who treats eye diseases is a`

`ophthalmologist` tokenizes as ` ophthalm` + `ologist`. This pilot attributes the first future-token prefix, following the paper's feature-selection rule.

## Baseline

- Article logits: `a`=20.125, `an`=19.125; `an-a` margin=-1.000.
- Future prefix ` ophthalm`: rank=1, probability=0.1459.

## Short Answer

The pilot found at least one feature active before the incorrect article whose suppression weakens both the losing `an` preparation and the later ` ophthalm` prefix. This is preliminary evidence of a future-answer pathway, not yet evidence that a competing `a` circuit conceals it.

- Shared pre-article features in both attribution graphs: 104
- Shared features with positive direct effects on both `an` and ` ophthalm`: 45
- Suppression candidates tested: 20
- Candidates whose suppression reduced both the future prefix and the `an-a` margin: 4

## Strongest Individual Result

Suppressing `L13/F10231` at the final prompt token:

- changed the incorrect `a` logit by 0.000;
- changed the losing `an` logit by -1.125;
- therefore changed the `an-a` margin by -1.125; and
- changed the later ` ophthalm` logit by -0.250.

This is the cleanest result because the same suppression selectively weakens the grammatically correct preparation and the future answer prefix while leaving the chosen `a` logit unchanged. It does not yet tell us what the feature represents.

## How Candidates Were Selected

The article and future-word attribution graphs were built separately. A candidate had to be the exact same circuit-tracer feature at the pre-article token position in both graphs, with a positive direct attribution to both `an` and ` ophthalm`. The 20 candidates with the largest minimum of those two direct effects were suppressed one at a time. No answer feature was augmented.

## Candidate Interventions

| Feature | Activation | Direct Effect on `a` | Direct Effect on `an` | Direct Effect on ` ophthalm` | Suppression Δ`a` | Suppression Δ`an` | Suppression Δ(`an-a`) | Suppression Δ` ophthalm` | Dual Effect? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `L13/F1230` | 1480.000 | 0.099 | 0.111 | 0.092 | 0.000 | 0.000 | 0.000 | -0.125 | False |
| `L12/F6229` | 1480.000 | 0.191 | 0.270 | 0.090 | -0.250 | -0.375 | -0.125 | -0.125 | True |
| `L13/F10231` | 912.000 | -0.480 | 0.891 | 0.040 | 0.000 | -1.125 | -1.125 | -0.250 | True |
| `L12/F4772` | 1184.000 | -0.017 | 0.039 | 0.037 | 0.125 | 0.125 | 0.000 | -0.125 | False |
| `L5/F383` | 328.000 | 0.543 | 0.613 | 0.023 | 0.250 | 0.250 | 0.000 | 0.125 | False |
| `L10/F2388` | 968.000 | 0.293 | 0.295 | 0.020 | 0.125 | 0.000 | -0.125 | -0.250 | True |
| `L14/F3283` | 1584.000 | -0.005 | 0.211 | 0.019 | -0.250 | -0.375 | -0.125 | -0.125 | True |
| `L13/F2593` | 1240.000 | 0.008 | 0.017 | 0.046 | -0.125 | -0.125 | 0.000 | -0.125 | False |
| `L14/F1949` | 1136.000 | 1.062 | 0.891 | 0.017 | 0.000 | 0.375 | 0.375 | -0.125 | False |
| `L8/F573` | 328.000 | 0.010 | 0.019 | 0.017 | 0.250 | 0.250 | 0.000 | -0.125 | False |
| `L13/F850` | 336.000 | 0.025 | 0.013 | 0.012 | 0.000 | 0.000 | 0.000 | -0.125 | False |
| `L11/F793` | 191.000 | 0.016 | 0.017 | 0.012 | 0.250 | 0.250 | 0.000 | -0.125 | False |
| `L10/F7032` | 748.000 | 0.516 | 0.539 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| `L1/F375` | 268.000 | 0.273 | 0.232 | 0.010 | -0.125 | -0.125 | 0.000 | 0.000 | False |
| `L14/F9721` | 540.000 | 0.570 | 0.543 | 0.010 | -0.125 | 0.000 | 0.125 | 0.000 | False |
| `L10/F2930` | 145.000 | 0.115 | 0.118 | 0.010 | 0.125 | 0.125 | 0.000 | -0.125 | False |
| `L0/F774` | 66.500 | 0.146 | 0.132 | 0.009 | 0.125 | 0.125 | 0.000 | 0.000 | False |
| `L12/F586` | 179.000 | -0.003 | 0.008 | 0.009 | 0.000 | 0.000 | 0.000 | -0.125 | False |
| `L11/F12690` | 386.000 | 0.578 | 0.578 | 0.007 | -0.125 | -0.125 | 0.000 | 0.000 | False |
| `L12/F548` | 175.000 | 0.025 | 0.026 | 0.007 | 0.125 | 0.125 | 0.000 | -0.125 | False |

## Interpretation Boundary

A dual-effect feature is evidence that a representation active before the article contributes to both the losing grammatical preparation `an` and the later answer prefix. Feature semantics still require validation across activating examples and held-out prompts. This single prompt cannot establish a general concealed-planning mechanism.

The model and transcoder ran in bfloat16 to fit the 16 GiB machine, so small changes are quantized in roughly 0.125-logit increments in this run. The strongest 1.125-logit article-margin effect is substantially larger than that resolution; the 0.125 effects require replication.

## Artifacts

- `results/graphs/article.pt`: attribution graph for `a` and `an`.
- `results/graphs/future.pt`: attribution graph for ` ophthalm` after the observed `a`.
- `results/summary.json`: machine-readable baselines, candidates, and interventions.
