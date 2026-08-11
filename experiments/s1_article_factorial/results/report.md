# S1 × article factorial

Generated: 2026-08-10T09:52:11.604922+00:00
Model: `google/gemma-3-270m`
S1 amplify factor: 5.0
Runtime: 262.5s

Primary analysis uses all pre-specified E1 held-out prompts. The free-switching subset is descriptive only.

## Attention recomputed

- S1 free article=`an`: 1.00; S1 free noun changed: 0.95; article-only top-1 reproduction: 1.00.
- Distributional decomposition (S1-free selected article versus baseline-free article): total TV=0.8932, article-only TV=0.8906, same-prefix residual TV=0.0278; residual/total=0.080 [0.025, 0.181].
- Post-hoc generated target/source contrast (descriptive): total ΔΔ=11.637, article-only=11.591, same-prefix residual=0.047.
- Descriptive article-switch subset (N=19): total TV=0.9397, article-only TV=0.9374, residual TV=0.0288; residual/total=0.031 [0.023, 0.040].
- Same-prefix `a`: JS=0.001068 [0.000782, 0.001401], TV=0.033009 [0.027145, 0.038972] (N=20).
- Same-prefix `an`: JS=0.000722 [0.000508, 0.000956], TV=0.027846 [0.020812, 0.034791] (N=20).
- Same-prefix `s1_selected`: JS=0.000722 [0.000505, 0.000957], TV=0.027846 [0.020934, 0.035002] (N=20).

Pre-specified twin target effects:
- force `a`: first-token ΔΔ=0.018 [-0.071, 0.107]; sequence log-odds Δ=0.018 [-0.071, 0.108] (N=7).
- force `an`: first-token ΔΔ=0.089 [0.027, 0.152]; sequence log-odds Δ=0.087 [0.024, 0.152] (N=7).

## Attention frozen

- S1 free article=`an`: 1.00; S1 free noun changed: 0.95; article-only top-1 reproduction: 1.00.
- Distributional decomposition (S1-free selected article versus baseline-free article): total TV=0.8930, article-only TV=0.8906, same-prefix residual TV=0.0262; residual/total=0.078 [0.024, 0.178].
- Post-hoc generated target/source contrast (descriptive): total ΔΔ=11.645, article-only=11.591, same-prefix residual=0.055.
- Descriptive article-switch subset (N=19): total TV=0.9393, article-only TV=0.9374, residual TV=0.0269; residual/total=0.029 [0.022, 0.037].
- Same-prefix `a`: JS=0.001229 [0.000900, 0.001581], TV=0.032447 [0.025677, 0.039195] (N=20).
- Same-prefix `an`: JS=0.000730 [0.000523, 0.000955], TV=0.026234 [0.020137, 0.032792] (N=20).
- Same-prefix `s1_selected`: JS=0.000730 [0.000527, 0.000963], TV=0.026234 [0.020179, 0.032656] (N=20).

Pre-specified twin target effects:
- force `a`: first-token ΔΔ=0.000 [-0.161, 0.152]; sequence log-odds Δ=0.001 [-0.152, 0.152] (N=7).
- force `an`: first-token ΔΔ=0.089 [-0.018, 0.179]; sequence log-odds Δ=0.088 [-0.019, 0.177] (N=7).

## Interpretation guardrails

- A small same-prefix S1-on/off distance supports an article-mediated account for this intervention; it does not prove absence of other noun pathways.
- A nonzero controlled effect establishes residual S1 control but does not identify its internal ontology.
- Frozen-attention and recomputed-attention runs estimate different intervention semantics and are reported separately.
