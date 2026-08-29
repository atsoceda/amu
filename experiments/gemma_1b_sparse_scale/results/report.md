# Selection-Criterion Ablation (E1)

Generated: 2026-08-29T10:11:04.659020+00:00
Model: `google/gemma-3-1b-pt`

## Question

Does *how* we pick sparse features determine whether gain-of-function looks like content-preserving wrappers or compiled trajectory packages?

## Design

- Amplify factor: 5.0×
- Selection prompts: 8
- Held-out test prompts: 20
- Features per set: 4
- Near-zero future |attr|: 0.05
- Near-zero article |attr(an)|: 0.05

## Aggregate Comparison

| Set | Decision | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | vs control Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 Dual-effect | `mixed_or_article_shift` | 0.025 | 0.000 | 0.050 | 0.900 | 0.100 | -0.025 |
| S2 Article-only | `mixed_or_article_shift` | -0.212 | 0.000 | 0.050 | 0.850 | 0.100 | 2.969 |
| S3 Content-only | `nonspecific_or_null` | 0.144 | 0.000 | 0.000 | 1.000 | 0.000 | 0.019 |
| S4 Competing / a-favoring | `mixed_or_article_shift` | -3.831 | 0.000 | 0.000 | 0.800 | 0.050 | -3.044 |

## Short Answer

No set cleanly beat controls into wrapper-like or trajectory-like territory. Inspect absolute Δ and illicit mismatch rates before redesigning selection thresholds.

## Selected Features

### S1 Dual-effect

Rule: +attr(an) and +attr(future)
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L18/F5015` | 8 | 0.079 | 0.081 | 0.103 |
| `L15/F2289` | 8 | 0.071 | 0.071 | 0.134 |
| `L15/F6600` | 8 | 0.038 | 0.055 | 0.042 |
| `L0/F4438` | 8 | 0.033 | 0.143 | 0.033 |

Control features: `L15/F186`, `L15/F2801`, `L18/F7630`, `L18/F9440`

### S2 Article-only

Rule: +attr(an), |attr(future)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L18/F8460` | 8 | 0.305 | 0.305 | 0.016 |
| `L1/F399` | 8 | 0.277 | 0.277 | -0.020 |
| `L14/F2097` | 8 | 0.156 | 0.156 | 0.007 |
| `L17/F4391` | 8 | 0.149 | 0.149 | 0.007 |

Control features: `L1/F9677`, `L14/F34`, `L17/F6057`, `L18/F5015`

### S3 Content-only

Rule: +attr(future), |attr(an)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L11/F1082` | 8 | 0.010 | -0.008 | 0.010 |
| `L8/F8388` | 8 | 0.008 | -0.004 | 0.008 |
| `L3/F4789` | 7 | 0.016 | 0.035 | 0.016 |
| `L9/F6100` | 7 | 0.007 | 0.012 | 0.007 |

Control features: `L8/F3352`, `L8/F4653`, `L11/F455`, `L11/F3012`

### S4 Competing / a-favoring

Rule: +attr(a-an), |attr(future)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L17/F6057` | 8 | 0.956 | -2.126 | -0.030 |
| `L18/F8460` | 8 | 0.167 | 0.305 | 0.016 |
| `L13/F8249` | 8 | 0.039 | 0.061 | 0.008 |
| `L17/F4391` | 8 | 0.034 | 0.149 | 0.007 |

Control features: `L13/F1997`, `L13/F6167`, `L17/F414`, `L17/F5378`

## Per-set article-change examples

### S1 Dual-effect

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who studies ancient civilizations through artifacts is` | ` a historian.` | ` an archaeologist.` | False | True | 0.875 |
| `Someone who creates visual art is` | ` an artist.` | ` a painter.` | False | True | -1.125 |

### S2 Article-only

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who studies ancient civilizations through artifacts is` | ` a historian.` | ` an archaeologist.` | False | True | 0.250 |
| `Someone who creates visual art is` | ` an artist.` | ` a painter.` | False | True | -1.500 |

### S3 Content-only

No article or class changes under amplification.

### S4 Competing / a-favoring

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who grows crops and raises livestock is` | ` a farmer.` | ` called a farmer.` | False | False | -3.750 |
| `Someone who extinguishes fires and rescues people is` | ` a firefighter.` | ` called a firefighter.` | False | False | -3.000 |
| `Someone who studies ancient civilizations through artifacts is` | ` a historian.` | ` called an archaeologist.` | False | False | -4.625 |
| `Someone who creates visual art is` | ` an artist.` | ` a painter.` | False | True | -5.812 |

## Interpretation Boundary

Wrapper-like ≥ 0.25 on any set supports Outcome A pathway for later dual-lock tests. Trajectory-like dominance on dual-effect (S1) with no wrapper-like set supports Outcome B. E2 dose-sweeps S1 and any set with wrapper-like ≥ 0.25 or strong article effect.

## Source Artifacts

- `results/selection.json`
- `results/summary.json`
- `results/graphs/`
