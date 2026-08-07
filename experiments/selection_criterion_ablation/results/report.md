# Selection-Criterion Ablation (E1)

Generated: 2026-08-07T04:24:24.843340+00:00
Model: `google/gemma-3-270m`

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
| S1 Dual-effect | `trajectory_like` | 3.312 | 0.000 | 0.950 | 0.050 | 0.950 | 3.175 |
| S2 Article-only | `mixed_or_article_shift` | 1.031 | 0.000 | 0.050 | 0.350 | 0.050 | 1.056 |
| S3 Content-only | `mixed_or_article_shift` | 0.113 | 0.000 | 0.050 | 0.950 | 0.050 | 0.300 |
| S4 Competing / a-favoring | `mixed_or_article_shift` | -5.112 | 0.000 | 0.000 | 0.650 | 0.000 | -4.944 |

## Short Answer

S1 dual-effect is trajectory-like and no rule yielded a clean wrapper-like held-out pattern. Favors Outcome B so far; E3 dual lock is still required.

## Selected Features

### S1 Dual-effect

Rule: +attr(an) and +attr(future)
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L12/F6229` | 8 | 0.143 | 0.256 | 0.143 |
| `L10/F2930` | 8 | 0.024 | 0.097 | 0.024 |
| `L13/F10231` | 8 | 0.020 | 0.861 | 0.020 |
| `L11/F793` | 8 | 0.019 | 0.028 | 0.020 |

Control features: `L11/F2954`, `L11/F5451`, `L11/F14119`, `L12/F6421`

### S2 Article-only

Rule: +attr(an), |attr(future)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L14/F1949` | 8 | 0.867 | 0.867 | -0.001 |
| `L13/F10231` | 8 | 0.861 | 0.861 | 0.020 |
| `L5/F383` | 8 | 0.757 | 0.757 | 0.008 |
| `L11/F12690` | 8 | 0.544 | 0.544 | 0.005 |

Control features: `L11/F2954`, `L11/F7366`, `L13/F568`, `L13/F3276`

### S3 Content-only

Rule: +attr(future), |attr(an)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L11/F292` | 8 | 0.021 | -0.013 | 0.021 |
| `L11/F793` | 8 | 0.020 | 0.028 | 0.020 |
| `L11/F6131` | 8 | 0.015 | -0.008 | 0.015 |
| `L13/F568` | 8 | 0.015 | 0.008 | 0.015 |

Control features: `L11/F5441`, `L11/F7366`, `L11/F10514`, `L13/F907`

### S4 Competing / a-favoring

Rule: +attr(a-an), |attr(future)| near zero
Fallback used: False

| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |
| --- | ---: | ---: | ---: | ---: |
| `L13/F10304` | 8 | 1.242 | -0.671 | -0.019 |
| `L14/F1949` | 8 | 0.165 | 0.867 | -0.001 |
| `L13/F9129` | 8 | 0.085 | -0.010 | -0.004 |
| `L11/F292` | 8 | 0.044 | -0.013 | 0.021 |

Control features: `L11/F5451`, `L11/F7366`, `L11/F14119`, `L14/F888`

## Per-set article-change examples

### S1 Dual-effect

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who flies airplanes is` | ` a pilot.` | ` an aviator.` | False | True | 2.875 |
| `Someone who represents clients in legal matters is` | ` a lawyer.` | ` an attorney.` | False | True | 3.250 |
| `Someone who studies matter and energy is` | ` a physicist.` | ` an astronomer.` | False | True | 3.250 |
| `Someone who takes professional pictures is` | ` a photographer.` | ` an artist.` | False | True | 3.375 |
| `Someone who studies human behavior and mental processes is` | ` a psychologist.` | ` an anthropologist.` | False | True | 3.875 |
| `Someone who studies rocks and earth formations is` | ` a geologist.` | ` an archaeologist.` | False | True | 3.375 |
| `Someone who grows crops and raises livestock is` | ` a farmer.` | ` an agriculturist.` | False | True | 3.375 |
| `Someone who educates children in schools is` | ` a teacher.` | ` an educator.` | False | True | 2.875 |
| `Someone who prepares meals in restaurants is` | ` a chef.` | ` an accountant.` | False | True | 3.375 |
| `Someone who extinguishes fires and rescues people is` | ` a firefighter.` | ` an arsonist.` | False | True | 3.000 |
| `Someone who treats bone and joint problems is` | ` a physiologist.` | ` an osteopath.` | False | True | 3.250 |
| `Someone who studies human cultures and societies is` | ` a sociologist.` | ` an anthropologist.` | False | True | 3.625 |

### S2 Article-only

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who flies airplanes is` | ` a pilot.` | ` not a biologist.` | False | False | 1.250 |
| `Someone who studies matter and energy is` | ` a physicist.` | ` an astronomer.` | False | True | 1.125 |
| `Someone who studies human behavior and mental processes is` | ` a psychologist.` | ` called a psychologist.` | False | False | 1.500 |
| `Someone who studies rocks and earth formations is` | ` a geologist.` | ` called a geologist.` | False | False | 0.875 |
| `Someone who grows crops and raises livestock is` | ` a farmer.` | ` also a biologist.` | False | False | 1.625 |
| `Someone who extinguishes fires and rescues people is` | ` a firefighter.` | ` called a firefighter.` | False | False | 1.000 |
| `Someone who studies human cultures and societies is` | ` a sociologist.` | ` called a sociologist.` | False | False | 1.375 |
| `Someone who studies ancient civilizations through artifacts is` | ` a historian.` | ` called an archaeologist.` | False | False | 1.500 |
| `Someone who performs in plays or movies is` | ` a director.` | ` called a director.` | False | False | 1.375 |
| `Someone who examines data and information is` | ` a statistician.` | ` called a statistician` | False | False | 1.000 |
| `Someone who examines financial records is` | ` a financial analyst.` | ` also a biologist.` | False | False | 0.625 |
| `Someone who creates new devices or processes is` | ` a scientist.` | ` called a scientist.` | False | False | 1.625 |

### S3 Content-only

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who creates new devices or processes is` | ` a scientist.` | ` an engineer.` | False | True | 0.375 |

### S4 Competing / a-favoring

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| `Someone who studies matter and energy is` | ` a physicist.` | ` called a physicist.` | False | False | -4.250 |
| `Someone who studies human behavior and mental processes is` | ` a psychologist.` | ` called a psychologist.` | False | False | -5.438 |
| `Someone who studies human cultures and societies is` | ` a sociologist.` | ` called a sociologist.` | False | False | -5.875 |
| `Someone who creates visual art is` | ` an artist.` | ` called an artist.` | False | False | -5.250 |
| `Someone who examines data and information is` | ` a statistician.` | ` called a statistician` | False | False | -5.812 |
| `Someone who creates new devices or processes is` | ` a scientist.` | ` called a scientist.` | False | False | -4.125 |
| `Someone who examines things for quality or compliance is` | ` a sociologist.` | ` called a scientist.` | False | False | -4.375 |

## Interpretation Boundary

Wrapper-like ≥ 0.25 on any set supports Outcome A pathway for later dual-lock tests. Trajectory-like dominance on dual-effect (S1) with no wrapper-like set supports Outcome B. E2 dose-sweeps S1 and any set with wrapper-like ≥ 0.25 or strong article effect.

## Source Artifacts

- `results/selection.json`
- `results/summary.json`
- `results/graphs/`
