# Planning Gain-of-Function Content Clincher

Generated: 2026-08-06T13:22:44.565901+00:00
Model: `google/gemma-3-270m`

## Question

If we amplify a frozen set of content-supporting features in the style of *Latent Planning Emerges with Scale*, do held-out continuations show content-preserving article repair, trajectory-class switching, or nonspecific disruption relative to activation-matched random controls?

## Design

- Demonstration: `Someone who studies living organisms is a biologist.`
- Selection prompts: 8 expected-`an` occupation prompts (disjoint from the test set)
- Feature rule: recurring pre-article features with positive direct effect on both `an` and the listed-word first token
- Minimum prompt recurrence: 3
- Frozen content features amplified by 5.0× their prompt-specific activation
- Control: 4 activation-matched random active features, same amplify factor
- Held-out test prompts: 20
- Selection fallback used: False

## Selected Content Features

| Feature | Prompt count | Mean score | Mean Δ-attr `an` | Mean Δ-attr future |
| --- | ---: | ---: | ---: | ---: |
| `L12/F6229` | 8 | 0.143 | 0.256 | 0.143 |
| `L10/F2930` | 8 | 0.024 | 0.097 | 0.024 |
| `L13/F10231` | 8 | 0.020 | 0.861 | 0.020 |
| `L11/F793` | 8 | 0.019 | 0.028 | 0.020 |

## Control Features

`L11/F2954`, `L11/F5451`, `L11/F14119`, `L12/F6421`

## Short Answer

Amplifying frozen content-supporting features changed behavior relative to controls, but mainly by changing content words / vowel-consonant class rather than preserving a fixed later word. The stricter framework is live and the Outcome-2 path is favored.

- Clincher decision: `framework_live_trajectory_path`

## Aggregate Scores

| Metric | Content-feature amplify | Random control amplify |
| --- | ---: | ---: |
| Mean Δ(`an`-`a`) | 3.312 | 0.138 |
| Article moved toward `an` | 1.000 | 0.700 |
| Generated article changed | 0.950 | 0.000 |
| Content preserved | 0.050 | 1.000 |
| Content word changed | 0.950 | 0.000 |
| Class shifted | 0.950 | 0.000 |
| Wrapper-like rate | 0.000 | 0.000 |
| Trajectory-like rate | 0.950 | 0.000 |
| Matched twin word | 0.350 | 0.000 |

## Prompts With Generated Article Changes under Content Amplification

| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Twin match? | Δ(`an`-`a`) |
| --- | --- | --- | --- | --- | --- | ---: |
| `Someone who flies airplanes is` | ` a pilot.` | ` an aviator.` | False | True | True | 2.875 |
| `Someone who represents clients in legal matters is` | ` a lawyer.` | ` an attorney.` | False | True | True | 3.250 |
| `Someone who studies matter and energy is` | ` a physicist.` | ` an astronomer.` | False | True | True | 3.250 |
| `Someone who takes professional pictures is` | ` a photographer.` | ` an artist.` | False | True | True | 3.375 |
| `Someone who studies human behavior and mental processes is` | ` a psychologist.` | ` an anthropologist.` | False | True | True | 3.875 |
| `Someone who studies rocks and earth formations is` | ` a geologist.` | ` an archaeologist.` | False | True | True | 3.375 |
| `Someone who grows crops and raises livestock is` | ` a farmer.` | ` an agriculturist.` | False | True | True | 3.375 |
| `Someone who educates children in schools is` | ` a teacher.` | ` an educator.` | False | True | False | 2.875 |
| `Someone who prepares meals in restaurants is` | ` a chef.` | ` an accountant.` | False | True | False | 3.375 |
| `Someone who extinguishes fires and rescues people is` | ` a firefighter.` | ` an arsonist.` | False | True | False | 3.000 |
| `Someone who treats bone and joint problems is` | ` a physiologist.` | ` an osteopath.` | False | True | False | 3.250 |
| `Someone who studies human cultures and societies is` | ` a sociologist.` | ` an anthropologist.` | False | True | False | 3.625 |
| `Someone who studies ancient civilizations through artifacts is` | ` a historian.` | ` an archaeologist.` | False | True | False | 3.500 |
| `Someone who performs in plays or movies is` | ` a director.` | ` an actor.` | False | True | False | 3.750 |
| `Someone who reviews and revises written content is` | ` a writer.` | ` an editor.` | False | True | False | 3.000 |
| `Someone who examines data and information is` | ` a statistician.` | ` an economist.` | False | True | False | 3.625 |
| `Someone who examines financial records is` | ` a financial analyst.` | ` an accountant.` | False | True | False | 3.000 |
| `Someone who creates new devices or processes is` | ` a scientist.` | ` an engineer.` | False | True | False | 3.375 |
| `Someone who examines things for quality or compliance is` | ` a sociologist.` | ` an economist.` | False | True | False | 2.750 |

## Every Held-Out Prompt under Content Amplification

| Prompt | Expected | Baseline continuation | Intervention continuation | Content preserved? | Class shifted? | Δ(`an`-`a`) |
| --- | --- | --- | --- | --- | --- | ---: |
| `Someone who flies airplanes is` | `a` | ` a pilot.` | ` an aviator.` | False | True | 2.875 |
| `Someone who represents clients in legal matters is` | `a` | ` a lawyer.` | ` an attorney.` | False | True | 3.250 |
| `Someone who studies matter and energy is` | `a` | ` a physicist.` | ` an astronomer.` | False | True | 3.250 |
| `Someone who takes professional pictures is` | `a` | ` a photographer.` | ` an artist.` | False | True | 3.375 |
| `Someone who studies human behavior and mental processes is` | `a` | ` a psychologist.` | ` an anthropologist.` | False | True | 3.875 |
| `Someone who studies rocks and earth formations is` | `a` | ` a geologist.` | ` an archaeologist.` | False | True | 3.375 |
| `Someone who grows crops and raises livestock is` | `a` | ` a farmer.` | ` an agriculturist.` | False | True | 3.375 |
| `Someone who educates children in schools is` | `a` | ` a teacher.` | ` an educator.` | False | True | 2.875 |
| `Someone who prepares meals in restaurants is` | `a` | ` a chef.` | ` an accountant.` | False | True | 3.375 |
| `Someone who extinguishes fires and rescues people is` | `a` | ` a firefighter.` | ` an arsonist.` | False | True | 3.000 |
| `Someone who treats bone and joint problems is` | `an` | ` a physiologist.` | ` an osteopath.` | False | True | 3.250 |
| `Someone who studies human cultures and societies is` | `an` | ` a sociologist.` | ` an anthropologist.` | False | True | 3.625 |
| `Someone who studies ancient civilizations through artifacts is` | `an` | ` a historian.` | ` an archaeologist.` | False | True | 3.500 |
| `Someone who creates visual art is` | `an` | ` an artist.` | ` an artist.` | True | False | 3.750 |
| `Someone who performs in plays or movies is` | `an` | ` a director.` | ` an actor.` | False | True | 3.750 |
| `Someone who reviews and revises written content is` | `an` | ` a writer.` | ` an editor.` | False | True | 3.000 |
| `Someone who examines data and information is` | `an` | ` a statistician.` | ` an economist.` | False | True | 3.625 |
| `Someone who examines financial records is` | `an` | ` a financial analyst.` | ` an accountant.` | False | True | 3.000 |
| `Someone who creates new devices or processes is` | `an` | ` a scientist.` | ` an engineer.` | False | True | 3.375 |
| `Someone who examines things for quality or compliance is` | `an` | ` a sociologist.` | ` an economist.` | False | True | 2.750 |

## Interpretation Boundary

This clincher asks whether Latent-Planning-style gain-of-function on frozen content-supporting features produces content-specific preparation effects on held-out prompts. A useful planning-supportive result requires article movement with content preservation above controls. A useful negative for content-specific planning is a control-beating effect that mainly class-switches content. If content features and random controls look alike, the framework should be redesigned before a larger study.

## Source Artifacts

- `results/selection.json`: recurring dual-effect feature ranking from the selection prompts
- `results/graphs/`: per-selection-prompt article and future attribution graphs
- `results/summary.json`: full machine-readable outputs
