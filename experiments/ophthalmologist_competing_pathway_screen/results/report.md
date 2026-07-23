# Ophthalmologist Competing-Pathway Screen

Generated: 2026-07-23T09:25:30.364538+00:00
Model: `google/gemma-3-270m`

## Question

Can suppressing a feature that favors incorrect `a` over correct `an` make Gemma choose `an` while preserving its later ` ophthalm` prediction?

## Exact Prompts

- Article decision: `Someone who studies living organisms is a biologist. Someone who treats eye diseases is`
- Future-word check with the observed article fixed: `Someone who studies living organisms is a biologist. Someone who treats eye diseases is a`

## Expectation Before Intervention

The strongest graph-selected candidate was expected to improve the `an-a` margin because its direct attribution favored `a` over `an` by more than the baseline deficit. A useful intervention also had to preserve the future ` ophthalm` logit within the preregistered 0.125-logit bfloat16 tolerance. No answer feature was augmented.

## Baseline

- `a` logit: 20.125
- `an` logit: 19.125
- `an-a` margin: -1.000
- ` ophthalm` logit: 19.375; rank: 1

## Short Answer

At least one preselected suppression regimen made `an` the top next token while retaining the future answer at rank 1. This is the expected prompt-level result: a removable competing pathway was causally blocking the preparation supported by future-answer information.

- Candidates tested individually: 20
- Individual suppressions improving the margin while preserving ` ophthalm`: 3
- Individual suppressions that made `an` outrank `a`: 0
- Preselected pairs tested: 10
- Pairs improving the margin while preserving ` ophthalm`: 8
- Pairs that made `an` outrank `a`: 1
- Pairs that made `an` the top next token: 1

## Best Article Flip

Suppressing `L13/F10304` + `L14/F1949` at the final prompt token changed:

- `a` by -0.125 logits;
- `an` by 1.000 logits;
- the `an-a` margin by 1.125, ending at 0.125; and
- ` ophthalm` by 0.000, with post-intervention rank 1.
- The top next token was ` an`, and continuation under the same regimen was ` an ophthalmologist.`.

## Candidate Selection

Candidates were fixed from the two source attribution graphs before interventions were run. Each candidate was the same feature at the pre-article position in both graphs, directly favored `a` over `an`, and had absolute direct attribution to ` ophthalm` no greater than 0.05. Candidates were ranked by their direct `a-an` advantage.

Because no individual suppression flipped the article, the screen also tested every pair among the five highest graph-ranked candidates. Pair membership therefore did not depend on the individual intervention outcomes.

## Individual Suppressions

| Feature | Graph `a-an` Advantage | Graph Effect on ` ophthalm` | Δ`a` | Δ`an` | Δ(`an-a`) | Post `an-a` | Δ` ophthalm` | Future Rank | Preserved? | Flipped? | Success? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `L13/F10304` | 1.062 | -0.024 | 0.000 | 0.875 | 0.875 | -0.125 | -0.125 | 1 | True | False | True |
| `L14/F1949` | 0.172 | 0.017 | 0.000 | 0.375 | 0.375 | -0.625 | -0.125 | 1 | True | False | True |
| `L13/F9129` | 0.081 | 0.000 | 0.125 | 0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L16/F4013` | 0.051 | 0.003 | 0.375 | 0.375 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L11/F292` | 0.045 | 0.012 | 0.000 | -0.125 | -0.125 | -1.125 | -0.125 | 1 | True | False | False |
| `L1/F375` | 0.041 | 0.010 | -0.125 | -0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L8/F12890` | 0.040 | -0.010 | 0.250 | 0.250 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L15/F171` | 0.038 | -0.006 | 0.250 | 0.250 | 0.000 | -1.000 | 0.125 | 1 | True | False | False |
| `L15/F2564` | 0.034 | 0.001 | 0.125 | 0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L14/F2622` | 0.032 | -0.001 | 0.125 | 0.000 | -0.125 | -1.125 | -0.125 | 1 | True | False | False |
| `L14/F9721` | 0.027 | 0.010 | -0.125 | 0.000 | 0.125 | -0.875 | 0.000 | 1 | True | False | True |
| `L11/F7366` | 0.023 | 0.004 | 0.250 | 0.250 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L5/F6996` | 0.020 | -0.016 | 0.125 | 0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L9/F6875` | 0.018 | 0.001 | 0.000 | 0.000 | 0.000 | -1.000 | -0.250 | 1 | False | False | False |
| `L0/F774` | 0.015 | 0.009 | 0.125 | 0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L12/F754` | 0.014 | 0.003 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L16/F3012` | 0.012 | -0.018 | 0.125 | 0.125 | 0.000 | -1.000 | 0.000 | 1 | True | False | False |
| `L13/F850` | 0.012 | 0.012 | 0.000 | 0.000 | 0.000 | -1.000 | -0.125 | 1 | True | False | False |
| `L5/F9107` | 0.012 | -0.004 | 0.125 | 0.125 | 0.000 | -1.000 | -0.125 | 1 | True | False | False |
| `L10/F1015` | 0.012 | 0.016 | 0.000 | 0.125 | 0.125 | -0.875 | -0.125 | 2 | False | False | False |

## Preselected Pair Suppressions

| Pair | Δ`a` | Δ`an` | Δ(`an-a`) | Post `an-a` | Top Token | Δ` ophthalm` | Future Rank | Preserved? | `an` Top? | Success? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `L13/F10304` + `L14/F1949` | -0.125 | 1.000 | 1.125 | 0.125 | ` an` | 0.000 | 1 | True | True | True |
| `L13/F10304` + `L13/F9129` | -0.125 | 0.750 | 0.875 | -0.125 | ` a` | -0.125 | 1 | True | False | True |
| `L13/F10304` + `L16/F4013` | 0.125 | 1.000 | 0.875 | -0.125 | ` a` | 0.000 | 1 | True | False | True |
| `L13/F10304` + `L11/F292` | 0.000 | 0.750 | 0.750 | -0.250 | ` a` | 0.000 | 1 | True | False | True |
| `L14/F1949` + `L13/F9129` | 0.000 | 0.250 | 0.250 | -0.750 | ` a` | 0.000 | 1 | True | False | True |
| `L14/F1949` + `L16/F4013` | 0.250 | 0.500 | 0.250 | -0.750 | ` a` | 0.000 | 1 | True | False | True |
| `L14/F1949` + `L11/F292` | -0.125 | 0.000 | 0.125 | -0.875 | ` a` | -0.125 | 1 | True | False | True |
| `L13/F9129` + `L16/F4013` | 0.250 | 0.375 | 0.125 | -0.875 | ` a` | 0.000 | 1 | True | False | True |
| `L13/F9129` + `L11/F292` | 0.000 | -0.125 | -0.125 | -1.125 | ` a` | 0.000 | 1 | True | False | False |
| `L16/F4013` + `L11/F292` | 0.250 | 0.125 | -0.125 | -1.125 | ` a` | 0.000 | 1 | True | False | False |

## Interpretation Boundary

Making `an` the top token and continuing to `ophthalmologist` is evidence for a removable competing pathway in this prompt. It is not yet a general intervention: the feature must be characterized and tested without reselection on held-out vowel-initial occupations, consonant controls, and paraphrases. Small 0.125-logit changes are at this bfloat16 run's measurement resolution and require replication.

## Source Artifacts

- `../ophthalmologist_planning_pilot/results/graphs/article.pt`
- `../ophthalmologist_planning_pilot/results/graphs/future.pt`
