# A/B few-shot preflight report

The behavioral preflight is promising but does not pass all four gates. No
residual intervention or causal route decomposition was run.

## Native mediator support

`A` or `B` is top-1 in all 45 code cells, matching the observed high completion
consistency. Probability support is stricter:

| Bank | Mean P(A)+P(B) | Minimum | Cells >= .95 |
|---|---:|---:|---:|
| High | .953 | .886 | 9/15 |
| Medium | .888 | .840 | 1/15 |
| Low | .796 | .738 | 0/15 |

Thus top-1 format compliance is perfect, but the prespecified .95 finite-support
criterion does not hold across banks.

## Code-to-noun leverage

At the neutral evaluated context, forcing `B` rather than `A` changes the signed
formal-minus-common noun probability contrast by:

| Bank | Mean probability leverage | Mean logit leverage |
|---|---:|---:|
| High | .071 | .225 |
| Medium | .022 | .075 |
| Low | .032 | .125 |

All five high-bank families have positive leverage. The high-minus-low mean
difference is .039 probability units. The low permutation nearly eliminates
leverage for journalism and cinema, but not for legal, education, or cycling.
The medium condition is not ordered between low and high.

## Context response with code fixed

Changing casual/source to formal/target context while holding the code fixed
produces positive formal-minus-common noun movement in every family. The mean
over forced `A` and forced `B` is .224 (high), .228 (medium), and .245 (low).
This gate passes strongly: semantic context remains useful at `Term:` outside
code identity.

## Code-policy response

The target-minus-source movement in the signed `B`-versus-`A` policy is .130 in
the high bank, .146 in medium, and -.049 in low. The low permutation therefore
removes and reverses the intended register-to-code relationship, but medium does
not form a clean intermediate policy regime. In the high bank `A` remains top-1
even for every formal target context; the effect is graded rather than a greedy
switch.

## Decision

The preflight establishes native pattern completion, positive high-bank branch
leverage, and substantial fixed-code context influence. It does not yet establish
a clean low/medium/high mediator-information continuum because support falls
below .95 outside the high bank, low-bank branch leverage persists in three
families, and medium is nonmonotonic. Pause before residual interventions and
align on whether to revise the label-bank construction or treat this as a partial
preflight.
