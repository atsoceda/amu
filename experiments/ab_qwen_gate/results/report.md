# A/B learnability gate on Qwen3: report

_Sources: `results/<model>/summary.json`, `rows.json`. Design and gate thresholds: `../run.py`, `../README.md`._

## Motivation

The A/B code is the only mediator we can build whose informativeness we set by
design: few-shot demonstrations tie `Code: A` / `Code: B` to everyday vs formal
terms with 100%, 75% or 50% reliability. Before any route assay, the model must
actually learn the code in both directions. Gemma 3 1B did not (it wrote `A`
whatever the context), so the earlier A/B results could not support any claim.

## What was measured

Behaviour only, on the frozen preflight config (5 held-out families):
whether the code follows the context (P(B | formal) - P(B | casual)), whether
forcing the code moves the term (forced-code leverage), and A/B support. Gate:
G1 code follows context >= 0.30; G2 leverage >= 0.15 at 100%; G3 leverage at 50%
<= one third of 100%; G4 A/B mass >= 0.90 in every bank.

## Results

| Model | Code follows context (100%) | Leverage 100 / 75 / 50% | A/B mass (100 / 75 / 50%) | Gate |
|---|---:|---|---|---|
| Qwen3-0.6B | +0.02 | -0.01 / +0.03 / +0.05 | >= 0.93 | fail (G1-G3) |
| Qwen3-1.7B | +1.00 | +0.24 / +0.19 / +0.10 | about 1.00 | fail (G3: 0.098 > 0.081) |
| Qwen3-4B | +1.00 | +0.64 / +0.37 / -0.20 | 1.00 / 1.00 / 0.86 | fail (G4 only) |

## Reading

An arbitrary in-context code becomes a mediator with scale: at 0.6B it is not
learned; at 1.7B it is learned and its leverage falls with reliability; at 4B
leverage is large and tracks reliability cleanly. The 4B failure is support in
the uninformative 50% bank, where the model sometimes writes neither code.
Leverage is therefore a quantity we can dial with a non-grammatical mediator,
and it grows with scale.

## Caveats

Five families; behavioural only. By the user's decision the route assay uses the
100% and 75% banks only (disclosed deviation); see `experiments/ab_qwen_route`.

## Place in the thesis

Evidence for **dialable leverage** and, tentatively, **leverage reflecting
association strength** (in-context codes weaker than agreement at small scale).
