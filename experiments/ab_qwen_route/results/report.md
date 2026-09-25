# A/B route assay on Qwen3-4B: report

_Sources: `summary.json`, `rows.json`. Design (frozen before running): `../run.py`, `../README.md`._

## Motivation

If route share is set by policy movement x leverage, then with the same upstream
change, a mediator with high leverage should carry more of the effect (emission)
than one with low leverage. The A/B code at 4B gives two leverage levels with
everything else fixed.

## What was measured

Upstream change casual -> formal context, applied as a **text edit** (context
words changed) and as a **state edit** (the pre-code position's states at every
layer replaced by the formal prompt's, text unchanged). Six cells: change off/on
x {free, do(A), do(B)}. Outcome: P(formal term) - P(common term). 5 families x
{original, swapped labels} = 10 paired units per bank.

## Results

| Change | Bank | Leverage (source context) | Emission | Persistence | Total |
|---|---|---:|---:|---:|---:|
| State edit | 100% | 0.559 | +0.559 | -0.010 | +0.548 |
| State edit | 75% | -0.011 | 0.000 | +0.004 | +0.004 |
| Text edit | 100% | 0.559 | +0.558 | +0.905 | +1.463 |
| Text edit | 75% | -0.011 | -0.001 | +1.271 | +1.270 |

Emission is higher in the 100% bank in 10/10 units for both changes (exact
one-sided sign-flip p = 0.00098).

## Reading

- The prediction holds: emission follows leverage.
- **Lost decision:** with the state edit, the effect travels about 100% through
  the code when the code has leverage, and vanishes when it does not. The
  decision held at the pre-code position is not retrieved or relayed by any
  other path.
- With the text edit, most of the effect is persistence, which here means the
  term position re-reading the visible context. It is larger when the code is
  uninformative.

## Caveats

Five families; one model; in this context the 75% bank has about zero leverage,
so this is a two-point contrast. The text-edit persistence is presumed
re-reading; its path cannot be split because the casual and formal contexts
differ in token length.

## Place in the thesis

Supports the **leverage law** with a non-grammatical mediator and provides the
**lost decision** result.
