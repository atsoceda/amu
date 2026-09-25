# A/B route assay on Qwen3-4B (leverage dial)

Design frozen 2026-09-25, before any result; the full specification is the
docstring of `run.py`.

- **Question:** with a non-grammatical mediator whose leverage is set by
  in-context reliability, does the public share of an upstream effect follow the
  mediator's leverage, as policy movement x leverage predicts?
- **Banks:** 100% ("high", gate leverage 0.64) and 75% ("medium", 0.37). The 50%
  bank failed the gate's support check (A/B mass 0.864 < 0.90) and is excluded,
  a disclosed deviation from the frozen gate approved by the user; support
  matters only in banks that are decomposed.
- **Units:** 5 held-out families x {original labels, A/B swapped} = 10 per bank,
  paired across banks.
- **Upstream change:** casual -> formal context, as a text change (private part =
  reading visible text) and as an all-layer state patch at the pre-code position
  with the text unchanged (private part = carried-forward state).
- **Primary prediction:** public(100%) > public(75%) for both changes (exact
  paired sign-flip test), with private roughly equal across banks.

## Results (2026-09-25)

Means over 10 family x labeling units per bank; outcome = change in P(formal
term) - P(common term) at the first term token, \(\tau=1\) over {A, B}.

| Change | Bank | Leverage (source context) | Public | Private | Total |
|---|---|---:|---:|---:|---:|
| State patch | 100% | 0.559 | +0.559 | -0.010 | +0.548 |
| State patch | 75% | -0.011 | -0.000 | +0.004 | +0.004 |
| Text change | 100% | 0.559 | +0.558 | +0.905 | +1.463 |
| Text change | 75% | -0.011 | -0.001 | +1.271 | +1.270 |

- Primary prediction confirmed: public(100%) - public(75%) = +0.559 for both
  changes, positive in 10/10 units (exact one-sided sign-flip p = 0.00098).
- State patch: the carried-forward pre-code state reaches the term only through
  the code. With leverage, the effect is about 100% public; without it, the
  effect vanishes rather than taking a private route.
- Text change: the private part is the term position reading the visible
  context; it is larger when the code is uninformative (1.271 vs 0.905; one-sided
  p for 100% > 75% = 0.95, i.e. the reverse direction is favoured).
- Leverage in the casual source context is about 0 in the 75% bank, below the
  0.37 measured by the gate in the neutral context, so the comparison is
  effectively leverage about 0.56 vs about 0. Leverage varies strongly by family
  (education about 0; cycling up to 1.8). Five families only.
