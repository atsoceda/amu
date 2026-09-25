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
