# Planning Gain-of-Function Content Clincher

This experiment asks whether Latent-Planning-style gain-of-function on a frozen
content-supporting feature set produces held-out content-preserving article
repair, trajectory-class switching, or nonspecific disruption.

## Design

1. Selection set: 8 expected-`an` occupation prompts, disjoint from the test set.
2. For each selection prompt, build an article attribution graph (`a`/`an`) and a
   future attribution graph for the listed word's first token after forcing `a`.
3. Keep pre-article features with positive direct effect on both `an` and the
   future token. Rank by recurrence across selection prompts, then freeze the
   top recurring features.
4. On 20 held-out prompts, amplify those frozen features by `5×` their
   prompt-specific activation.
5. Compare against the same amplify operation on activation-matched random
   active features.
6. Score article movement, content preservation, class shifting, and twin matches.

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/planning_gain_content_clincher/run.py
```

Outputs:

- `results/selection.json`
- `results/summary.json`
- `results/report.md`
- `results/graphs/`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `planning_gain_content_clincher` results.
