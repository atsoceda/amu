# Ophthalmologist Planning Pilot

This targeted mechanistic pilot follows the reproducible behavioral continuation:

```text
Someone who studies living organisms is a biologist.
Someone who treats eye diseases is a ophthalmologist.
```

It asks whether a feature active before the incorrect article supports both the
losing `an` logit and the future answer prefix ` ophthalm`.

The pilot uses a streaming bfloat16 transcoder loader, a batch size of 32, and
disk offload during attribution. The stock Gemma Scope 2 loader temporarily
duplicates roughly 12 GiB of float32 weights and is not suitable for a 16 GiB
machine.

## Result

Gemma chose `a` over `an` by 1.000 logit, then generated `ophthalmologist`.
Four of the 20 tested pre-article features had the required dual causal effect:
suppressing the feature weakened both the losing `an` preparation and the later
` ophthalm` prefix. The strongest was `L13/F10231`; its suppression changed
`a` by 0.000 logits, `an` by -1.125, and ` ophthalm` by -0.250.

This is preliminary evidence that future-answer information contributes to the
correct preparation before Gemma emits the wrong article. It does not yet show
that the feature represents ophthalmologist, generalizes to other prompts, or
that a competing `a` pathway conceals planning.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/ophthalmologist_planning_pilot/run.py
```

Outputs:

- `results/graphs/article.pt`
- `results/graphs/future.pt`
- `results/summary.json`
- `results/report.md`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `ophthalmologist_planning_pilot` results.
