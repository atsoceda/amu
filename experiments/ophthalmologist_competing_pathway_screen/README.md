# Ophthalmologist Competing-Pathway Screen

This experiment follows the `ophthalmologist_planning_pilot`. That pilot found
features active before the article that causally support both `an` and the
future ` ophthalm` token. This screen asks the complementary question:

> Can suppression remove a pathway favoring incorrect `a`, allowing `an` to
> win while preserving the future answer?

Candidates are selected before intervention from the existing attribution
graphs. A candidate must:

1. be active at the pre-article token position in both graphs;
2. have a positive direct advantage for `a` over `an`; and
3. have an absolute direct effect on ` ophthalm` no greater than 0.05.

Candidates are suppressed individually. If no individual feature flips the
article, every pair among the five highest graph-ranked candidates is also
tested. The pair membership is fixed by graph attribution, not selected from
the individual intervention outcomes. No answer feature is augmented.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/ophthalmologist_competing_pathway_screen/run.py
```

If either source graph is missing, the runner regenerates that graph through
the memory-safe source pilot first.

To regenerate it, prompt an agent with:

> Regenerate the `ophthalmologist_competing_pathway_screen` results.

## Result

No individual suppression changed the selected article, although the strongest
candidate moved the `an-a` margin from -1.000 to -0.125. One preselected pair,
`L13/F10304 + L14/F1949`, produced the expected behavioral correction:

```text
baseline:      a ophthalmologist.
intervention: an ophthalmologist.
```

The pair improved the `an-a` margin by 1.125 logits, ending at +0.125. The
` ophthalm` logit changed by 0.000 and remained rank 1. This establishes a
prompt-level causal result, not a reusable intervention; the fixed pair must
next be tested without feature reselection on held-out prompts and controls.
