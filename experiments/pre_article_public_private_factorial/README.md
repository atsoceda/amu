# Pre-article public/private factorial

This assay patches a target-cued full decoder-layer residual into a source-cued
run at the final prompt position, before the article is inserted. Source and
target nouns belong to the same native article class (`a`). The private-state
patch is crossed with inserted `a` and `an`.

Layers are selected only on the four development pairs under native inserted
`a`. The selected layer is evaluated on twelve held-out pairs under both article
values and against ten matched-norm random directions per pair. Results are
reported as target-minus-source logit difference-in-differences, full-vocabulary
TV, target top-1 rate, and the public-token × private-state interaction.

Run with:

```bash
/Users/anthony/miniconda3/bin/python experiments/pre_article_public_private_factorial/run.py
```

This is a full-residual capacity/reference intervention. It does not establish
that S1 naturally contains the same lexical state.
