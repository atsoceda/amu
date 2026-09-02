# Repaired neutral matched-semantic triads

This experiment removes the obsolete first-letter prompt and lexical-preference
rules from the earlier triad construction. Every evaluated prompt is neutral.
Explicit donor prompts construct two directions from the same source state:
one to a same-article target and one to a cross-article target.

The frozen screen is route-blind. It checks human-curated semantic equivalence,
single-token lexical realizations, local fixed-target-article efficacy at layer
17 and strength 1, and retention of the binary `a/an` mediator support. It never
computes or selects on public/private route contrast.

Run with checkpointing:

```bash
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads_repaired/run.py screen
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads_repaired/run.py assay
```

The assay refuses to run unless at least eight independent triads pass the
frozen gate. Primary inference is paired across the two arms of each semantic
family at layer 17, strength 1, and article-policy temperature 1.
