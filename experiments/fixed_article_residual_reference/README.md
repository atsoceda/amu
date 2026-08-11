# Fixed-article full-residual reference

This experiment supplies an in-stack upstream sensitivity control for the
S1 generated-article factorial.

For each source/target pair, the core occupation sentence is identical and
the prompt differs only in an explicit source-versus-target vocabulary cue.
The complete target-cued decoder-layer output residual is patched into the
source-cued run at the fixed article-token position in one layer.

Layers are selected on four fixed development pairs. The selected layer is
then evaluated on four held-out pairs against ten matched-norm random
directions, averaging random seeds within each prompt pair before inference.

This intervention is:

- an upstream full-residual noun-computation reference at the article position;
- attention-recomputed downstream;
- not an oracle;
- not evidence that a sparse feature carries the same information.

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/fixed_article_residual_reference/run.py
```
