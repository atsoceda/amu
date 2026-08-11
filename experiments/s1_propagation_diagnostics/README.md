# S1 propagation diagnostics

This diagnostic verifies that S1 interventions placed at the pre-article
position alter internal computation one token later under a fixed `an`.

For four fixed held-out prompts, it records sparse CLT activation differences
at the article-token position for every layer, plus the resulting noun-logit
distribution shift. Recomputed attention is primary; frozen attention is a
mechanistic ablation.

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/s1_propagation_diagnostics/run.py
```
