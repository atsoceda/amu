# Prompt-aligned article boundary

For each held-out prompt, binary-search the S1 gain at which the `an` minus
`a` logit margin crosses zero. The final bracketing gains define a just-below
and just-above comparison. Noun distributions are measured with both articles
inserted at both gains, yielding an exact local token-substitution plus
fixed-token-residual split across the decoding boundary.

```bash
/Users/anthony/miniconda3/bin/python experiments/prompt_aligned_article_boundary/run.py
```
