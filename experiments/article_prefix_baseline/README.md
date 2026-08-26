# Article-prefix baseline

Circuit-free control: insert `a` or `an` with no sparse intervention and
measure the full-vocabulary noun-token change.

This estimates how much causal leverage the article already has on this model,
independent of S1.

Run from the repository root after the six-cell sweep (they should not share
RAM with the transcoder model):

```bash
/Users/anthony/miniconda3/bin/python experiments/article_prefix_baseline/run.py
```
