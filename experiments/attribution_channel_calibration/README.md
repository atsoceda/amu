# Attribution-score calibration against causal channel type

Selects individual sparse features using only the eight existing selection
graphs, stratifies them by article and future-token attribution, and evaluates
their article-logit, free-noun, token-substitution, and fixed-token effects on
the twenty held-out occupations. No new attribution graphs are constructed.

```bash
/Users/anthony/miniconda3/bin/python experiments/attribution_channel_calibration/run.py
/Users/anthony/miniconda3/bin/python experiments/attribution_channel_calibration/plot.py
```
