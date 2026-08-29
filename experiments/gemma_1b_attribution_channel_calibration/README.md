# Gemma 3 1B attribution-to-channel calibration

This held-out calibration mirrors the 270M 32-feature study while enforcing the
paper's two-stage identification rule. Thirty-two features are selected from the
eight development attribution graphs (eight per article/future attribution-rank
stratum). Each feature is amplified individually at 5x on the frozen 20-prompt
evaluation set.

The primary analysis asks whether selection attribution predicts local article-
margin efficacy. Channel outcomes are secondary and are reported only for
feature-prompt cases in which both untreated and treated free next tokens remain
within the intended `a`/`an` mediator support.

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_attribution_channel_calibration/run.py
```
