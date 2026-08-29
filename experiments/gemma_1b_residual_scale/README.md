# Gemma 3 1B residual-scale comparison

Architecture-independent first phase of the 270M--1B comparison. It uses
the same occupation prompts and natural source/target pairs, but no sparse
dictionary or transcoder.

The runner reports baseline behavior, article-prefix leverage, a full
26-layer natural residual sweep, strength-matched public/private
decomposition, target enhancement/source suppression, wrong-target and
matched-norm controls, and exact stochastic two-article mixtures.

```bash
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_residual_scale/run.py
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_residual_scale/plot.py
```
