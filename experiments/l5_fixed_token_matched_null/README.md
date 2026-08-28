# L5/F383 matched null

Builds an empirical fixed-token null for the S2 feature `L5/F383`. Twenty
frozen layer-5 features are selected from the eight development/selection
prompts to match its mean activation while excluding all S1--S4 features.
Each single feature is amplified at 5x on the 20 held-out prompts under fixed
`an` and compared on full-vocabulary TV and pre-specified twin contrasts.

```bash
/Users/anthony/miniconda3/bin/python experiments/l5_fixed_token_matched_null/run.py
```
