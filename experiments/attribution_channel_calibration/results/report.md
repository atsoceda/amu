# Attribution-score calibration against causal channel type

Features: 32; held-out prompts: 20.

| Predictor | Outcome | Spearman rho | p |
| --- | --- | ---: | ---: |
| article_attribution | article_margin_effect | 0.530 | 0.0021 |
| article_attribution | total_tv | -0.105 | 0.564 |
| article_attribution | mediator_tv | -0.176 | 0.329 |
| article_attribution | residual_tv_treated | -0.052 | 0.776 |
| article_attribution | fixed_mean_tv | 0.026 | 0.888 |
| future_attribution | article_margin_effect | 0.064 | 0.726 |
| future_attribution | total_tv | -0.122 | 0.503 |
| future_attribution | mediator_tv | -0.071 | 0.69 |
| future_attribution | residual_tv_treated | -0.178 | 0.323 |
| future_attribution | fixed_mean_tv | -0.174 | 0.342 |
