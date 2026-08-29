# Gemma 3 1B attribution-to-channel calibration

Features: 32; held-out prompts: 20; fully mediator-valid features: 31.

Channel correlations exclude any feature that leaves `a`/`an` top-1 support on any held-out prompt.

| Predictor | Outcome | rho | 95% bootstrap CI | n |
| --- | --- | ---: | ---: | ---: |
| article_attribution | article_margin_effect | 0.432 | [0.014, 0.729] | 32 |
| article_attribution | fixed_mean_tv | -0.077 | [-0.422, 0.299] | 32 |
| article_attribution | total_tv_valid | 0.088 | [-0.256, 0.425] | 31 |
| article_attribution | mediator_tv_valid | 0.043 | [-0.316, 0.403] | 31 |
| article_attribution | residual_tv_valid | -0.123 | [-0.478, 0.280] | 31 |
| future_attribution | article_margin_effect | 0.083 | [-0.314, 0.465] | 32 |
| future_attribution | fixed_mean_tv | -0.261 | [-0.548, 0.080] | 32 |
| future_attribution | total_tv_valid | -0.025 | [-0.398, 0.347] | 31 |
| future_attribution | mediator_tv_valid | 0.051 | [-0.330, 0.420] | 31 |
| future_attribution | residual_tv_valid | -0.285 | [-0.589, 0.079] | 31 |
