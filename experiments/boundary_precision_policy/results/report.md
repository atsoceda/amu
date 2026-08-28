# Boundary precision and stochastic-policy audit

Precision-audited prompts: 3; stochastic-policy prompts: 19.
Native dense-grid unique margins: 3; float32-head unique margins: 43.

| Temperature | Total TV | Policy TV | Fixed-token TV | Policy/total cosine | a/an mass low/high |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.456 | 0.454 | 0.015 | 0.999 | 0.999/1.000 |
| 0.25 | 0.212 | 0.211 | 0.015 | 0.996 | 0.994/0.996 |
| 0.5 | 0.109 | 0.108 | 0.015 | 0.987 | 0.982/0.984 |
| 1 | 0.057 | 0.055 | 0.015 | 0.957 | 0.899/0.904 |
