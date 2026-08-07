# Causal edge independence

Generated: 2026-08-07T14:00:53.498624+00:00
Model: `google/gemma-3-270m`
Runtime seconds: 890.2

## Protocol

1. Measure article logits at pre-article position P (optional b-step).
2. Paste native baseline article b.
3. Keep content-feature clamps active at the same P while predicting noun c.
4. Compare content-on vs content-off under identical fixed b.

## Interpretation

N0: content-on vs off differs on 100% of smoke rows (must be >0 for a valid C→c assay). N1 best `baseline`: c→c signal=0.00, content_changed_on=0.00, matched_same_class=0.00, mean Δ(same−source) logit=-4.898. N1 verdict: no independent within-class C→c dial with these handles (S3/contrast/controls). Supports packaged trajectories for this sparse set. N2 `S3_content_only_amplify_x5`: mean Δ(an−a)=0.172. N2 `contrast_amplify_x5`: mean Δ(an−a)=-0.844. N2 not used as H1 evidence (no validated N1 dial). N3 `S2_article_only_amplify_x5`: mean Δ(an−a)=0.953, content_changed_on=0.00. N3 `S2_article_only_zero`: mean Δ(an−a)=-0.891, content_changed_on=0.00. N4: mean S3@b=385.7, S3@c(fixed b)=385.7; same-class logit gap on=-4.938. N5 skipped (no pure A feature set; gated on N1 dial).

## N1 condition table

| Condition | c→c on | contentΔ on | match same | Δ(same−src) | proto differs |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00 | 0.00 | 0.00 | -4.898 | 0.00 |
| S3_content_only_amplify_x5 | 0.00 | 0.00 | 0.00 | -4.914 | 0.62 |
| S3_content_only_amplify_x8 | 0.00 | 0.00 | 0.00 | -4.961 | 0.50 |
| S3_content_only_zero | 0.00 | 0.00 | 0.00 | -4.883 | 0.75 |
| S2_article_only_amplify_x5 | 0.00 | 0.00 | 0.00 | -4.898 | 0.00 |
| S2_article_only_amplify_x8 | 0.00 | 0.00 | 0.00 | -4.898 | 0.00 |
| control_amplify_x5 | 0.00 | 0.00 | 0.00 | -4.922 | 0.62 |
| control_amplify_x8 | 0.00 | 0.00 | 0.00 | -4.891 | 0.75 |
| contrast_amplify_x5 | 0.00 | 0.00 | 0.00 | -4.906 | 0.50 |
