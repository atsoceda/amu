# Semantic Commitment Sprint Report

Generated: 2026-06-29T16:03:41+0900
Model: `google/gemma-3-270m`
Intervention prompt: `The city most idealized is`

## Baseline

- rank 1: ` the` prob=0.313978 logit=21.473
- rank 2: ` a` prob=0.059314 logit=19.806
- rank 3: ` that` prob=0.056253 logit=19.753
- rank 4: ` New` prob=0.027364 logit=19.033
- rank 5: ` one` prob=0.019214 logit=18.679
- rank 6: ` not` prob=0.018263 logit=18.628
- rank 7: ` San` prob=0.015925 logit=18.491
- rank 8: ` its` prob=0.014413 logit=18.392
- rank 9: ` of` prob=0.012832 logit=18.275
- rank 10: ` ` prob=0.011802 logit=18.192

## Best Interventions

- `semantic_candidate_5_zero` score=0.676; semantic ` Paris` delta_logit=1.003, rank_delta=6; fallback ` the` delta_logit=0.327, rank_delta=0
- `semantic_candidate_top8_double` score=0.670; semantic ` Paris` delta_logit=0.858, rank_delta=5; fallback ` the` delta_logit=0.188, rank_delta=0
- `semantic_candidate_1_double` score=0.573; semantic ` Paris` delta_logit=0.700, rank_delta=6; fallback ` the` delta_logit=0.127, rank_delta=0
- `semantic_candidate_top3_double` score=0.423; semantic ` Paris` delta_logit=0.496, rank_delta=2; fallback ` the` delta_logit=0.073, rank_delta=0
- `semantic_candidate_6_double` score=0.297; semantic ` Paris` delta_logit=0.259, rank_delta=4; fallback ` the` delta_logit=-0.038, rank_delta=0
- `fallback_candidate_2_zero` score=0.224; semantic ` Paris` delta_logit=0.144, rank_delta=2; fallback ` the` delta_logit=-0.080, rank_delta=0
- `fallback_candidate_7_double` score=0.223; semantic ` Paris` delta_logit=0.307, rank_delta=2; fallback ` the` delta_logit=0.085, rank_delta=0
- `semantic_candidate_4_double` score=0.210; semantic ` Paris` delta_logit=0.383, rank_delta=3; fallback ` the` delta_logit=0.173, rank_delta=0
- `semantic_candidate_8_double` score=0.180; semantic ` Paris` delta_logit=0.249, rank_delta=2; fallback ` the` delta_logit=0.068, rank_delta=0
- `semantic_candidate_2_zero` score=0.160; semantic ` Paris` delta_logit=-0.038, rank_delta=4; fallback ` the` delta_logit=-0.198, rank_delta=0
- `fallback_candidate_1_double` score=0.109; semantic ` Paris` delta_logit=-0.409, rank_delta=0; fallback ` the` delta_logit=-0.518, rank_delta=0
- `semantic_candidate_top5_double` score=0.105; semantic ` Paris` delta_logit=0.030, rank_delta=2; fallback ` the` delta_logit=-0.076, rank_delta=0

## Interpretation

This sprint is exploratory. Treat a large directional logit movement as a candidate causal handle, not a final representation label. The next validation step is to rerun the strongest interventions across a larger matched prompt family and test specificity against unrelated semantic targets.
