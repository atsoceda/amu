# Semantic Commitment Validation Report

Generated: 2026-06-29T17:34:13+0900
Model: `google/gemma-3-270m`
Pilot interventions reused: semantic_candidate_5_zero, semantic_candidate_top8_double, semantic_candidate_1_double, semantic_candidate_top3_double

## Verdict

The reused pilot interventions still move `Paris`, but they leak into controls; treat them as broad causal handles, not clean semantic-commitment features.

## Human-Readable Result

This validation run tested whether the strongest pilot interventions were
specific to Paris/France-like prompts. They were not.

What we did:

- We reused the strongest interventions from the flawed pilot prompt
  `The city most idealized is`.
- We applied them to several prompt families: France/Paris prompts, near-match
  city prompts, competitor-country prompts such as Germany and Italy, syntax
  controls, and a non-city control.
- We measured whether `Paris` moved up where it should move up, and whether it
  stayed weak where it should not be relevant.

The most important failure is that the interventions increased `Paris` in
places where `Paris` should not be helped. For example:

- On the non-city control prompt, `The instrument most idealized is`,
  `semantic_candidate_top8_double` increased `Paris` by 5.847.
- On the Germany capital prompt, where the expected answer is `Berlin`,
  `semantic_candidate_1_double` increased `Paris` by 1.734.
- On the Italy capital prompt, where the expected answer is `Rome`,
  `semantic_candidate_top3_double` increased `Paris` by 1.702.
- On the syntax-control prompt, `The city most described is`,
  `semantic_candidate_5_zero` increased `Paris` by 0.812.

That pattern is the opposite of what we need. A useful commitment intervention
should help the contextually correct answer while staying weak on unrelated
controls and competitor prompts. Here, the interventions broadly pushed
`Paris`, including in prompts where `Paris` is wrong or irrelevant.

Plain interpretation:

- This is a negative validation result.
- It shows broad Paris induction, not context-sensitive commitment.
- It supports your concern that amplifying answer-token features is likely to
  be confounded.
- It is evidence against using these pilot candidates as the basis for a paper.

Conclusion: this run argues against treating the pilot candidates as clean
semantic-commitment features. It shows broad Paris induction, not a validated
stall-to-commitment mechanism. It is also affected by the bad `idealized`
prompt inherited from the pilot.

## Family Summary

- `domain_control` n=4; mean Paris delta_logit=2.802; mean specificity_margin=-0.386; positive Paris=3/4; positive specific=0/4
- `france_semantic` n=8; mean Paris delta_logit=0.493; mean specificity_margin=-0.125; positive Paris=6/8; positive specific=0/8
- `near_match` n=16; mean Paris delta_logit=0.357; mean specificity_margin=-0.097; positive Paris=14/16; positive specific=1/16
- `semantic_competitor` n=12; mean Paris delta_logit=1.023; mean specificity_margin=-0.137; positive Paris=9/12; positive specific=4/12
- `syntax_control` n=8; mean Paris delta_logit=0.556; mean specificity_margin=-0.240; positive Paris=8/8; positive specific=0/8

## Strongest Prompt-Level Effects

- `semantic_candidate_top8_double` on `noncity_control` (domain_control): Paris delta_logit=5.847, rank_delta=1656, specificity_margin=-0.277
- `semantic_candidate_1_double` on `noncity_control` (domain_control): Paris delta_logit=2.975, rank_delta=1371, specificity_margin=-0.001
- `semantic_candidate_top3_double` on `noncity_control` (domain_control): Paris delta_logit=2.387, rank_delta=1078, specificity_margin=-1.264
- `semantic_candidate_1_double` on `germany_capital_competitor` (semantic_competitor): Paris delta_logit=1.734, rank_delta=53, specificity_margin=0.033
- `semantic_candidate_top3_double` on `germany_capital_competitor` (semantic_competitor): Paris delta_logit=1.719, rank_delta=34, specificity_margin=-0.499
- `semantic_candidate_top3_double` on `italy_capital_competitor` (semantic_competitor): Paris delta_logit=1.702, rank_delta=51, specificity_margin=-0.252
- `semantic_candidate_top8_double` on `italy_capital_competitor` (semantic_competitor): Paris delta_logit=1.631, rank_delta=76, specificity_margin=0.069
- `semantic_candidate_top8_double` on `germany_capital_competitor` (semantic_competitor): Paris delta_logit=1.566, rank_delta=46, specificity_margin=-0.302
- `semantic_candidate_1_double` on `italy_capital_competitor` (semantic_competitor): Paris delta_logit=1.563, rank_delta=75, specificity_margin=0.205
- `semantic_candidate_1_double` on `uk_capital_competitor` (semantic_competitor): Paris delta_logit=1.225, rank_delta=137, specificity_margin=0.075
- `semantic_candidate_1_double` on `clear_france_capital` (france_semantic): Paris delta_logit=1.212, rank_delta=0, specificity_margin=-0.019
- `semantic_candidate_5_zero` on `idealized_city` (near_match): Paris delta_logit=1.003, rank_delta=6, specificity_margin=-0.057
- `semantic_candidate_top3_double` on `clear_france_capital` (france_semantic): Paris delta_logit=0.958, rank_delta=0, specificity_margin=-0.448
- `semantic_candidate_top3_double` on `uk_capital_competitor` (semantic_competitor): Paris delta_logit=0.884, rank_delta=73, specificity_margin=-0.578
- `semantic_candidate_top8_double` on `idealized_city` (near_match): Paris delta_logit=0.858, rank_delta=5, specificity_margin=-0.217
- `semantic_candidate_5_zero` on `syntactic_described` (syntax_control): Paris delta_logit=0.812, rank_delta=8, specificity_margin=-0.046

## Baseline Snapshots

- `idealized_city`: top5 ` the`, ` a`, ` that`, ` New`, ` one`; `Paris` rank=20 prob=0.005538
- `romantic_city`: top5 ` the`, ` Venice`, ` Paris`, ` always`, ` a`; `Paris` rank=3 prob=0.048050
- `fashion_city`: top5 ` the`, ` New`, ` London`, ` Milan`, ` undoubtedly`; `Paris` rank=8 prob=0.028530
- `art_city`: top5 ` the`, ` New`, ` San`, ` Venice`, ` undoubtedly`; `Paris` rank=24 prob=0.006128
- `france_famous_city`: top5 ` the`, ` a`, ` known`, ` one`, ` Paris`; `Paris` rank=5 prob=0.036311
- `clear_france_capital`: top5 ` Paris`, ` the`, ` a`, ` one`, ` known`; `Paris` rank=1 prob=0.423593
- `italy_capital_competitor`: top5 ` Rome`, ` the`, ` Florence`, ` Bologna`, ` Milan`; `Paris` rank=143 prob=0.000317
- `uk_capital_competitor`: top5 ` London`, ` the`, ` a`, ` one`, ` known`; `Paris` rank=279 prob=0.000137
- `germany_capital_competitor`: top5 ` Berlin`, ` the`, ` a`, ` one`, ` Frankfurt`; `Paris` rank=100 prob=0.000633
- `syntactic_described`: top5 ` the`, ` `, ` San`, ` a`, ` New`; `Paris` rank=20 prob=0.005260
- `syntactic_discussed`: top5 ` the`, ` that`, ` `, ` San`, ` its`; `Paris` rank=41 prob=0.002554
- `noncity_control`: top5 ` the`, ` a`, ` that`, ` an`, ` called`; `Paris` rank=1754 prob=0.000004

## Interpretation

A useful validation result would show positive `Paris` movement on near-match
and France prompts while remaining weak or negative on semantic competitors and
syntax/domain controls. Positive movement everywhere is evidence for a broad
perturbation rather than a clean semantic-commitment handle.

For the current hypothesis, this report should be read as a warning label:
answer-token amplification can look impressive in a single prompt while failing
as soon as it is tested against wrong-answer controls.
