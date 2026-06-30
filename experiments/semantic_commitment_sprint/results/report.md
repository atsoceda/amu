# Semantic Commitment Sprint Report

Generated: 2026-06-29T17:35:04+0900
Model: `google/gemma-3-270m`
Intervention prompt: `The city most idealized is`

## Human-Readable Result

This experiment should be treated as an early pilot, not evidence for the
stall/commitment hypothesis. It answered one narrow tooling question: can we
select circuit-tracer features from an attribution graph and causally move next
token logits by clamping those features during a model rerun? Yes. It did not
answer the research question we care about.

What we did:

- We used the prompt `The city most idealized is`.
- We looked for features that supported either `Paris` or the fallback token
  `the`.
- We reran the model while clamping selected features, either suppressing them
  to zero or increasing them.
- We measured whether `Paris` moved up relative to `the`.

Why this prompt is flawed:

`The city most idealized is` is too broad for the research question. It is not
country-specific. If an intervention moves the model toward `Paris`, that could
mean only that `Paris` is a culturally salient city. It does not show that the
model is resolving an ambiguous country-conditioned prompt, and it does not
show that we found a reusable "stop stalling and commit" mechanism.

The best interventions raised the `Paris` logit by up to 1.003, but they did
not clearly suppress the main stall token `the`. For the top intervention,
`semantic_candidate_5_zero`, `Paris` increased by 1.003 while `the` also
increased by 0.327. That is not a clean shift from stalling to commitment.

Plain interpretation:

- Positive result here: circuit-tracer interventions can move logits.
- Negative result here: the movement is not specific enough to support our
  hypothesis.
- Main confound: we may have found Paris-salience features, not commitment
  features.
- Publication value: tooling validation only; not a publishable mechanistic
  result by our current bar.

Conclusion: this run validated that circuit-tracer interventions can move
logits, but it did not validate a reusable commitment or stalling circuit.
Results from this run should not be used as positive evidence for the main
hypothesis.

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

This sprint is exploratory. Treat a large directional logit movement as a
candidate causal handle, not a final representation label. The correct lesson is
that broad answer-token amplification is easy to produce and easy to
misinterpret. The next experiments therefore moved away from this prompt and
toward country-specific prompts where the expected answer is known in advance
and where wrong-city induction can be measured directly.
