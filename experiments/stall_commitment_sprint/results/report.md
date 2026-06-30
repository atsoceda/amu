# Stall Commitment Sprint Report

Generated: 2026-06-29T17:33:20+0900
Model: `google/gemma-3-270m`
Primary prompt: `The city people most strongly associate with France is`

## Verdict

The sprint found a real commitment-margin handle, but validation suggests it may partly affect broad answer format rather than only France/Paris commitment.

## Human-Readable Result

This was the first better-aligned experiment because it used a country-specific
prompt: `The city people most strongly associate with France is`.

What we did:

- We generated an attribution graph for the France association prompt.
- In that graph, the model's top next token was `the`; `Paris` was second.
- We treated `the` as a possible stall or bridge token: the model may be about
  to say something like "the city of Paris" rather than immediately emitting
  `Paris`.
- We selected features that directly supported either `the` or `Paris`.
- We reran the model with interventions on those features.

The intervention types were:

- `stall_positive_*_zero`: suppress features that supported `the`.
- `paris_positive_*_double`: increase features that supported `Paris`.
- `commitment_combo_*`: combine stall suppression with Paris amplification.

At baseline, the model put `the` ahead of `Paris`. The margin `Paris - the`
was -0.284, so the model looked like it might continue with a phrase instead
of directly answering `Paris`.

The strongest intervention, `commitment_combo_top8`, flipped the top answer:
`Paris` became rank 1 and `the` became rank 2. The margin improved by 1.175.
Importantly, this happened mostly because `the` went down by 1.340, not because
`Paris` went up. That is the closest result so far to the original intuition:
reduce a stall/bridge token and the expected answer wins.

Important negative detail:

The pure stall-suppression interventions on this France prompt did not work by
themselves. For example, `stall_positive_top8_zero` made the `Paris - the`
margin worse by -0.672. The successful `commitment_combo_top8` mixed two
things: suppression of `the`-supporting features and amplification of
`Paris`-supporting features. Because of that, this experiment cannot tell us
whether we found a clean stalling circuit.

However, this is still not enough for the main hypothesis. The family summary
shows that competitor-country prompts also moved under these interventions
(`country_competitor` mean delta_margin = 1.430), and syntax controls moved
somewhat too (`syntax_control` mean delta_margin = 0.282). That means the
intervention may be changing broad answer format or city-answer behavior, not
only France-specific commitment.

Conclusion: this run produced a plausible candidate effect, but did not prove a
commitment circuit. It justified the next cross-country screen, where the key
question became whether analogous interventions help the correct country answer
without injecting the wrong city elsewhere.

## Primary Baseline

- rank 1: ` the` prob=0.247503 logit=18.743
- rank 2: ` Paris` prob=0.186269 logit=18.458
- rank 3: ` France` prob=0.129690 logit=18.096
- rank 4: ` that` prob=0.025772 logit=16.481
- rank 5: ` Marseille` prob=0.023912 logit=16.406
- rank 6: ` ` prob=0.016389 logit=16.028
- rank 7: ` a` prob=0.013417 logit=15.828
- rank 8: ` Toulouse` prob=0.011801 logit=15.699
- rank 9: ` Saint` prob=0.011405 logit=15.665
- rank 10: ` its` prob=0.010238 logit=15.557

Baseline commitment margin `Paris - the`: -0.284

## Best Primary Interventions

- `commitment_combo_top8` (stall_suppression_plus_paris_amplification): delta_margin=1.175; Paris delta_logit=-0.166; the delta_logit=-1.340; Paris rank=1; the rank=2; Paris beats the=True
- `commitment_combo_top5` (stall_suppression_plus_paris_amplification): delta_margin=0.931; Paris delta_logit=-0.755; the delta_logit=-1.687; Paris rank=1; the rank=2; Paris beats the=True
- `paris_positive_top8_double` (paris_amplification): delta_margin=0.921; Paris delta_logit=-0.079; the delta_logit=-1.000; Paris rank=1; the rank=2; Paris beats the=True
- `paris_positive_top5_double` (paris_amplification): delta_margin=0.754; Paris delta_logit=-0.291; the delta_logit=-1.045; Paris rank=1; the rank=2; Paris beats the=True
- `paris_positive_top1_double` (paris_amplification): delta_margin=-0.031; Paris delta_logit=-0.841; the delta_logit=-0.810; Paris rank=2; the rank=1; Paris beats the=False
- `paris_positive_top3_double` (paris_amplification): delta_margin=-0.071; Paris delta_logit=-1.181; the delta_logit=-1.110; Paris rank=3; the rank=1; Paris beats the=False
- `commitment_combo_top3` (stall_suppression_plus_paris_amplification): delta_margin=-0.139; Paris delta_logit=-1.696; the delta_logit=-1.557; Paris rank=3; the rank=1; Paris beats the=False
- `stall_positive_top1_zero` (stall_suppression): delta_margin=-0.144; Paris delta_logit=-0.072; the delta_logit=0.071; Paris rank=2; the rank=1; Paris beats the=False
- `commitment_combo_top1` (stall_suppression_plus_paris_amplification): delta_margin=-0.217; Paris delta_logit=-0.989; the delta_logit=-0.773; Paris rank=2; the rank=1; Paris beats the=False
- `stall_positive_top3_zero` (stall_suppression): delta_margin=-0.252; Paris delta_logit=-0.583; the delta_logit=-0.331; Paris rank=2; the rank=1; Paris beats the=False
- `stall_positive_top5_zero` (stall_suppression): delta_margin=-0.396; Paris delta_logit=-0.964; the delta_logit=-0.568; Paris rank=2; the rank=1; Paris beats the=False
- `stall_positive_top8_zero` (stall_suppression): delta_margin=-0.672; Paris delta_logit=-0.981; the delta_logit=-0.310; Paris rank=3; the rank=1; Paris beats the=False

## Mechanism Note

The best primary intervention, `commitment_combo_top8`, flips the margin mostly
by lowering the `the` logit. But because that intervention also included
Paris-feature amplification, it should not be described as proof that stall
suppression alone works.

## Family Summary

- `ambiguous_france_association` n=24; mean delta_margin=0.078; positive=4/24; Paris beats stall=16/24; mean specificity_margin=-0.311
- `country_competitor` n=24; mean delta_margin=1.430; positive=17/24; Paris beats stall=1/24; mean specificity_margin=-0.274
- `direct_france` n=12; mean delta_margin=0.000; positive=0/12; Paris beats stall=12/12; mean specificity_margin=0.000
- `near_match_france` n=12; mean delta_margin=0.000; positive=0/12; Paris beats stall=0/12; mean specificity_margin=0.000
- `syntax_control` n=24; mean delta_margin=0.282; positive=16/24; Paris beats stall=0/24; mean specificity_margin=-0.037

## Interpretation

This sprint treats the high `the` logit as a potential stall/bridge
continuation rather than a lack of Paris knowledge. It is a useful bridge from
tool validation to hypothesis testing, but it remains inconclusive.

A publishable positive result would require more than flipping this one prompt.
It would need to show that suppressing stall-supporting features reliably helps
the contextually correct answer across matched country prompts, while avoiding
wrong-city induction and syntax-control movement.
