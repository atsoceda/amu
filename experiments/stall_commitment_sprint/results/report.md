# Stall Commitment Sprint Report

Generated: 2026-06-29T17:33:20+0900
Model: `google/gemma-3-270m`
Primary prompt: `The city people most strongly associate with France is`

## Verdict

The sprint found a real commitment-margin handle, but validation suggests it may partly affect broad answer format rather than only France/Paris commitment.

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

The best primary intervention, `commitment_combo_top8`, flips the margin mostly by suppressing the stall token rather than amplifying `Paris`.

## Family Summary

- `ambiguous_france_association` n=24; mean delta_margin=0.078; positive=4/24; Paris beats stall=16/24; mean specificity_margin=-0.311
- `country_competitor` n=24; mean delta_margin=1.430; positive=17/24; Paris beats stall=1/24; mean specificity_margin=-0.274
- `direct_france` n=12; mean delta_margin=0.000; positive=0/12; Paris beats stall=12/12; mean specificity_margin=0.000
- `near_match_france` n=12; mean delta_margin=0.000; positive=0/12; Paris beats stall=0/12; mean specificity_margin=0.000
- `syntax_control` n=24; mean delta_margin=0.282; positive=16/24; Paris beats stall=0/24; mean specificity_margin=-0.037

## Interpretation

This sprint treats the high `the` logit as a potential stall/bridge continuation rather than a lack of Paris knowledge. A publishable positive result would show that suppressing stall-supporting features or amplifying Paris-supporting features increases the `Paris - the` margin on France association prompts without doing the same on syntax and country-competitor controls.
