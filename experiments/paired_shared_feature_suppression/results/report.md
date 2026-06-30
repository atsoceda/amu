# Paired Shared Feature Suppression Report

Generated: 2026-06-30T11:40:14+0900
Model: `google/gemma-3-270m`

## Question

This experiment asks whether the two individually promising shared features from the previous screen work better when suppressed together.

The key pair is `L7/F89 + L10/F9037`, applied at the final token position of each prompt. The goal is not merely to lower `the`. The goal is to increase the expected-country-city margin over `the` without increasing wrong-country city tokens.

## What Was Suppressed

- `l7_f89`: layer=7, feature_idx=89, source_pos=9. Best individual behavioral lead from shared_stall_suppression: improved expected-city-over-the margin on 6 of 8 ambiguous prompts with small average wrong-city movement.
- `l10_f9037`: layer=10, feature_idx=9037, source_pos=9. Second individual behavioral lead from shared_stall_suppression: improved expected-city-over-the margin on 6 of 8 ambiguous prompts, but looked more entangled with expected-city suppression.
- `l5_f383`: layer=5, feature_idx=383, source_pos=9. High direct-effect comparison feature. It strongly supported the stall token in the attribution overlap, but was not selected as the cleanest behavioral lead.

## Regimens Tested

- `l7_f89`: Suppress L7/F89 at the final token position. Feature keys: l7_f89.
- `l10_f9037`: Suppress L10/F9037 at the final token position. Feature keys: l10_f9037.
- `l7_f89_plus_l10_f9037`: Suppress the two individually promising shared features together at the final token position. Feature keys: l7_f89, l10_f9037.
- `l7_f89_plus_l5_f383`: Suppress the cleaner L7/F89 lead plus the highest-direct-effect shared feature as a comparison bundle. Feature keys: l7_f89, l5_f383.

## Short Answer

The pair has directional signal, but it does not clearly beat the cleaner single-feature intervention.

The pair did improve the average ambiguous-prompt margin more than either single feature. That is the favorable part. It is not yet a clean win because it produced zero hard-prompt flips and its wrong-city/specificity behavior was worse than `L7/F89` alone. In plain terms: the pair pushed `the` down, but it often pushed the expected city down too and sometimes let wrong city tokens move more than the correct city token.

## Did The Pair Beat The Singles?

- Pair minus `L7/F89` on ambiguous mean margin: 0.059
- Pair minus `L10/F9037` on ambiguous mean margin: 0.089
- Pair beat `L7/F89` on ambiguous mean margin: True
- Pair beat `L10/F9037` on ambiguous mean margin: True
- Pair produced more hard-prompt flips than `L7/F89`: False
- Pair produced more hard-prompt flips than `L10/F9037`: False

## Aggregate Results

| Regimen | Features | Ambiguous Mean Margin Change | Ambiguous Helped | Hard Prompt Flips | Hard Prompt Count | Expected Logit Change | `the` Logit Change | Wrong-City Logit Change | Specificity | Control Margin Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `l7_f89_plus_l10_f9037` | l7_f89, l10_f9037 | 0.148 | 6/8 | 0 | 4 | -0.217 | -0.240 | 0.068 | -0.348 | 0.124 |
| `l7_f89_plus_l5_f383` | l7_f89, l5_f383 | 0.123 | 7/8 | 0 | 4 | 0.014 | -0.245 | -0.141 | -0.344 | -0.046 |
| `l7_f89` | l7_f89 | 0.090 | 6/8 | 0 | 4 | -0.018 | -0.063 | 0.007 | -0.056 | 0.058 |
| `l10_f9037` | l10_f9037 | 0.059 | 6/8 | 0 | 4 | -0.202 | -0.180 | 0.066 | -0.324 | 0.034 |

## Key Pair Prompt-Level Results

This table shows the actual ambiguous prompts for the main pair. `Margin Change` means expected city logit minus `the` logit, after intervention minus before intervention. Positive is the desired direction.

| Prompt | Expected | Before city > `the`? | After city > `the`? | Margin Change | Expected Logit Change | `the` Logit Change | Wrong-City Check | Plain Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `The city people most strongly associate with France is` | `Paris` | False | False | 0.021 | -0.237 | -0.259 | specificity weakened | helped margin |
| `The city most people associate with France is` | `Paris` | True | True | 0.146 | -0.285 | -0.431 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city people most strongly associate with Germany is` | `Berlin` | False | False | 0.099 | -0.427 | -0.525 | specificity weakened | helped margin |
| `The city most people associate with Germany is` | `Berlin` | True | True | 0.332 | -0.500 | -0.832 | specificity weakened | helped margin |
| `The city people most strongly associate with Italy is` | `Rome` | False | False | 0.291 | -0.128 | -0.419 | no obvious wrong-city pollution | helped margin |
| `The city most people associate with Italy is` | `Rome` | True | True | 0.514 | 0.045 | -0.469 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city people most strongly associate with Spain is` | `Madrid` | False | False | -0.159 | -0.651 | -0.492 | specificity weakened | hurt margin |
| `The city most people associate with Spain is` | `Madrid` | True | True | -0.056 | -0.698 | -0.641 | wrong-city movement exceeded expected-city movement | hurt margin |

## Single-Feature Comparison On Ambiguous Prompts

These tables are included so the pair can be judged against the two ingredients, not just against baseline.

### L7/F89 Alone

| Prompt | Expected | Before city > `the`? | After city > `the`? | Margin Change | Expected Logit Change | `the` Logit Change | Wrong-City Check | Plain Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `The city people most strongly associate with France is` | `Paris` | False | False | -0.002 | -0.076 | -0.073 | wrong-city movement exceeded expected-city movement | hurt margin |
| `The city most people associate with France is` | `Paris` | True | True | 0.076 | 0.058 | -0.018 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city people most strongly associate with Germany is` | `Berlin` | False | False | 0.082 | -0.090 | -0.171 | no obvious wrong-city pollution | helped margin |
| `The city most people associate with Germany is` | `Berlin` | True | True | 0.171 | 0.001 | -0.170 | no obvious wrong-city pollution | helped margin |
| `The city people most strongly associate with Italy is` | `Rome` | False | False | 0.158 | 0.001 | -0.157 | no obvious wrong-city pollution | helped margin |
| `The city most people associate with Italy is` | `Rome` | True | True | 0.233 | 0.085 | -0.147 | no obvious wrong-city pollution | helped margin |
| `The city people most strongly associate with Spain is` | `Madrid` | False | False | -0.026 | -0.171 | -0.144 | no obvious wrong-city pollution | hurt margin |
| `The city most people associate with Spain is` | `Madrid` | True | True | 0.026 | -0.094 | -0.121 | wrong-city movement exceeded expected-city movement | helped margin |

### L10/F9037 Alone

| Prompt | Expected | Before city > `the`? | After city > `the`? | Margin Change | Expected Logit Change | `the` Logit Change | Wrong-City Check | Plain Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `The city people most strongly associate with France is` | `Paris` | False | False | 0.037 | -0.153 | -0.191 | specificity weakened | helped margin |
| `The city most people associate with France is` | `Paris` | True | True | 0.075 | -0.333 | -0.408 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city people most strongly associate with Germany is` | `Berlin` | False | False | 0.017 | -0.350 | -0.368 | no obvious wrong-city pollution | helped margin |
| `The city most people associate with Germany is` | `Berlin` | True | True | 0.156 | -0.499 | -0.655 | specificity weakened | helped margin |
| `The city people most strongly associate with Italy is` | `Rome` | False | False | 0.136 | -0.142 | -0.278 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city most people associate with Italy is` | `Rome` | True | True | 0.271 | -0.053 | -0.325 | wrong-city movement exceeded expected-city movement | helped margin |
| `The city people most strongly associate with Spain is` | `Madrid` | False | False | -0.136 | -0.498 | -0.362 | specificity weakened | hurt margin |
| `The city most people associate with Spain is` | `Madrid` | True | True | -0.085 | -0.610 | -0.525 | specificity weakened | hurt margin |

## Interpretation Rules

- A useful result is not `the went down` by itself. A useful result is the expected city improving relative to `the`.
- A stronger result flips a hard prompt where `the` beat the expected city before the intervention.
- A polluted result is one where the wrong-country city movement is larger than the expected-city movement.
- A reusable intervention should help several countries with the same feature set, not require choosing a different feature per country.

## What This Decides Next

If `L7/F89 + L10/F9037` beats `L7/F89` alone without wrong-city pollution, the next experiment should move to held-out prompt phrasings. If it does not beat `L7/F89`, then the next held-out test should use `L7/F89` alone as the cleaner candidate.

## Regenerate

Run `/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/paired_shared_feature_suppression/run.py`.