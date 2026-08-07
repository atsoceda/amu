# Selective b-step intervention with force-native article

Generated: 2026-08-07T08:50:01.267032+00:00
Model: `google/gemma-3-270m`
Amplify factor: 5.0
Runtime seconds: 2095.4

## Protocol

1. Intervene only on the forward pass that scores article `b`.
2. Force the native baseline article token into the string.
3. Generate content `c` with interventions **off**.
4. Companion: free generation with intervention left on (packager check).

Feature mapping: `S3` ≈ content concept \(C\); `S2` ≈ article/licensing \(B\); `S1` = dual-effect (Latent-Planning-style joint set). No pure \(A\) set.

## Condition summaries

| Condition | Δ(an−a) | Pref. article changed | Content preserved (force) | Content preserved (free) | Trajectory-like (free) | Wrapper-logit+force | Illicit (free) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| S1_dual_effect_amplify | 3.312 | 0.95 | 1.00 | 0.05 | 0.95 | 0.95 | 0.00 |
| S1_dual_effect_zero | -1.031 | 0.05 | 1.00 | 0.95 | 0.00 | 0.00 | 0.00 |
| S2_article_only_amplify | 1.031 | 0.40 | 1.00 | 0.35 | 0.05 | 0.40 | 0.00 |
| S2_article_only_zero | -0.944 | 0.05 | 1.00 | 0.95 | 0.00 | 0.00 | 0.00 |
| S3_content_only_amplify | 0.113 | 0.05 | 1.00 | 0.95 | 0.05 | 0.05 | 0.00 |
| S3_content_only_zero | -0.075 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| control_amplify | 0.138 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| control_zero | -0.056 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

## Interpretation

S3 amplify: Δ(an−a)=0.113, force content preserve=1.00, free trajectory=0.05, wrapper_logit_force=0.05. S3 zero: Δ(an−a)=-0.075, force content preserve=1.00, free trajectory=0.00, wrapper_logit_force=0.00. S1 amplify: free trajectory=0.95, force content preserve=1.00, wrapper_logit_force=0.95, illicit free=0.00. S2 amplify: Δ(an−a)=1.031, force preserve=1.00. Packager-consistent: free generation class-switches, while forcing the native article restores baseline content. No clean modular C→B wrapper.

## Example rows (S1 amplify)

- `Someone who flies airplanes is` baseline `a pilot.` | force `a pilot.` | free `an aviator.` | Δ(an−a)=2.875
- `Someone who represents clients in legal matters is` baseline `a lawyer.` | force `a lawyer.` | free `an attorney.` | Δ(an−a)=3.250
- `Someone who studies matter and energy is` baseline `a physicist.` | force `a physicist.` | free `an astronomer.` | Δ(an−a)=3.250
- `Someone who takes professional pictures is` baseline `a photographer.` | force `a photographer.` | free `an artist.` | Δ(an−a)=3.375
- `Someone who studies human behavior and mental processes is` baseline `a psychologist.` | force `a psychologist.` | free `an anthropologist.` | Δ(an−a)=3.875
- `Someone who studies rocks and earth formations is` baseline `a geologist.` | force `a geologist.` | free `an archaeologist.` | Δ(an−a)=3.375
- `Someone who grows crops and raises livestock is` baseline `a farmer.` | force `a farmer.` | free `an agriculturist.` | Δ(an−a)=3.375
- `Someone who educates children in schools is` baseline `a teacher.` | force `a teacher.` | free `an educator.` | Δ(an−a)=2.875
