# Ophthalmologist Reclassify (E6)

Generated: 2026-08-07T05:42:26.681122+00:00
Model: `google/gemma-3-270m`

Classification: `true_wrapper_repair`

Conditions ['lof_pair', 'gof_s1', 'dual'] look like content-preserving article repair on the ophthalmologist mismatch.

| Condition | Baseline | Intervention | Wrapper repair? | Package coincidence? | Δ(an−a) |
| --- | --- | --- | --- | --- | ---: |
| baseline | ` a doctor.` | ` a doctor.` | False | False | 0.000 |
| lof_pair | ` a doctor.` | ` an ophthalmologist.` | True | False | 1.125 |
| gof_s1 | ` a doctor.` | ` an ophthalmologist.` | True | False | 3.250 |
| content_lock | ` a doctor.` | ` a doctor.` | False | False | 0.250 |
| dual | ` a doctor.` | ` an ophthalmologist.` | True | False | 3.625 |
