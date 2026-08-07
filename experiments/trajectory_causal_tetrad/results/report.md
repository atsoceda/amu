# Trajectory Causal Tetrad (E4)

Generated: 2026-08-07T05:41:39.341132+00:00
Model: `google/gemma-3-270m`
Feature set: `S1_dual_effect`
Amplify factor: 5.0

GoF on the frozen set moves twin/class packages above matched controls; supports causal role for trajectory-class features.

## pilot_aviator

| Condition | Baseline | Intervention | Package | Twin? | Class shift? | Δ(an−a) |
| --- | --- | --- | --- | --- | --- | ---: |
| baseline | ` a pilot.` | ` a pilot.` | baseline | False | False | 0.000 |
| lof_zero | ` a pilot.` | ` a pilot.` | baseline | False | False | -0.875 |
| gof_amplify | ` a pilot.` | ` an aviator.` | twin | True | True | 2.875 |
| rescue_amplify | ` a pilot.` | ` an aviator.` | twin | True | True | 2.875 |
| control_amplify | ` a pilot.` | ` a pilot.` | baseline | False | False | 0.125 |

## lawyer_attorney

| Condition | Baseline | Intervention | Package | Twin? | Class shift? | Δ(an−a) |
| --- | --- | --- | --- | --- | --- | ---: |
| baseline | ` a lawyer.` | ` a lawyer.` | baseline | False | False | 0.000 |
| lof_zero | ` a lawyer.` | ` a lawyer.` | baseline | False | False | -1.125 |
| gof_amplify | ` a lawyer.` | ` an attorney.` | twin | True | True | 3.250 |
| rescue_amplify | ` a lawyer.` | ` an attorney.` | twin | True | True | 3.250 |
| control_amplify | ` a lawyer.` | ` a lawyer.` | baseline | False | False | 0.250 |
