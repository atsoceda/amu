# S2 forensic analysis

S2 combines free-generation disruption with a consistent subthreshold fixed-an target signal: its free generations frequently fail the legal article+noun parse, yet all seven pre-specified twins move in the target direction. Because the forced-prefix top noun almost never changes and only one frozen random set is available, S2 is evidence of target-aligned fixed-token influence but is not yet a clean persistence-dominant intervention.

| Handle | Legal free completion | Other-prefix rate | Forced-`an` top-1 changed | Forced-`an` TV | Twin ΔΔ `an` | Positive twins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S2_5x` | 0.35 | 0.65 | 0.05 | 0.101 [0.078, 0.124] | 0.830 [0.545, 1.125] | 1.00 |
| `S1_5x` | 1.00 | 0.00 | 0.00 | 0.028 [0.021, 0.035] | 0.089 [0.027, 0.161] | 0.86 |
| `S1_random_5x` | 0.95 | 0.05 | 0.00 | 0.020 [0.014, 0.026] | 0.071 [0.027, 0.134] | 0.71 |
| `S3_5x` | 1.00 | 0.00 | 0.05 | 0.021 [0.016, 0.025] | 0.018 [-0.027, 0.071] | 0.29 |
| `S4_5x` | 0.65 | 0.35 | 0.00 | 0.021 [0.016, 0.027] | 0.080 [0.036, 0.125] | 0.71 |

## Limitations

- Occupation-vocabulary probability mass is a lower bound computed from stored top-10 tokens.
- This reanalysis cannot recover full-vocabulary entropy because full logit vectors were not stored.
- The comparison random arm contains one frozen four-feature set, not a multi-set empirical null distribution.
