# Fixed-article full-residual reference

Generated: 2026-08-10T10:08:04.569414+00:00
Runtime: 133.1s
Development-selected layer: 12

This is an upstream full-residual reference intervention, not an oracle.

## Development layer sweep

| Layer | Mean target-specific ΔΔ |
| ---: | ---: |
| 0 | 0.000 |
| 1 | 0.016 |
| 2 | 0.031 |
| 3 | 0.031 |
| 4 | 0.000 |
| 5 | 0.031 |
| 6 | 0.047 |
| 7 | 0.016 |
| 8 | 0.078 |
| 9 | 0.047 |
| 10 | 0.344 |
| 11 | 1.531 |
| 12 | 7.938 |
| 13 | 6.062 |
| 14 | 6.156 |
| 15 | 7.719 |
| 16 | 7.719 |
| 17 | 7.438 |

## Held-out test

- Target patch mean ΔΔ: 6.438 [4.188, 10.062] over 4 pairs.
- Random matched-norm mean ΔΔ: 0.143 [-0.327, 0.612] after averaging seeds within each pair.
- Assay sensitivity validated: **True**.

Validation requires a positive held-out target-specific effect exceeding the configured minimum and the mean matched-norm random effect.
