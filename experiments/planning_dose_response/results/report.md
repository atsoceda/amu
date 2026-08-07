# Planning Dose–Response (E2)

Generated: 2026-08-07T05:29:48.068935+00:00
Model: `google/gemma-3-270m`

## Question

Is there a dose window for content-preserving article movement, or does gain scale as package switching?

Sets swept: S1_dual_effect, S2_article_only, S3_content_only, S4_competing_a
Factors: [1.5, 2.0, 3.0, 5.0, 8.0]

## Results

### S1_dual_effect

| Factor | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Control Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 0.456 | 0.000 | 0.100 | 0.900 | 0.100 | 0.006 |
| 2.0 | 0.938 | 0.000 | 0.500 | 0.500 | 0.500 | 0.031 |
| 3.0 | 1.794 | 0.000 | 0.850 | 0.150 | 0.850 | 0.062 |
| 5.0 | 3.312 | 0.000 | 0.950 | 0.050 | 0.950 | 0.138 |
| 8.0 | 5.306 | 0.000 | 0.950 | 0.050 | 0.950 | 0.200 |

Dose interpretation: Monotone (non-decreasing) trajectory-like rate with dose; no wrapper window.

### S2_article_only

| Factor | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Control Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 0.269 | 0.000 | 0.050 | 0.950 | 0.050 | -0.025 |
| 2.0 | 0.456 | 0.000 | 0.100 | 0.850 | 0.100 | -0.019 |
| 3.0 | 0.706 | 0.000 | 0.150 | 0.600 | 0.150 | -0.062 |
| 5.0 | 1.031 | 0.000 | 0.050 | 0.350 | 0.050 | -0.025 |
| 8.0 | 1.528 | 0.000 | 0.000 | 0.000 | 0.000 | -0.075 |

Dose interpretation: Trajectory-like effects dominate across doses; no wrapper window.

### S3_content_only

| Factor | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Control Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 0.013 | 0.000 | 0.000 | 1.000 | 0.000 | -0.037 |
| 2.0 | 0.037 | 0.000 | 0.000 | 1.000 | 0.000 | -0.056 |
| 3.0 | 0.081 | 0.000 | 0.000 | 1.000 | 0.000 | -0.100 |
| 5.0 | 0.113 | 0.000 | 0.050 | 0.950 | 0.050 | -0.188 |
| 8.0 | 0.113 | 0.000 | 0.050 | 0.950 | 0.050 | -0.338 |

Dose interpretation: Mixed dose pattern; see per-factor table.

### S4_competing_a

| Factor | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Control Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | -0.544 | 0.000 | 0.000 | 0.950 | 0.050 | -0.025 |
| 2.0 | -1.100 | 0.000 | 0.000 | 0.950 | 0.050 | -0.044 |
| 3.0 | -2.312 | 0.000 | 0.000 | 0.850 | 0.050 | -0.069 |
| 5.0 | -5.112 | 0.000 | 0.000 | 0.650 | 0.000 | -0.169 |
| 8.0 | -9.689 | 0.000 | 0.000 | 0.000 | 0.000 | -0.225 |

Dose interpretation: Mixed dose pattern; see per-factor table.

## Overall

No wrapper dose window observed across swept sets; trajectory-like scaling dominates.
