# Closed-loop public gain law

Leave-one-feature-out prediction of public TV.

| Model | tau | Attribution only R2 | + susceptibility R2 | + leverage R2 | Measured margin only R2 | + susceptibility R2 | Full identity R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma_270m | 0.1 | -0.551 | 0.828 | 0.839 | -0.326 | 0.989 | 1.000 |
| gemma_270m | 0.25 | -0.530 | 0.862 | 0.875 | -0.291 | 0.988 | 1.000 |
| gemma_270m | 0.5 | -0.194 | 0.859 | 0.871 | 0.021 | 0.988 | 1.000 |
| gemma_270m | 1.0 | 0.394 | 0.847 | 0.857 | 0.555 | 0.986 | 1.000 |
| gemma_1b | 0.1 | -0.059 | 0.158 | 0.204 | -0.451 | 0.966 | 1.000 |
| gemma_1b | 0.25 | -0.113 | 0.066 | 0.111 | -0.751 | 0.968 | 1.000 |
| gemma_1b | 0.5 | -0.071 | -0.071 | -0.032 | -1.186 | 0.964 | 1.000 |
| gemma_1b | 1.0 | 0.267 | -0.078 | -0.063 | -0.663 | 0.957 | 1.000 |
