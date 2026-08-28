# Boundary precision and stochastic-policy audit

Audits the S1 article boundary under the native bfloat16 path and a float32
final normalization/unembedding recomputation. It also evaluates dense local
gain grids and decomposes an exact conditional `a`/`an` stochastic article
policy at four temperatures into policy-mediated and fixed-token terms.

Run from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/boundary_precision_policy/run.py
/Users/anthony/miniconda3/bin/python experiments/boundary_precision_policy/plot.py
```

Outputs follow the repository convention under `results/`.
