# Semantic Commitment Sprint

This repeatable experiment tests whether Gemma 3 270M contains causal handles that move an ambiguous prompt between semantic commitment and syntactic fallback.

Regenerate results from the repository root:

```bash
/Users/anthony/miniconda3/bin/python experiments/semantic_commitment_sprint/run.py
```

Primary outputs:

- `experiments/semantic_commitment_sprint/results/summary.json`
- `experiments/semantic_commitment_sprint/results/report.md`
- `experiments/semantic_commitment_sprint/results/graphs/*.pt`

The experiment intentionally starts intervention-first. It:

1. Loads Gemma 3 270M and the Gemma Scope 2 cross-layer transcoder.
2. Records baseline next-token logits for a small Paris/France/romance prompt family.
3. Builds attribution graphs with explicit semantic and fallback token targets.
4. Selects top direct contributors to the semantic target and observed fallback target.
5. Runs zero, double, source-patch, and grouped feature interventions.
6. Writes a compact report ranking interventions by movement toward `Paris` vs fallback tokens.

If Hugging Face metadata access is unavailable but the default artifacts are cached, the script maps the cached Gemma Scope 2 weights directly and loads the local Gemma 3 snapshot.
