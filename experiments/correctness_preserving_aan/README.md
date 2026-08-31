# Correctness-preserving a/an carrier experiment

Tests whether the causal route depends on whether the generated article can
encode a distinction between two semantically valid lexical realizations. The
behavioral screen uses the same underlying occupational definition with
noun-unnamed first-letter lexical constraints. Only pairs that greedily produce
both intended article+noun realizations and have single-token noun readouts are
eligible for the causal assay.

```bash
/Users/anthony/miniconda3/bin/python experiments/correctness_preserving_aan/screen.py
/Users/anthony/miniconda3/bin/python experiments/correctness_preserving_aan/screen.py --model gemma_1b
```

The planned causal phase uses one development-selected natural-state-derived
residual patch procedure across between-article, within-`a`, and within-`an`
pairs. The arithmetic `is/are` experiment is not redefined here.

The primary 1B causal analysis selects one screened pair per independent
semantic family and uses leave-one-family-out layer selection based only on
fixed-target-article lexical efficacy. A frozen layer-14 transfer attempt is
retained under `results/frozen_layer14/`; it fails local efficacy and is not
used for the route claim.

```bash
/Users/anthony/miniconda3/bin/python experiments/correctness_preserving_aan/run.py
/Users/anthony/miniconda3/bin/python experiments/correctness_preserving_aan/analyze.py
/Users/anthony/miniconda3/bin/python experiments/correctness_preserving_aan/plot.py
```
