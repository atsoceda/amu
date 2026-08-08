# Per-noun fixed-\(b\) \(C \rightarrow c\) (Experiment 2)

Latent-Planning-style **per-noun** feature selection, evaluated under the
Stage XVI protocol (paste native article; keep content clamps **on** at
the pre-article position while predicting the noun).

## Methods

For each twin family:

1. Attribute article logits (`a`/`an`) on the pre-article prompt.
2. Attribute source and same-class noun tokens on `prompt + native article`.
3. Select features at planning position \(P\):
   - **lp_target:** \(+\mathrm{attr}(\text{target})\), \(|\mathrm{attr}(a/an)|\) small
   - **contrast:** maximize \(\mathrm{attr}(\text{target})-\mathrm{attr}(\text{source})\), same article bound
4. Amplify (and zero) those features under fixed native \(b\); score
   within-class noun change vs content-off.

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python -m experiments.per_noun_fixed_b_c_to_c.run
```

Graphs are built in isolated subprocesses (one attribution at a time).
