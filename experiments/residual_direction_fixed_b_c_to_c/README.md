# Residual / activation-direction fixed-\(b\) \(C \rightarrow c\) (Experiment 1)

Tests whether Stage XVI/XVII nulls are sparse-dictionary artifacts by
steering a **dense MLP-in residual direction** toward the same-class
noun under the Stage XVI protocol (paste native article; keep the
direction ON at planning position \(P\) during noun prediction).

## Method

For each twin family:

1. Build layer-wise directions at mid/late layers:
   \(\Delta = \mathrm{mlp\_in}(\texttt{Think of a \{target\}. prompt})
   - \mathrm{mlp\_in}(\texttt{Think of a \{source\}. prompt})\)
   at each prompt’s pre-article position; unit-normalize per layer.
2. Add \(\alpha \Delta\) at the original prompt’s position \(P\) on
   `prompt + native article` (and during force-native generation).
3. Compare to matched-norm **random** control directions.
4. Score within-class noun change and same−source noun logits.

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python -m experiments.residual_direction_fixed_b_c_to_c.run
```
