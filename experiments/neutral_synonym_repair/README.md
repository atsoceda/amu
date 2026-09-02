# Neutral-test synonym repair

Repairs the first-letter confound in `correctness_preserving_aan` by separating
donor construction from evaluation. Source and target donors explicitly name
the desired synonym, while the evaluated prompt contains an unrelated
occupation-format demonstration plus only the shared occupational definition.
The demonstration restores the intended `a/an` mediator support without
specifying either tested synonym, initial, or article class. At layer `l`, the intervention is

`h_neutral[l] + strength * (h_target_donor[l] - h_source_donor[l])`.

Thus neither synonym conflicts with the visible evaluated prompt. The assay
reuses the frozen six between-article and eight within-article semantic
families, leave-one-family-out layer selection, strengths, temperatures, and
conditional `a/an` public/private decomposition from the original experiment.
Direct full-prompt inference is used throughout; no prefix KV cache is used.
Results are atomically checkpointed after each completed family.
On restart, completed assay families are skipped. The earlier bare-neutral
pilot is retained under `results_bare_neutral_invalid_support/` and explicitly
excluded because it failed mediator-support validity.

```bash
/Users/anthony/miniconda3/bin/python experiments/neutral_synonym_repair/run.py
```
