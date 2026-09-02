# S1 planning-feature audit

This artifact-only audit asks how closely the frozen S1 feature set matches the
future-noun planning-feature motivation of Hanna and Ameisen. It does not rerun
the model and does not change feature selection.

Run:

```bash
python experiments/s1_planning_feature_audit/audit.py
```

The audit verifies, feature by feature, that each selected S1 feature was active
at the pre-article position and had positive direct attribution to the intended
future noun on every frozen selection prompt. It separately records the crucial
limitation: S1 was selected jointly for article and future-noun attribution, and
the same four features recur across eight different nouns. The evidence therefore
supports a *future-noun-contributing causal handle*, not a noun-specific semantic
feature selected independently of its route.
