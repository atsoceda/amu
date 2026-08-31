# Matched semantic triads

Paired confirmatory test of mediator-relative distinguishability. Every triad
uses one semantic definition and one source realization, with a same-article
target and a cross-article target. The source prompt, source noun, source hidden
state, layer, strength, and decomposition are therefore paired within family.

The protocol was frozen before screening. Gemma 3 1B PT, layer 18, strength 1,
and temperature 1 are primary. Strengths 0.5/1.5 and other temperatures are
robustness analyses. Signed target-aligned route effects are primary; TV is
secondary.

The class-only direction is estimated from the previously frozen between-class
families in `experiments/correctness_preserving_aan`, not from triad outcomes.

```bash
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads/screen.py
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads/screen.py --config config_batch2.json --output-suffix batch2
```

Batch 1 is the frozen fresh bank. Because it yields only one admissible triad,
batch 2 is a separately frozen discovery bank based on lexical arms that had
independently passed the earlier Draft 24 screen. It is not labeled a fresh
confirmatory holdout.

Batch 3 is an outcome-blind exhaustive discovery bank generated from the
pre-existing synonym lexicon. It uses no new logits or causal outcomes during
construction and permits at most one selected triad per semantic family:

```bash
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads/build_exhaustive_bank.py
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads/screen.py --config config_batch3.json --output-suffix batch3
/Users/anthony/miniconda3/bin/python experiments/matched_semantic_triads/analyze_screening.py
```
