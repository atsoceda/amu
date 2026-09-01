# Natural modifier-to-noun route crossover

Purpose-built Gemma 3 1B PT benchmark for testing whether target-aligned causal
control crosses from private-dominant through hybrid to public-dominant as a
generated modifier becomes more useful for predicting a later near-synonymous
noun. The neutral evaluated instruction never names a modifier, noun, initial,
or synonym.

The pipeline is restart-safe and preserves every generation and rejection:

1. `generate_candidates.py`: cached Gemini lexical generation.
2. `tokenizer_filter.py`: local one-token and format filtering.
3. `branch_leverage.py`: restart-safe forced-modifier scoring with exact
   per-family prefix-cache reuse.
4. `prepare_context_candidates.py`: freeze low/medium/high constructions.
5. `generate_contexts.py`: cached dev/test contexts plus cross-cell audit.
6. `audit_semantics.py`: independent strict cross-product audit without repair.
7. `assemble_semantic_pass.py`: immutable join of generated and audited items.
8. `policy_probe.py`: development-only natural modifier-policy screen.
9. `prepare_top50_support.py`, `audit_top50_support.py`, and
   `compute_top50_support.py`: expand and validate the actual natural mediator
   support before any decomposition.
10. discovery decomposition on development paraphrases.
11. immutable confirmatory freeze and untouched test paraphrases.
12. held-out decomposition and pre-mediator private-state lesions.

Run candidate generation from a login shell so `GOOGLE_GEMINI_API_KEY` is
available from `~/.zprofile`:

```bash
/Users/anthony/miniconda3/bin/python experiments/natural_modifier_routes/generate_candidates.py
/Users/anthony/miniconda3/bin/python experiments/natural_modifier_routes/tokenizer_filter.py
/Users/anthony/miniconda3/bin/python experiments/natural_modifier_routes/branch_leverage.py
```

Raw API payloads and responses live under `artifacts/00_raw_generation/`.
Parsed candidates and tokenizer decisions are immutable downstream inputs.
