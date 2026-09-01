# A/B few-shot mediator preflight

Cheap behavioral gate for the proposed route-dominance crossover. Gemma 3 1B
PT sees the same eight demonstration contexts and terms in all conditions; only
the public `A/B` code labels change:

- `high`: code perfectly predicts common versus formal lexical realization;
- `medium`: code predicts lexical register on 75% of demonstration families;
- `low`: code is balanced and independent of lexical register.

Five held-out semantic families are tested. The evaluated context is neutral and
names neither noun, code, register, initial, nor synonym. Source/target contexts
are used only for the behavioral context-sensitivity preflight. No residual
intervention, layer selection, causal route decomposition, or test confirmation
is run at this stage.

```bash
/Users/anthony/miniconda3/bin/python experiments/ab_fewshot_preflight/run.py
```

The default rerun output is `results_corrected_direct/`. The runner uses direct
full-prompt inference with `use_cache=False` and atomically checkpoints after
each bank-family cell. Re-running the command resumes completed cells. The
historical `results/` directory is retained unchanged for comparison.
