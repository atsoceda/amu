# Gemma 1B sparse causal-channel scale test

This experiment repeats the independently selected S1--S4 feature-family assay
on Gemma 3 1B PT with the matched Gemma Scope 2 affine CLT. Selection uses the
same eight frozen occupation prompts as the 270M study. Evaluation is held out.

The first execution milestone is a single isolated attribution graph, used as a
memory-feasibility test on the 16 GB M1 host. If it succeeds, run the complete
selection phase and then the held-out assay.

```bash
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_sparse_scale/run.py --graph-only someone-who-installs-and-repairs-electrical-systems-is article
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_sparse_scale/run.py --selection-only
/Users/anthony/miniconda3/bin/python experiments/gemma_1b_sparse_scale/run.py
```

