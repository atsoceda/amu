# Collaborator report build

Rebuild the full collaborator DOCX from **experiment result summaries** plus the
**polished Markdown writeup**. Checked-in PNGs are outputs, not sources.

## Inputs

| Input | Role |
| --- | --- |
| [`research-narrative.md`](research-narrative.md) | Polished prose (source of truth for text/equations) |
| `experiments/*/results/summary.json` | Raw result tables for figures 1–11 |
| [`styles-reference.docx`](styles-reference.docx) | Word styles only (optional; Pandoc defaults if missing) |

Figure → experiment map (see [`generate_figures.py`](generate_figures.py)):

1. `a_an_majority_baseline`, `a_an_full_dataset_screen`
2. `ophthalmologist_competing_pathway_screen`
3. `fixed_pair_generalization`
4. `selection_criterion_ablation`
5. `planning_dose_response`
6. `forced_content_lock`
7. `trajectory_causal_tetrad`
8. `selective_bc_force_native`
9. `causal_edge_independence`
10. `per_noun_fixed_b_c_to_c`
11. `residual_direction_fixed_b_c_to_c`

## One-command rebuild

```bash
bin/build-collaborator-report
```

Or the two steps:

```bash
cd docs/collaborator-report
/Users/anthony/miniconda3/bin/python generate_figures.py
/Users/anthony/miniconda3/bin/python build_docx.py
```

Output: [`collaborator-report-v3.docx`](collaborator-report-v3.docx) (copy also under `build/`, gitignored).

Pandoc turns `$$...$$` / `\(...\)` into Word math (`<m:oMath>`) and embeds the regenerated figures.
