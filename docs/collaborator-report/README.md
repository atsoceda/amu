# Collaborator report build

## Source of truth

- Narrative: [`research-narrative.md`](research-narrative.md)
- Existing figures 1–3: [`figures/figure1_article_recall.png`](figures/figure1_article_recall.png), [`figure2_source_intervention.png`](figures/figure2_source_intervention.png), [`figure3_generalization.png`](figures/figure3_generalization.png)
- New figures 4–7: regenerate from experiment summaries with [`generate_figures.py`](generate_figures.py)
- Style reference DOCX (manual collaborator edits): `From Stalling to Coordinated Preparation–Content Control in Gemma 3 270M — Revised for Collaborators.docx`

## Rebuild

```bash
cd /Users/anthony/repos/amu/docs/collaborator-report
/Users/anthony/miniconda3/bin/python generate_figures.py
/Users/anthony/miniconda3/bin/python build_docx.py
```

Output: `collaborator-report-v3.docx` (and a convenience copy under `build/`, which is gitignored)

The build uses Pandoc so `$$...$$` equations become Word math (`<m:oMath>`), and embeds figures from `figures/`.
