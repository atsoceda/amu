# NeurIPS 2026 Extension Assets

> **Frozen use only.** The NeurIPS 2026 workshop submission (v35) is archived in
> `submissions/neurips-2026-workshop-v35/`, and its source is Git tag
> `neurips-2026-workshop-v35`. These assets are kept so that tagged version can
> be rebuilt. The live manuscript now targets ICLR 2027; do not render the
> `neurips` target from it.

This directory contains the curated NeurIPS 2026 LaTeX assets used by the
Quarto publishing profiles.

Source bundle:
`_incoming_templates/Formatting_Instructions_For_NeurIPS_2026/`

Tracked files:

- `neurips_2026.sty`: official NeurIPS 2026 style file.
- `checklist.tex`: official NeurIPS 2026 paper checklist template.
- `header-submission.tex`: Quarto header that loads the submission style.
- `header-camera-ready.tex`: Quarto header that loads the camera-ready style.

The raw `_incoming_templates/` directory is ignored and should be treated as a
local drop zone, not as part of the maintained build surface.

The headers also set `\sfdefault` to Computer Modern Sans for local BasicTeX
compatibility. The official style requests Helvetica (`phv`), but this local
TeX installation does not include the required Helvetica metrics. The official
`neurips_2026.sty` file is not modified.
