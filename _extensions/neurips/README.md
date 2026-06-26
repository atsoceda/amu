# NeurIPS 2026 Extension Assets

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
