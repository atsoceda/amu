# ICLR 2027 layout assets

The ICLR 2027 deadline draft was **not submitted**. Its last local PDF is frozen
under `milestones/iclr-2027-deadline-unsubmitted/`, with source tag
`iclr-2027-deadline-unsubmitted`. The live paper has no chosen venue; its
internal layout loads these official files without changing them, then
replaces the visible ICLR review claims in `_extensions/working-paper/`.

Official ICLR 2027 LaTeX template files, copied unmodified from
https://github.com/ICLR/Master-Template/tree/master/iclr2027 (downloaded
2026-09-24; checksums in `SHA256SUMS`).

- `iclr2027_conference.sty`, `iclr2027_conference.bst`: style and bibliography style.
- `math_commands.tex`, `fancyhdr.sty`, `natbib.sty`: supporting files shipped with the template.
- `iclr2027_conference.tex`: the template's example paper, kept for reference only.
- `header-submission.tex` / `header-camera-ready.tex`: project-local headers
  included by the `iclr` Quarto profiles. Camera-ready adds `\iclrfinalcopy`.

Do not modify the official files.

Local build notes: the style needs Helvetica (`phv`) and Courier (`pcr`)
metrics that the local TeX Live basic install lacks. The TeX Live `helvetic`
and `courier` packages are unpacked under `_extensions/texmf/`, loaded through
`\pdfmapfile` in the headers, and `bin/render-paper` sets
`TEXMFHOME=_extensions/texmf`.
