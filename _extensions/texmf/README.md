# Repo-Local TeX Runtime Files

This directory is added to `TEXINPUTS` by `bin/render-paper`.

It contains small LaTeX runtime files missing from the local BasicTeX
installation but required by vendored venue styles.

- `environ/environ.sty`: generated from CTAN `environ` package source.
- `trimspaces/trimspaces.sty`: extracted from CTAN `trimspaces` package source,
  which `environ` requires.
