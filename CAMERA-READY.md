# NeurIPS 2026 workshop paper: camera-ready line

This branch, `paper/neurips-workshop`, holds the camera-ready version of the
accepted NeurIPS 2026 workshop paper ("Interpretability as a Science"). It
starts from tag `neurips-2026-workshop-v35`, the exact submitted source, and is
never merged into `main`, which carries a different, venue-undecided paper.

- Render: `bin/render-paper neurips camera-ready` (the only profile allowed
  here) to `dist/neurips-camera-ready/`.
- The submitted paper is frozen: tag `neurips-2026-workshop-v35`, exact PDF in
  `submissions/neurips-2026-workshop-v35/` on `main`. Never edit that archive.
- When the camera-ready is uploaded: tag the final commit
  `neurips-2026-workshop-camera-ready`, archive the exact PDF, TeX, manifest
  and checksums in `submissions/neurips-2026-workshop-camera-ready/` on `main`,
  and add the version to `paper-versions.json` and the root README on `main`.

## Camera-ready to-do

- [x] Use the workshop camera-ready style option (`[dblblindworkshop,final]`
      with the workshop title); the tagged header wrongly used `[main,final]`.
- [ ] Authors and affiliations in `manuscript/metadata.yml` (currently anonymous).
- [ ] Acknowledgements and funding, if the workshop allows them.
- [ ] Decide how to treat claims retracted since submission (mediator-relative
      routing; the S1 description), within the workshop's camera-ready rules.
- [ ] Final answers in the NeurIPS checklist (`_extensions/neurips/checklist.tex`).
- [ ] Check the workshop's camera-ready instructions (page limit, format, deadline).
