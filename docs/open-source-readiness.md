# Open-source readiness

The version index in the root README makes the papers readable and rebuildable
without navigating research branches. A public replication release additionally
needs the following checks; this page is a checklist, not a claim they are done.

- [ ] Choose a repository license and verify the licenses of redistributed
      figures, datasets, style files, and other third-party assets.
- [ ] Test `bin/check-paper-env` and all three `bin/build-paper-version` commands
      from a fresh clone on a documented system, not just the development Mac.
- [ ] Record a pinned, redistributable Quarto/TeX build environment. Exact PDF
      byte reproducibility additionally requires deterministic TeX timestamps.
- [ ] Audit tracked files and Git history for personal paths, credentials,
      private prompts or raw external-service payloads before public release.
- [ ] Give each headline experiment a small start-here command, dependency list,
      expected runtime/memory, model-license requirements, and a test that
      recomputes published numbers from committed rows without downloading
      weights. See `docs/research-record.md`.
- [ ] Verify figure regeneration and frozen-result recomputation from a clean
      environment; distinguish these from the heavier model-inference runs.
- [ ] Decide whether an annotated GitHub Release should package each archived
      paper. Tags and archive folders already preserve the versions; releases
      are optional distribution conveniences, not additional sources of truth.
