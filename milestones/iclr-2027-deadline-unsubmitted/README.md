# ICLR 2027 deadline draft — not submitted

This folder preserves the last local ICLR-formatted paper at the deadline. It
was **not submitted to ICLR**. The PDF is the exact local numbered build
`dist/iclr-submission/paper_65.pdf`; the generated TeX and original manifest
are preserved alongside it. Do not edit or replace these files.

The source checkpoint is the Git tag `iclr-2027-deadline-unsubmitted` at commit
`ffbc8cdb724e465459e7d74cbca256ec24cd252a`. To rebuild without changing
your working checkout, run from the repository root:

```bash
bin/build-paper-version iclr-2027-deadline-unsubmitted
```

The archived manifest records the **pre-commit render** (`git_dirty: true`,
`git_commit: ed41ea8...`). This is a historical fact, not a claim of a clean
render. The version builder independently checks that rendering the tagged
source reproduces the archived TeX SHA-256. Rebuilt PDF bytes may differ because
TeX embeds build timestamps; the archived `paper.pdf` is the exact local build.

`SHA256SUMS` verifies the three archived artifacts.
