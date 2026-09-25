# Research record: maintenance principles

Goal: anyone (or any agent) with only this repository can reconstruct every
result, how it was obtained, and what we concluded from it, without the
conversation that produced it.

## 1. Every experiment is self-describing

Each `experiments/<name>/` has a `README.md` that holds, in this order:

1. **Question** and the terms used (see `docs/glossary/route-accounting-glossary.html`).
2. **Design, frozen before any result**, with the freeze date. Materials, selection
   rules, conditions, estimands, prespecified predictions and gates.
3. **Deviations**, each dated and marked as decided before or after seeing results.
4. **Results**, dated, per model, with intervals and the key reading in one or two
   sentences. Include nulls, failures and stop decisions.
5. **How to run** (exact commands).

Summary tables in a README should be generated from committed JSON (or say which
file they come from), so they can be regenerated.

## 2. What to commit and what to regenerate

- **Commit:** scripts, configs, per-item rows (`*_rows.json`/`.jsonl`), summaries
  (`*_summary.json`), selections, and small tensors needed to reproduce an
  intervention exactly (for example selected decoder rows).
- **Do not commit** re-derivable intermediates: feature cards, safetensors headers,
  captured activations (`*mlp_in*.pt`), model or transcoder caches. They are
  gitignored; the script that makes them must skip finished work and re-create
  missing pieces.
- **No untracked results.** After a run, results are either committed or deleted.
  `git status` should show no stray result files at the end of a working session.

## 3. Remote runs

Results computed on the Mac Studio are fetched into the repository and committed
on the same day, in the experiment's `results/` folder, with the README noting
which machine produced them. Intermediates stay on the Mac Studio. See
`skills/mac-studio-remote/`.

## 4. Branches

- One branch per workstream (for example `couplets-routes`), branched from `main`.
- Commit at every result; push after every commit.
- **Merge into `main` at each milestone** (a completed model size, a completed
  experiment, or before any manuscript work), so `main` always reflects the
  current state. Delete merged branches locally and on GitHub.
- Frozen submissions are protected by tags and `submissions/`, not by branches.

## 5. State of the project

- `docs/handoff/` holds dated snapshots of the overall state: thesis, what each
  experiment established, environment facts, and next steps with gates. Write a
  new dated snapshot at each milestone and mark the previous one as superseded.
- `docs/glossary/route-accounting-glossary.html` is the vocabulary. Update term
  statuses (published, found, hinted, hypothesis, retracted) when results change,
  and republish the shared page.
- Negative and superseded results stay in their experiment READMEs and in the
  manuscript's attempt ledger; nothing is deleted to tidy the story.

## 6. Commit messages

State the result, not just the action: model, key numbers with intervals, and
the reading. A reader of `git log` should be able to follow the findings.
