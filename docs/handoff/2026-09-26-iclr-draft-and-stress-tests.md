# Handoff: ICLR 2027 draft, stress tests, remaining runs (2026-09-26, afternoon)

Supersedes `2026-09-26-overnight-scale-and-boundary.md`. Read this first when resuming.
Rules: `docs/research-record.md`, `AGENTS.md`. Vocabulary: `docs/glossary/route-accounting-glossary.html`.
Naming convention for experiments in prose: "model name & generation - model size -
experiment class - experiment name" (for example "Gemma 3 - 27B - relay distance - window curve").

## 1. Framing (agreed with the user)

Principle-led meta-framing: **"Look it up, don't carry it"**. Unwritten information is read
near the point of use from a fixed prompt position (the source, the visible sources it can be
recomputed from, or a copy stored once at a structural token). It is not carried along the
generated text. Surprises are presented as **stress tests** of the principle (recurrent memory,
distance beyond the attention window, hidden content), each claim at the strength of its
evidence. The results are organised on two axes: route (emission; persistence split into
direct retrieval, relay and re-reading) and observability (visible vs hidden).

## 2. Manuscript (ICLR 2027, live source)

- `paper.qmd` plus `manuscript/sections/01-introduction.qmd` ... `10-appendix.qmd`; title
  and abstract in `manuscript/metadata.yml`. The old workshop section files were removed from
  the live tree (preserved at tag `neurips-2026-workshop-v35`).
- Figures: `manuscript/figures/make_iclr_figures.py` builds `iclr_fig1`-`iclr_fig5` from the
  committed results, drawn at printed size. Re-run after every sync.
- Render: `bin/render-paper iclr submission` -> `dist/iclr-submission/paper.pdf`. 24 pages;
  main text ends on page 16 (limit 9). The user said not to worry about length yet.
- Open items are marked `\pending{...}` in the text (red in the PDF). Skill:
  `skills/iclr-manuscript/SKILL.md`.
- The AI use statement is a draft for the authors to confirm. The appendix mentions a
  non-archival workshop version and its retracted claim (mediator-relative routing); the
  authors should confirm this is compatible with anonymity.

## 3. Results added today (all recorded in the experiment READMEs)

| Experiment | Result |
|---|---|
| Qwen3.5 - 27B - recurrent carry - pilot (24 couplets) | recurrent-only 2% of persistence: stop |
| Gemma 3 - 27B - relay distance - pilot v2 extension (50) | D = 2000: retrieval 40%, line-end copy necessity 49%, generated-text relay 12% (between stop and go) |
| Gemma 3 - 12B - relay distance - pilot v2 | plan lost at D = 2000 (0% rhyme, persistence +0.6) |
| Qwen3.5 - 9B - relay distance - pilot v2 | plan fades (92% -> 4% rhyme); what remains is retrieved; relay flat |
| Hidden choice replication (animals, pre-registered) | Gemma 3 4B/12B/27B 22/18/40%; Qwen3 8B/14B/32B 15/3/7%; Qwen3.5-27B 4% |
| Hidden choice localization | Gemma 3 27B fruits: "the" in "about the weather" 77%; 12B fruits same site; 12B animals same site (73%) |
| Hidden choice forced split, Qwen3-32B | relay through the sentence +0.06 [0.04, 0.09] (2% of text swap): the one nonzero case |
| Qwen3.5 - 27B - hidden choice - out-of-list | pick specificity 3% (family-like, as predicted) |

## 4. Mac Studio queue (rigour-ranked; runner `~/amu_jobs/queue.txt`, watcher `skills/mac-studio-remote/scripts/wait_runner.sh`)

Running at the time of writing: core-qwen35-27b (trimmed core suite: forced route split,
then induction), loc-gemma12-alt. Queued in order: core-gemma4-31b, core-qwen35-35b,
win-gemma27 (window curve D = 500/1000/1500), rc-27b-100 (Qwen3.5-27B recurrent pilot at 100),
null-gemma27 and null-qwen32 (same-pick null), loc-gemma27-animals, loc-gemma27-alt,
recut-qwen32 (written chains K = 5 re-score), chain-32b-resume, gemma27pt-plain-resume.
Each finished result: sync, record in the README with the naming convention, update the
manuscript (remove its `\pending` marker, update numbers everywhere), re-run the figure
script, render, commit and push.

## 5. Known gaps

- Qwen3-4B written chains at K = 3 were generated locally, so the remote re-score failed;
  the committed K = 3 summary (93% emission) stands.
- The laptop must not run new jobs until the user says so. A 9-hour `caffeinate` is running
  locally at the user's request.
- IDRIS GPU access is shelved (administrative signature pending).
