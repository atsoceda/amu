# Handoff: ICLR 2027 submission state (2026-09-26, 19:15 JST)

Supersedes `2026-09-26-iclr-draft-and-stress-tests.md`. Read this first.
Rules: `AGENTS.md`, `docs/research-record.md`. Vocabulary: `docs/glossary/route-accounting-glossary.html`.
Name results as "model name & generation - model size - experiment class - experiment name".

## 1. Where things are

| What | Where |
|---|---|
| Manuscript source (ICLR 2027) | `paper.qmd`, `manuscript/metadata.yml` (title, abstract), `manuscript/sections/01-introduction.qmd` ... `10-appendix.qmd`, `manuscript/references.bib` |
| Figures | `manuscript/figures/make_iclr_figures.py` builds `iclr_fig1`-`iclr_fig5` and appendix `iclr_figA_*` from committed results; drawn at printed size |
| Render | `bin/render-paper iclr submission` -> `dist/iclr-submission/paper.pdf` plus a numbered copy `paper_N.pdf` every render (latest: `paper_12.pdf`; `dist/` is gitignored) |
| Frozen workshop version | `submissions/neurips-2026-workshop-v35/`, git tag `neurips-2026-workshop-v35` (never render the neurips target) |
| Experiments | `experiments/<class>/README.md` (design frozen before results, dated deviations and results) and `experiments/<class>/results/<model>/` (rows and summaries) |
| Skills | `skills/iclr-manuscript`, `skills/state-edit-experiments`, `skills/mac-studio-remote` (scheduler, deadline planning), plus circuit-tracer skills |

Experiment classes: `couplet_routes`, `derived_value_carry`, `hidden_choice`, `relay_positive_control`,
`recurrent_carry`, `relay_distance`, `architecture_generalization` (Qwen3.5 / Gemma 4 core suites),
`qwen3_planning_six_cell` (a/an leverage). Older workshop-era folders remain for the record.

## 2. Paper status

- Title: "Look It Up, Don't Carry It: How Language Models Move Unwritten Information Across Their Own Text".
- 27 pages; main text ~17 pages against the 9-page ICLR limit. The user deferred length; no text has been cut.
- No `\pending` markers remain. Three rounds of agent review were applied (see git log 2026-09-26).
- To confirm by the authors: the AI use statement (`09-statements.qmd`), the appendix note on the earlier
  non-archival workshop version (anonymity), and the two latent-reasoning references
  (`goyal2024pause`, `hao2024coconut`) added without source-page verification.

## 3. Final results of 2026-09-26 (all in READMEs and the paper)

- Gemma 3 - 27B - couplet routes - necessity (plain): stored copy necessary for 74%; on the newline.
- Gemma 3 - 27B - relay distance - window curve: gradual shift to the stored copy, largest step 1,000-1,500 tokens; generated-text block (incl. last 3 positions, not split) 9-13%.
- Gemma 3 - 12B - relay distance - window curve: plan fades inside the window (distance, not window).
- Hidden choice localization: storage moves with the rewording at 4B, 12B and 27B; same-pick nulls hold at 12B and 27B.
- Gemma 4 - 31B - hidden choice: 82% (fruits) / 91% (animals) stored; route split and induction not run.
- Qwen3.5 - 9B and 27B: induction partial passes; hidden-choice relay 1-2%, sentence positions 7-8%.

## 4. Not run (listed in the paper's limitations; commands in `~/amu_jobs/queue.later.txt` on the Mac Studio)

Gemma 4 route split and induction; Qwen3.5-35B-A3B core suite; Qwen3.5-27B recurrent pilot at 100
couplets; Qwen3-32B variable chains and written chains at K = 5; Gemma 3 27B PT plain chain; Gemma 3
27B animal localization; Qwen3-32B same-pick null. Most valuable next analysis: split the 27B distance
generated-text block into early relay and late lookup.

## 5. Mac Studio

Queue empty at 12:05 CEST; runner exited. Everything was synced and committed. Restart the runner per
`skills/mac-studio-remote/SKILL.md` before queuing new jobs.
