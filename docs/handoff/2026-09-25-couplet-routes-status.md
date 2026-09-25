# Handoff: couplet routes, current thesis, and next steps (2026-09-25, evening)

Supersedes `2026-09-25-routes-thesis-and-next-steps.md`. Read this first when
resuming. Maintenance rules: `docs/research-record.md`. Vocabulary:
`docs/glossary/route-accounting-glossary.html`.

## 1. Thesis (current wording)

Future-relevant information crosses positions by **emission** (a mediator token,
scaled by leverage) or by **retrieval** (a later position reading an earlier
position or the visible text). **Relay**, carrying information privately through
the positions in between, is small wherever we have measured it. Retrieval is
weighted toward structural anchors but not restricted to them.

## 2. Evidence, by experiment (each has `results/report.md`)

| Experiment | Established | Status |
|---|---|---|
| `qwen3_planning_six_cell` (a/an) | Published planning features at Qwen3-4B act on the planned word about 92-94% by emission; emission scales with leverage; persistence small and constant | Found |
| same, el/la | Null, matching the authors' published null; uninformative for routes | Found null |
| `ab_qwen_gate`, `ab_qwen_route` | In-context code becomes a mediator with scale; emission follows leverage; a state-held decision with no leveraged mediator is lost | Found |
| `matched_triads_construction` | Earlier "mediator-relative routing" was a vector-construction artifact | Found; old claim retracted |
| `couplet_routes` | Rhyme plan: persistence dominates; about 90-96% of it is direct retrieval of the line-1 anchor, relay small (median share 4% / 8% / 8% at 1.7B / 4B / 8B); same under the authors' feature steering at 4B; anchor about 2x any other position | Found (14B pending) |

The authors' released attention intervention (blocking anchor-reading heads)
breaks rhyming at 1.7B-14B: retrieval is necessary up to 14B.

## 3. What is running or pending

- Mac Studio job `couplets-8b-14b-v2`: 8B step 6 (feature steering), then 14B
  (steps 2-4, anchor specificity, null control, step 6). Fetch results with
  `skills/mac-studio-remote/scripts/fetch.sh experiments/couplet_routes/results/<model>`,
  then run `make_report.py`, document, commit.
- Local: 4B anchor specificity.

## 4. Next steps

1. **Relay-tail analysis** (existing data): 5-9% of couplets have relay share above
   30%, and a few have relay above retrieval. Find what distinguishes them.
2. **Derived-value carry task** (new): information that is computed, not visible at
   any token, carried across emitted filler before use. The most likely place for
   relay to matter, and the case relevant to chain-of-thought monitoring.
3. Rewrite the manuscript around the thesis (the live `paper.qmd` still presents
   the retracted routing claim).

## 5. Environment

- Laptop: M1, 16 GB. Mac Studio: M4 Max, 128 GB, 1.4 TB free, about 69 MB/s to
  Hugging Face; venv at `~/amu_jobs/.venv` (user-approved). Use
  `skills/mac-studio-remote/` (allowlisted bundle only; never the codebase).
- Transcoders are streamed (`skills/hf-range-streaming/`): resolve once, CDN range
  requests, concurrent rows and 8-way tensor fetches.
- Branch for this workstream: `couplets-routes`, merged into `main` at this
  milestone.
