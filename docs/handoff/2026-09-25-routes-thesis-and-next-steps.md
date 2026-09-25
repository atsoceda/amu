# Handoff: route accounting, current thesis, and next steps (2026-09-25)

> **Superseded** by `2026-09-25-couplet-routes-status.md`.

Read this first when resuming. It records where the ICLR 2027 extension stands,
what each experiment established, and the planned next steps with their gates.
Terms follow the glossary: `docs/glossary/route-accounting-glossary.html` (open it
in a browser; also published as a private artifact,
https://claude.ai/artifact/5JLFDjiBPmn9Duxw6WEc72).

## 1. Status of the paper

- NeurIPS 2026 workshop version (v35) is submitted and frozen (see `AGENTS.md`).
- The live manuscript (`paper.qmd`, `manuscript/sections/`) has the Priority-1
  corrections (S1 single feature, gain fit, triad provenance, attempt ledger) but
  **not** the new framing. It still presents "mediator-relative routing", which is
  now retracted (section 3). Do not rewrite it until the couplets decision in
  section 5 is made.
- ICLR 2027 style is vendored; `bin/render-paper iclr submission` works. The main
  text is about one page over the 9-page limit (do not trim; report to the user).

## 2. Current thesis (hypothesis)

Future-relevant information crosses positions by **emission** (a mediator token
that carries it, scaled by leverage) or by **retrieval** (a later position reading
the source position or the visible text). **Relay**, a plan carried privately
through the positions in between, is rare. The accounting is
`E_total = E_emission + E_persistence`, with `E_emission = policy movement x
leverage` for a binary mediator, and persistence split by attention path into
direct retrieval, relay, and re-reading.

## 3. What each experiment established

| Experiment directory | Result | Status |
|---|---|---|
| `experiments/qwen3_planning_six_cell/` (a/an) | Published Hanna & Ameisen planning nodes, selection reproduced (exact count match 344/338/297 of 349 at 0.6B/1.7B/4B). At 4B, effect on the planned word is about 92-94% emission; persistence small and constant; emission scales with leverage (`leverage_analysis.py`) | Found |
| same, `AMU_TASK=el_la` | Null at 0.6B and 1.7B, consistent with the authors' published null; 4B stopped after 31 prompts. 71% of el/la nodes are English-only (lead) | Found null; lead hinted |
| `experiments/ab_qwen_gate/` | In-context A/B code becomes a mediator with leverage 0 / 0.24 / 0.64 at 0.6B / 1.7B / 4B; at 4B leverage follows reliability (0.64 / 0.37 / -0.20); 4B fails only the 50%-bank support check | Found |
| `experiments/ab_qwen_route/` | 4B, 100% vs 75% banks: emission 0.559 vs about 0 (10/10 units, p = 0.001). State patch: about 100% emission with leverage, effect vanishes without it (lost decision). Text edit: large persistence, presumably re-reading (path not isolated) | Found / hinted |
| `experiments/matched_triads_construction/` | Named-donor steering vectors carry a readout component (about 4% of norm); removing it collapses the published route shifts (0.120 -> 0.018; 0.264 -> 0.057) with target efficacy unchanged | Found; "mediator-relative routing" retracted |
| `experiments/archived_feature_vectors/` | Exact 1B CLT slices kept before deleting the 30 GB cache | Utility |

## 4. Environment facts

- M1, 16 GB, no GPU; project Python `/Users/anthony/miniconda3/bin/python`.
- Cached models: Qwen3-0.6B / 1.7B / 4B, Gemma 3 270M / 1B. The 270M and 1B Gemma
  Scope CLTs are deleted (exact slices archived). About 26 GB disk free.
- Qwen3 transcoders are never stored: use `skills/hf-range-streaming/` (resolve
  once, range requests direct to the CDN, 32 concurrent workers). This fixed a
  Hugging Face rate limit and cut card fetching from 4-5 min to about 30 s per
  layer.
- Speed: about 0.3 s per forward at 0.6B, about 13 s at 4B (full recomputation);
  a KV-cache shortcut was rejected for bf16 numerical drift.
- Re-derivable intermediates (feature cards, headers, `mlp_in_last.pt`) are
  gitignored; `select_features.py` re-creates them and skips finished layers.

## 5. Next steps, in order of information per compute

| # | Step | Compute | Gate |
|---|---|---|---|
| 1 | Read Hanna & Ameisen's released couplet results (`external/model-planning-public/couplets/results/`) for rhyme success by size | seconds | Stop if 4B rarely rhymes |
| 2 | Couplets state edit at 1.7B, about 50 couplets: patch the newline states from a different-rhyme couplet, generate line 2 | 10-20 min | Rhyme must follow the patch; else try 4B (2b), else text-edit fallback |
| 3 | Replay cell: edit on, original line-2 words forced; measure the rhyme distribution | 5 min (1.7B) | Emission vs persistence |
| 4 | Attention-path split on the same couplets (clean keys/values at the anchor for the rhyme position vs for line-2 positions) | about 15 min (1.7B) | Relay vs direct retrieval: the thesis test |
| 4v | Validate the split on the A/B text edit at 1.7B, in parallel | about 10 min | Paths must add up to measured persistence |
| 5 | Scale to all usable couplets at 1.7B and 4B | a few hours | Final numbers |
| 6 | Hanna & Ameisen rhyme-feature steering at 4B (needs streaming) | about 2 h download + 1 h | Only if 3-4 show the interesting pattern |
| 7 | Secondary: steering-vector readout audit; el/la Spanish-vs-English node split | varies | In GPU gaps |

Dropped as uninformative: more agreement tasks (is/are, el/la at more sizes) and
more a/an variants.

## 6. Where to look

- Frozen designs and results: each experiment's `README.md`.
- Commit history on `main` (merged from `priority1-hardening`, 2026-09-25) records
  every design freeze before its result.
- Related papers used in the framing: Hanna & Ameisen 2026 (latent planning,
  couplets in section 5 and Appendix D for el/la); Lindsey et al. 2025 (rhyme
  planning); Wu, Morris & Levine 2024 (pre-caching vs breadcrumbs); Pal et al. 2023
  (Future Lens); Ma & Rui 2026; Shih et al. 2026; Lanham et al. 2023; Turpin et al.
  2023; Pfau et al. 2024; Hao et al. 2024 (Coconut); Korbak et al. 2025 (CoT
  monitorability).
