# Handoff: scale extension, line-boundary relay, derived value (2026-09-26, overnight)

Supersedes `2026-09-25-couplet-routes-status.md`. Read this first when resuming.
Maintenance rules: `docs/research-record.md`. Vocabulary:
`docs/glossary/route-accounting-glossary.html`. The user left Claude running the
study autonomously overnight, with full use of the Mac Studio.

## 1. Thesis (current wording)

Future-relevant information crosses positions by **emission** (a mediator token,
scaled by leverage) or by **retrieval** (a later position reading an earlier one).
**Relay through the generated text** (carrying information privately along the
positions in between) stays near zero up to Qwen3-32B. What grows with scale is a
second **storage site at the line boundary**: from 14B on, the rhyme plan is also
written to the tokens between line 1 and line 2 and read back from there. This
connects the retrieval account (Hanna & Ameisen 2026) to line-end planning sites
(Lindsey et al. 2025, Claude; Ma & Rui 2026, Gemma-3-27B).

## 2. New evidence tonight (details in `experiments/couplet_routes/README.md` and `results/report.md`)

| Result | Numbers |
|---|---|
| Relay share of persistence rises with scale (criterion 1, frozen) | Median 4 / 8 / 8 / 11 / 13% at Qwen3 1.7B / 4B / 8B / 14B / 32B; slope +6.2 pp per decade of parameters, 95% CI [2.8, 9.6] |
| Long-range carrying through line 2 (criterion 2, frozen) | Early line-2 relay -0.06 / +0.07 / +0.04 / -0.01 / +0.16: fails at every size |
| Boundary relay (not anticipated by the frozen readings) | +0.00 / +0.02 / +0.06 / +1.32 / +1.72: appears between 8B and 14B |
| Late relay (last 3 positions before the rhyme word) | +0.77 / +1.53 / +2.27 / +1.63 / +1.80 |
| Qwen3-32B route split | Retrieval +18.2, relay +3.5 of persistence +22.2; donor rhyme 66%; anchor specificity A +22.2, C +11.5, E +11.0 |
| Qwen3-14B feature steering | Retrieval +19.3, relay +3.2 of persistence +23.1 (authors' intervention) |
| Derived value, stage 0/1 (1.7B, 8B) | Accuracy 95-100% at every filler length; the donor state at the question mark does nothing (+0.07, -0.08); only operands move the answer: the sum is recomputed at the answer |

Related work found tonight (not yet in `manuscript/references.bib`): Maar et al.
ICLR 2026 (arXiv 2601.20164); Jacopin 2026 (arXiv 2609.18440, no newline effect at
0.6-2.6B). Ma & Rui 2026 (arXiv 2605.07984) is already cited.

## 3. Decisions taken and why

1. **Stay with frozen criteria.** Criterion 2 failed, so the headline is not "private
   carrying grows with scale". The boundary component is reported as a finding the
   frozen readings did not anticipate, never relabelled as criterion 2.
2. **New experiment: plain-format couplets** (design frozen in the README before any
   result). Ma & Rui's prompt with an explicit newline boundary, same checkpoints.
   Tests whether the chat-format boundary relay is line-boundary storage and
   cross-validates Ma & Rui's Gemma-3-27B hand-off with route shares.
3. **Same-rhyme null for the position split** at every size, so boundary relay cannot
   be edit disruption.
4. **Derived-value task:** stage 1 fails as anticipated (recomputation). Waiting on
   14B/32B before deciding whether to redesign (for example, a two-hop task where
   recomputation at the answer is harder).

## 4. What is running (Mac Studio jobs; `skills/mac-studio-remote/scripts/status.sh <name>`)

- `derived-qwen`: derived-value stages 0/1 at 8B (done), 14B, 32B.
- `couplets-null-positions`: same-rhyme position splits at 8B, 14B, 32B.
- `gemma-chat`: Gemma 3 12B then 27B IT, chat format, full chain.
- `plain-queue`: plain format, Qwen3 8B, 14B, 32B, then Gemma 3 12B, 27B IT.
- `dl-gemma-pt`: Gemma 3 27B/12B PT downloads (secondary check).
- Jobs gate on free memory, with a lock for models over 50 GB. Laptop: position
  nulls at 1.7B/4B, then plain format at 1.7B/4B.
- The Mac Studio also runs another user's long DeepLabCut job (`ioannaporfyri`).
  Leave it alone.

## 5. Update (later in the night)

- **Boundary storage localized.** At Qwen3-14B (chat) the whole boundary relay
  (+1.32) sits on the comma ending line 1; template tokens carry nothing. At 8B,
  nothing on any boundary token.
- **Plain format (Ma & Rui's prompt) replicates it.** The boundary token (`,\n`)
  carries +0.10 at 8B and +2.56 at 14B (median relay share 11% -> 21%). Frozen
  prediction (3) is supported. Base checkpoint Qwen3-8B-Base matches 8B instruct;
  14B-Base queued (pretraining vs post-training).
- **Gemma 3.** 12B chat: relay share 0%, no boundary storage. 12B plain: boundary
  +0.68. 27B chat: large path interaction (additivity gap +12.6, 29% of
  persistence), a signature of redundant downstream storage that the relay-only
  measure understates. A necessity split (frozen before results) is queued to
  measure it.
- **Derived value.** Two-digit sums are recomputed at the answer at every size
  (1.7B-32B). Variable chains (the relay positive control) at 8B: accuracy 100 /
  63 / 17% at K = 1 / 3 / 5. No position after the starting value holds the running
  value, even as a block (+0.05 / +0.15 / -0.07 vs +45 / +31 / +18 for the start
  digits). 14B/32B queued.
- **Mac Studio.** Free-memory gates caused thrashing (about 170 GB of models on
  128 GB). Replaced by a memory-budget scheduler: `experiments/couplet_routes/jobs/runner.sh`
  (queue `~/amu_jobs/queue.txt`, budget 108 GB), watched by
  `skills/mac-studio-remote/scripts/wait_runner.sh`.
