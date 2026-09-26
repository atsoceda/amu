# Relay at a distance: does relay appear when attention cannot reach the source? (pilot)

Pilot design frozen 2026-09-26, before any result.

## Question (H1, distance form)

Relay through generated text is about zero in every model and task so far, and in a
recurrent hybrid (Qwen3.5) the plan still travels by attention. In all of those, some
attention layer can always read the source directly. Gemma 3's layers are 5 sliding
(1,024-token window) to 1 full attention: beyond 1,024 tokens, five of six layers can no
longer see the source. Qwen3 is full attention everywhere (control). If relay is used
only when direct access is impossible, it should appear in Gemma at long distance.

## Pilot

- Couplets, chat prompt, the existing subsets (Gemma 3 27B IT: `results/gemma-3-27b-it`;
  Qwen3-14B: the authors' subset), first 20 couplets.
- Distance: the assistant turn is prefilled with a neutral filler of D tokens (seeded
  arrangement of fixed, answer-neutral sentences) and a blank line, then line 2.
  D = 0 and D = 2000 (beyond the 1,024-token window). The unedited line 2 is the model's
  greedy line after the filler.
- Cells (original line-2 words fixed; R = log mass on donor rhymes minus original rhymes
  at the rhyme position): persistence (donor anchor state, all layers); direct
  retrieval (every in-between position clean); relay (anchor clean, every in-between
  position with its edited-run state); filler relay (only the filler positions edited).
  Replacements are contiguous-slice writes.
- **Stop/go rule (fixed in advance):** go (full study: more couplets and distances,
  hidden choice, Gemma 12B) if Gemma 3 27B's relay share of persistence at D = 2000 is
  at least 25% with its 95% CI above zero; stop if below 10%; in between, extend to 50
  couplets. Qwen3-14B is the control (predicted flat).

**Note (2026-09-26, after the first rows, before any summary):** at D = 2000 the Qwen3-14B
control sometimes abandons the task (line 2 starts "Okay, I need to write…" or repeats
line 1), which makes its rhyme measure uninterpretable for those couplets. The frozen
design had no behavioural gate; results will be reported for all couplets and, as a
post-hoc filter, for couplets whose line 2 rhymes with line 1 (`line2_rhymes_orig`).
