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

## Pilot v1 results: Gemma 3 - 27B - relay distance - pilot (2026-09-26)

| | D = 0 (20 couplets) | D = 2000 (5 couplets) |
|---|---|---|
| line 2 rhymes with line 1 | 100% | 40% |
| persistence | +43.11 [36.18, 48.78] | +21.27 [9.82, 30.61] |
| direct retrieval | +30.89 | +7.40 |
| **relay (all in-between positions)** | **+0.65 [0.00, 1.89]** (share 2%) | **+13.59 [5.23, 21.26]** (share 66% [40, 93]) |
| relay through the filler only | 0.00 | +0.38 [0.26, 0.49] |

By the frozen number the pilot is a "go" (share >= 25%, CI above zero), but it is not
decisive: at D = 2000 only 5 of 20 couplets produced usable rows (15 skipped silently
because line 2 was empty or one word; one of the 5 continued the filler), and the
filler itself carries almost nothing, so the relay sits either in the boundary tokens
after line 1 (known boundary storage, read by the global layers) or in the line-2
positions before the rhyme word (relay through generated text, the headline-changing
case). Pilot v2 separates them.

## Pilot v2 (design frozen 2026-09-26, before any v2 result)

Same couplets and distances (D = 0, 2000) with a fixed cue after the filler ("Next
line:") at both distances; 24 couplets; skips logged with the generated text; relay split
into three groups: boundary (prompt positions after the anchor), filler, and cue plus
line-2 words. Same stop/go number, judged on relay through the cue plus line-2 words
(relay through generated text). Qwen3 - 14B - relay distance - pilot v2 as the control.

Control v1: Qwen3 - 14B - relay distance - pilot (20 couplets): D = 0 relay +2.87 [1.98,
3.76] (share 12%); D = 2000 relay +0.29 [0.03, 0.62] (share 3%) but **line 2 rhymes in
0 of 20** (the model abandons the task after the filler), so the D = 2000 control is not
interpretable. The v1 contrast (Gemma relay large at distance, Qwen small) fits the
sliding-window hypothesis but needs v2 on both sides.

### Qwen3 - 14B - relay distance - pilot v2 (control; 24 couplets per distance)

| | D = 0 | D = 2000 |
|---|---|---|
| line 2 rhymes | 59% | 8% |
| persistence | +23.60 | +8.02 |
| direct retrieval | +20.34 | +5.40 |
| relay (absolute) | +2.31 [1.49, 3.12] | +2.77 [1.73, 3.97] |
| relay share | 13% | 32% [23, 40] |
| relay via boundary / filler / line 2 | +0.92 / 0.00 / +1.60 | +1.13 / +0.12 / +1.76 |

In the full-attention control, **absolute relay does not grow with distance**; the share
rises only because direct retrieval weakens over the long context. Reading rule added
before the Gemma v2 result (clarifying, not changing, the frozen criterion): a relay
share rise counts as relay emerging only if **absolute** relay also grows with distance;
otherwise it is a smaller denominator. At D = 2000 Qwen3 14B rarely rhymes (8%): the plan
largely fades after long filler.
