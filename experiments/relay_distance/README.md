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

### Gemma 3 - 27B - relay distance - pilot v2 (24 couplets per distance; 0 skipped)

| | D = 0 | D = 2000 |
|---|---|---|
| line 2 rhymes | 54% | 71% |
| persistence | +43.65 | +28.19 |
| direct retrieval | +29.93 | +12.30 |
| **relay (absolute)** | **+1.44 [0.18, 3.05]** (share 3%) | **+15.49 [11.78, 19.25]** (share 55% [43, 67]) |
| via boundary (prompt positions after line 1) | +0.08 | **+12.08 [8.27, 16.16]** |
| via filler | 0.00 | +0.72 [0.02, 1.40] |
| via cue + line-2 words (generated text) | +0.91 | **+2.91 [1.59, 4.29]** (about 10%) |

Relay emerges at distance in the sliding-window model: absolute relay grows about 10x
(the full-attention control stays flat, +2.3 -> +2.8). It is mostly **boundary storage**:
with five of six layers unable to reach line 1, the plan is read far more from the
tokens just after line 1 than from the rhyme word. Relay through generated text rises
too but modestly (+0.9 -> +2.9, about 10% of persistence). By the frozen v2 rule (judged
on generated-text relay) this is at the stop/extend boundary: extend to 50 couplets.

## Pilot v2 extension (frozen 2026-09-26, before any extension result)

Gemma 3 - 27B - relay distance - pilot v2 extension: 50 couplets, D = 0 and 2000, same
cells, plus necessity for the three groups (edit the anchor; reset one group to clean),
added because Gemma's paths interact (sufficiency understated boundary storage before).
Decision on generated-text relay share (sufficiency, as frozen): go >= 25%, stop < 10%;
necessity reported alongside.

### Gemma 3 - 27B - relay distance - pilot v2 extension (50 couplets per distance; 0 skipped)

| | D = 0 | D = 2000 |
|---|---|---|
| line 2 rhymes | 98% | 82% |
| persistence | +42.43 [38.57, 45.96] | +26.43 [23.77, 28.91] |
| direct retrieval | +28.00 | +10.48 |
| relay (absolute) | +0.98 [0.12, 1.92] | +13.09 [10.70, 15.59] (share 53%) |
| boundary, sufficiency / necessity | +0.06 / +4.56 | +9.75 / **+13.00 [10.00, 16.11]** |
| filler, sufficiency / necessity | 0.00 / 0.00 | +0.31 / +0.15 |
| generated line 2 (cue + words), sufficiency / necessity | +0.45 / +7.34 | +3.20 [2.24, 4.27] / +2.70 [1.78, 3.65] |

Decision on the frozen number (generated-text relay share, sufficiency): mean ratio 12%,
median 10%, between the stop (10%) and go (25%) lines after the extension: generated-text
relay at 2,000 tokens is small but nonzero. The dominant change with distance is the
boundary: by necessity it carries about half of persistence at D = 2000 (13.0 of 26.4),
while the generated line's necessity falls (7.3 -> 2.7; at D = 0 it is the late lookup in
the last positions before the rhyme word). Distance shifts reliance onto the stored copy
at the line boundary, not onto the generated text.

## Window curve (design frozen 2026-09-26, before any result)

Gemma 3 - 27B - relay distance - window curve: the v2 extension cells at D = 500, 1000
and 1500 (first 24 couplets), completing 0 / 500 / 1000 / 1500 / 2000. The sliding-window
account predicts that boundary reliance (necessity) and the loss of direct retrieval
change specifically once the target is more than 1,024 tokens from line 1 (between
D = 1000 and D = 1500), not gradually with distance. Outputs `pilot_v2x_window_*`.

### Gemma 3 - 12B - relay distance - pilot v2 (24 couplets per distance; 1 skipped at D = 2000)

| | D = 0 | D = 2000 |
|---|---|---|
| line 2 rhymes | 100% | **0%** |
| persistence | +40.49 [34.49, 45.86] | **+0.60 [-0.67, 1.84]** |
| direct retrieval | +35.09 | -0.54 [-1.39, 0.19] |
| relay (absolute) | +0.70 [0.08, 1.44] | +1.20 [0.36, 2.14] |
| via boundary / filler / cue + line-2 words | +0.07 / 0.00 / +0.57 | +0.87 / +0.83 / -0.05 [-0.45, 0.36] |

At 12B the rhyme plan does not survive 2,000 tokens of filler: line 2 is still a poem line
("And echoes linger, soft and low.") but never rhymes with line 1, and the donor edit has no
effect on the rhyme position. Nothing takes over when direct access is lost: no relay through
the generated line (-0.05) and almost no boundary copy (Gemma 3 12B stores 1.4% at the line
end in the chat format, against 11% at 27B). Read with the 27B result: at distance the model
keeps the plan only when a stored copy at the line boundary exists; otherwise the plan is
lost rather than relayed. The frozen stop/go number (relay share of a near-zero persistence)
is not interpretable here and is not used.

### Qwen3.5 - 9B - relay distance - pilot v2 (recurrent hybrid; 24 couplets per distance)

| | D = 0 | D = 2000 |
|---|---|---|
| line 2 rhymes | 92% | 4% |
| persistence | +14.83 [12.74, 16.85] | +3.58 [1.99, 5.41] |
| direct retrieval | +13.53 (91%) | +2.90 [1.38, 4.62] (81% of what remains) |
| relay (absolute) | +1.16 [0.64, 1.81] | +0.82 [0.50, 1.20] |
| via boundary / filler / cue + line-2 words | -0.00 / 0.00 / +1.17 | +0.02 / +0.36 / +0.48 [0.18, 0.85] |

In the recurrent hybrid (1 in 4 layers full attention, no local window) the plan mostly fades
over 2,000 tokens of filler (rhyming 92% -> 4%, persistence down by three quarters). What
remains is still read directly from line 1 by the full-attention layers; absolute relay does
not grow (+1.16 -> +0.82), and nothing moves onto the boundary. As in Gemma 3 12B, distance
without a stored copy loses the plan rather than moving it onto a relay path.
