# Derived-value carry: is computed information relayed?

Design drafted 2026-09-25, before any model run. It will be frozen (dated) after
the behavioural gate (stage 0) and before any route measurement.

## Question

In couplets the carried information (the rhyme) is a property of a **visible
word**, and the model retrieves it from that word's position. Relay should matter
most when the information is **derived**: computed from the context, not present
at any single token. Does a computed value reach a later answer by retrieval of
the position where it was computed, by relay through the tokens in between, or
is it simply recomputed from the visible text at the answer (re-reading)? This is
also the case relevant to chain-of-thought monitoring: a conclusion reached and
then carried under text that does not state it.

## Task

Chat prompt (Qwen3, `/no_think`) posing a two-digit addition, with an
instruction to write a fixed filler sentence before the answer:

    What is 17 + 25? First write "<filler>", then write only the number.

The assistant turn is forced to contain the filler; the target is the first digit
of the answer, predicted right after the filler. Filler lengths: 0, 8, 16 and 32
tokens (fixed, answer-neutral sentences).

Items: sums whose original and donor answers differ in the first digit (for
example 17 + 25 = 42, donor 31 + 46 = 77), balanced over first digits. About 100
original/donor pairs, drawn before any model run with a fixed seed.

## Interventions

- **State edit at the computation position** (primary): every layer's state at
  the last token of the question (where the sum is presumably computed) is
  replaced by the donor question's state at its last token. Text unchanged.
- **State edit at the operand positions** (check): donor states at the operand
  tokens instead.

## Measurements

Outcome: log p(donor first digit) - log p(original first digit) at the target.
For each filler length: persistence with the filler fixed (the filler is forced
text, so there is no emission route), split into **direct retrieval** (the target
reads the edited position; the filler positions get clean states), **relay**
(edited position clean; the filler positions get their edited-run states) and the
remainder. **Recomputation / re-reading** shows up as the edit failing to move
the answer: the model reads the visible operands and computes the original sum.

## Gates and predictions

- **Stage 0, behavioural gate:** with no edit, the answer is correct in at least
  80% of items at every filler length, per model. Models failing are dropped.
- **Stage 1, efficacy gate:** with filler length 0, the computation-position edit
  must move the answer toward the donor (mean outcome shift clearly above zero).
  Otherwise that position does not hold the computed value.
- **Predictions under the thesis:** the answer is mostly recomputed or retrieved;
  relay small, and not growing with filler length.
- **The novel alternative:** relay becomes a substantial share of persistence as
  the filler grows, or grows with model size.

## Sizes and compute

Qwen3-1.7B and 4B on the laptop; 8B and 14B on the Mac Studio
(`skills/mac-studio-remote/`). Short prompts: minutes per size locally.

## Results

### Stage 0/1, Qwen3-1.7B and 4B (2026-09-26, laptop)

Format fixed at stage 0 (before any route measurement): user turn
`/no_think What is {a} + {b}?`, assistant prefill `{filler} The answer is `. Left
free, the model restates the operands ("17 + 25 = 42"), which would put them back in
the text right before the answer. 104 items (13 per tens digit).

| | 1.7B | 4B |
|---|---|---|
| Accuracy, filler 0 / 8 / 16 / 32 tokens | 100 / 95 / 100 / 99% | 100 / 100 / 100 / 100% |
| Donor state at `?` (pre-registered computation position) | +0.07 [-0.01, 0.14] | +0.01 [-0.11, 0.12] |
| Donor states at every position from `?` to the target | +0.30 [0.14, 0.46] | +0.58 [0.34, 0.83] |
| Donor states at the four operand digits | +47.8 [46.4, 49.1] | +48.0 [46.9, 49.1] |

Stage 0 passes; **stage 1 fails**: no position after the question holds the sum;
only changing the operands moves the answer. The sum is recomputed from the visible
operands at the answer (re-reading), as predicted under the thesis. By the frozen
design, routes are not measured where stage 1 fails. 8B, 14B, 32B on the Mac Studio.

## Stage 2 redesign: variable chains, a relay positive control (design frozen 2026-09-26, before any result)

**Why.** Two-digit addition is recomputed from the visible operands at the answer
(stage 1 fails at 1.7B-8B), so it cannot show relay. A chain of updates makes
recomputation at the answer costly: to answer, the model must either re-execute the
whole chain in one forward pass or carry the running value from statement to
statement (relay by construction). The chain is therefore a positive control for
the route accounting: if relay exists anywhere in these models, it should be here,
and a method that never finds relay would be suspect.

**Task.** Chat prompt (Qwen3, `/no_think`), user turn
`a = {v0}\nb = a + {d1}\nc = b + {d2}\n...\nWhat is {last}?`, assistant prefill
`The answer is `; target = first digit of the final value. Chain lengths K = 1, 3, 5
updates. v0 two-digit (10-59), increments 1-9, final value 10-99. Donor: same
increments, a different two-digit v0 whose final value has a different tens digit,
so original and donor prompts align token by token. 96 items per K (seed 20260925).

**Measurements** (outcome R = log p(donor first digit) - log p(original first digit)
at the target, relative to no edit; the answer is immediate, so there is no emission):
- Stage 0: greedy accuracy per K (gate >= 80%).
- Efficacy scan: donor state (all layers) at each single position from v0 to the
  token before the target; plus v0 digits together (the operand edit).
- Route split for an edit at each statement end s_j (the newline ending statement j):
  persistence; direct retrieval (positions after s_j clean, so only the target reads
  s_j); relay (s_j clean, positions after s_j with edited-run states); and chain relay
  (only the later statement ends s_{j+1..K} with edited-run states).

**Predictions.** Under the thesis, statement-end edits do little and the answer is
re-read from v0 and the increments (as in stage 1). The positive-control outcome:
for K = 3-5, an edit at an early statement end moves the answer and most of that
effect goes through later statement ends (chain relay CI above zero and at least 25%
of persistence). Chain relay growing with K and with model size would be relay in
exactly the case the thesis says it should appear: information that is computed, not
visible, and costly to recompute.

### Stage 0/1 across Qwen3 1.7B-32B (2026-09-26)

| | 1.7B | 4B | 8B | 14B | 32B |
|---|---|---|---|---|---|
| Accuracy (all filler lengths) | 95-100% | 100% | 100% | 100% | 100% |
| Donor state at `?` | +0.07 | +0.01 | -0.08 | -0.06 | -0.03 |
| Positions `?` .. target together | +0.30 | +0.58 | -0.12 | +1.62 | +1.10 |
| Operand digits | +47.8 | +48.0 | +53.6 | +47.3 | +17.2 |
| Post-question block / operands | 0.6% | 1.2% | -0.2% | 3.4% | 6.4% |

Stage 1 fails at every size: the sum is recomputed from the visible operands at the
answer. The post-question positions hold a small, slightly growing fraction of it at
14B-32B. The variable-chain redesign (stage 2) is running on the Mac Studio.

**Addition (2026-09-26, after the 8B K=1/K=3 rows, before any 14B/32B run):** a
post-v0 block edit (donor states at every position after v0's digits, up to the
token before the target). Single-position edits cannot detect a running value that
is carried redundantly across many downstream positions; this block edit can. Run
at every size (8B via `--block-only`).

### Stage 2 (variable chains), Qwen3-8B (2026-09-26, Mac Studio)

Accuracy K=1 100%, K=3 63%, K=5 see `results/Qwen3-8B/chain_summary.json` (gate
80%: only K=1 passes). Editing the starting value's digits moves the answer (+45 at
K=1, +31 at K=3); donor states at any statement end, or at any single later
position (the full scan), do nothing (all within +/-0.2). No running value is held
downstream; the answer re-reads the chain. Post-v0 block edit queued (`--block-only`).

Post-v0 block edit at 8B (donor states at every position after v0's digits): +0.05
[-0.12, 0.21] / +0.15 [-0.25, 0.54] / -0.07 [-0.36, 0.22] at K = 1 / 3 / 5, while the
v0 digits alone give +45.1 / +31.1 / +18.4. Nothing downstream holds the running
value, even redundantly: the answer position reads v0 and the increments and
recomputes, and accuracy falls with chain length (100 / 63 / 17%).

Chain task, Qwen3-1.7B and 4B (Mac Studio): accuracy 97 / 17 / 3% and 100 / 55 / 1% at
K = 1 / 3 / 5; post-v0 block -0.02 / +0.11 / +0.04 and -0.07 / +0.29 / -0.20 (all CIs
include zero) against v0-digit edits of +36 / +24 / +14 and +47 / +32 / +20. Same as
8B: nothing downstream holds the running value.

## Stage 3: written chains, public vs private (design frozen 2026-09-26, before any result)

**Why.** Unwritten derived values are recomputed from the visible sources (stages 1-2).
The complementary case matters for chain-of-thought monitoring: when the model
writes its intermediate values, does the final answer follow that visible text
(emission) or private state (persistence)?

**Task.** The variable chains of stage 2 (K = 3, 5), user turn ending "Compute each
variable in order, one per line, then state the value of {last}." (Qwen3 chat,
thinking off). The model's greedy continuation is the written chain (CoT) followed
by the answer; the target is the first digit of the final value, read at the
position where the answer begins (prefix: the written chain plus "The answer is ").
Items where the unedited written chain is fully correct only.

**Edit and cells.** Donor state (all layers) at the v0 digits (the donor's chain has a
different final tens digit). CoT text generated without the edit (text0) and with it
(text1). Cells as in the six-cell assay: R(off, text0), R(on, text1), R(off, text1),
R(on, text0). Emission = R(off, text1) - R(off, text0); persistence = R(on, text1) -
R(off, text1) (and with text0). R = log p(donor first digit) - log p(original first
digit).

**Prediction.** When intermediate values are written, emission carries nearly all of
the effect (the answer reads the last written value); persistence is small. A large
persistence would mean the answer is computed privately despite the visible chain,
the unfaithful-CoT case.

Implementation note (2026-09-26, before results): the model writes each step as
`b = a + 4 = 23 + 4 = 27` (LaTeX). The written text is cut after the line computing
the next-to-last variable and followed by "\nThe answer is ", so one addition from
the last written value remains. Deviation from the frozen text: an item is usable
when the written next-to-last value is correct (not every line checked). Laptop: 4B,
50 items per K; Mac Studio: 8B, 14B, 32B, all items.

Chain task, Qwen3-14B: accuracy 100 / 61 / 8% at K = 1 / 3 / 5; post-v0 block +0.09 /
+0.44 [0.08, 0.82] / +0.22 vs v0-digit edits +30.9 / +18.5 / +6.4. At most about 2% of
the value is held downstream; recomputation still dominates at 14B.

### Stage 3 (written chains), Qwen3-4B, K = 3 (2026-09-26, laptop; 50 items, all usable)

With the start-digit edit, the model writes the donor's values in 100% of items.
Total +18.02 [15.09, 20.89] = **emission +16.80 [14.35, 19.14]** (93%) + persistence
+1.23 [0.58, 1.85] (with the edited text; +2.55 [2.08, 3.02] with the original text).
When intermediate values are written, the answer mostly follows the written values;
a small part re-reads the edited source digits. K = 5 and 8B-32B pending.

**Batching (2026-09-26).** `written_chain.py` now generates in batches of 16
(left-padded; `couplet_routes/batching.py`). Check on 4 items at 1.7B against the
one-at-a-time path: 7 of 8 generations identical; where the text is identical, every
cell agrees to 0.001; one greedy generation differs (bf16 near-ties under batching).
The 4B and 8B runs used the one-at-a-time path; 14B and 32B use the batched path.

### Stage 3 across sizes (2026-09-26)

| | 4B K=3 | 4B K=5 | 8B K=3 | 8B K=5 | 14B K=3 | 14B K=5 |
|---|---|---|---|---|---|---|
| Usable items | 50/50 | 50/50 | 75/108 | 17/103 | 108/108 | 103/103 |
| Total | +18.0 | +16.3 | +12.9 | +12.2 | +10.0 | +9.6 |
| Emission (share) | +16.8 (93%) | +15.6 (96%) | +12.1 (94%) | +11.5 (94%) | +8.7 (88%) | +8.0 (83%) |
| Persistence, edited text | +1.2 | +0.7 | +0.8 | +0.8 | +1.2 | +1.6 |

When the chain is written out, the answer follows the written values (83-96% emission).
The part that bypasses the text (re-reading the edited source digits) grows modestly
with size (about 4-7% at 4B-8B, 12-17% at 14B). 32B pending.

### Stage 3 at Qwen3-32B, and a parser fix (2026-09-26)

Qwen3 - 32B - derived value - written chains, K = 3: 90/108 usable; total +6.64 [5.50, 7.73]
= **emission +6.58 (99%)** + persistence +0.06 [-0.05, 0.16]. The bypass (4-7% at 4B-8B,
12-17% at 14B) is about 1% at 32B: it does not grow with scale. K = 5 was 0/103 usable
because 32B writes each step in bold (`**b = a + 1 = 11 + 1 = 12**`) and the parser
expected the number at the end of the line. Fixed (markdown characters ignored;
identical parses on the earlier data) and all sizes re-scored from the stored
generations (`--from-rows`, outputs `written_K*_rows_recut.json`, `written_summary_recut.json`).

Qwen3 - 14B - derived value - written chains, re-scored with the fixed parser (Mac Studio,
`--from-rows`): identical to the first scoring (K = 3: 108/108 usable, emission +8.74 of
+9.98, 88%; K = 5: 103/103, emission +8.00 of +9.62, 83%), as expected: 14B does not use
bold markup. Qwen3-8B re-scored: K = 3 108/108 usable (was 75), emission +12.62 of +13.46
(94%); K = 5 103/103 (was 17), +12.15 of +13.15 (92%).

Deadline plan (2026-09-26): Qwen3 - 32B - derived value - variable chains (resume) and the 32B
written-chain re-score at K = 5 were dropped (not run). The variable-chain results cover 1.7B to
14B; written chains at 32B cover K = 3 only.

Qwen3 - 1.7B - derived value - written chains (Mac Studio, 2026-09-26): K = 3 108/108 usable,
emission +13.20 of +14.30 (92%); K = 5 103/103, emission +11.41 of +11.85 (96%).
