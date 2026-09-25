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
