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
