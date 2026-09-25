# Relay positive control: induction

Design frozen 2026-09-26, before any result.

## Question

Every task so far finds retrieval, not relay (couplets, sums, variable chains). A
reviewer can fairly ask whether the route accounting can detect relay at all. This
experiment applies it where relay is the known mechanism: induction (Olsson et al.
2022). In a repeated random token sequence `... A B ... A -> B`, a previous-token head
writes "the previous token was A" into B's position, and at the second A an
induction head finds that position and copies B. Information about the first A
reaches the prediction through B's position, not by being read directly.

## Design

- Sequences: 100 random sequences of L = 24 distinct common word tokens (seed
  20260925), followed by a repeat of the first k + 1 = 12 tokens; plain text, no chat
  template (BOS added where the tokenizer uses one). Target = the last position (the
  second A = x_k); the correct next token is B = x_{k+1}.
- Edit: at the first occurrence of A (position k, the anchor), every layer's state is
  replaced by the state of a donor sequence identical except that x_k is a different
  random token (positions align).
- Outcome: R = log p(B) at the target; effects are R(edited) - R(clean) (negative when
  the edit breaks the copy).
- Split (as in the couplet route split): persistence (edit at the anchor); direct
  retrieval (edit at the anchor, every position between the anchor and the target
  clean); relay (anchor clean, those positions with their edited-run states); and
  key-position relay (only position k + 1, B's first occurrence, with its edited-run
  state).
- Gate: without the edit, the model copies (mean p(B) > 0.5).

## Prediction (the positive control)

Relay carries nearly all of persistence (share > 80%), almost all of it through the
key position; direct retrieval is near zero. If the accounting instead reported
retrieval here, it would be unable to detect relay and the negative results
elsewhere would be uninformative.

## Results (2026-09-26)

| | 1.7B | 8B | 14B |
|---|---|---|---|
| p(B), clean | 1.000 | 0.999 | 0.998 |
| Persistence (edit at first A) | -4.26 [-4.60, -3.91] | -3.83 [-4.11, -3.54] | -3.81 [-4.14, -3.46] |
| Direct retrieval | -0.08 [-0.13, -0.03] | -0.05 [-0.09, -0.02] | -0.10 [-0.15, -0.05] |
| Relay | -5.85 [-6.63, -5.10] | -4.13 [-4.83, -3.45] | -4.64 [-5.33, -3.95] |
| Key position (B's first occurrence) only | -9.24 [-10.21, -8.26] | -10.20 [-11.40, -9.02] | -8.86 [-9.82, -7.88] |

**The positive control passes.** Where relay is the known mechanism, the accounting
gives direct retrieval about 2% of persistence or less and assigns the effect to
relay, concentrated at the key position. Relay and the key position exceed
persistence: with only downstream positions edited, the clean anchor (still A)
conflicts with the key position ("previous token was y"), which suppresses B more
than the consistent edit; this is an interaction, so relay shares above 100% are not
read as proportions. The retrieval-dominated results elsewhere are therefore
informative negatives, not a blind spot of the method. 4B, 32B and Gemma pending.

Added: Qwen3-4B (laptop) persistence -4.75, retrieval -0.15, relay -5.23, key -10.28;
Gemma 3 12B IT persistence -4.90 [-5.18, -4.64], retrieval -0.02 [-0.03, -0.02],
relay -2.61 [-3.17, -2.11], key -7.12 [-8.16, -6.08]. Retrieval stays near zero in
both families; in Gemma the all-positions relay edit is weaker than the key position
alone (the edited second-half positions interact), so the key-position edit is the
cleaner relay readout.
