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
