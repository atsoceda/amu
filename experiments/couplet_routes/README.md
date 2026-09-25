# Couplet routes: emission vs retrieval vs relay for rhyme planning

Tests whether a rhyme "plan" at the end of line 1 reaches the end of line 2 by
**emission** (through the line-2 words), **direct retrieval** (the rhyme position
reads the line-1 anchor), or **relay** (carried through the positions in between).
Terms: `docs/glossary/route-accounting-glossary.html`.

Materials: Hanna & Ameisen's released 100-couplet steering subset per model
(`couplets/results/rhyme_intervention_sample/<model>.csv`), their prompt format,
their anchor (last word of line 1, two tokens before the end of the user turn) and
their donor pairing (`chosen_index`, a couplet with a different rhyme group).
Rhymes are scored with Datamuse `rel_rhy`, their perfect-rhyme criterion.

- `step2_state_edit.py`: replace every layer's state at the anchor with the donor
  prompt's anchor states (text unchanged); generate line 2 with and without.
- `step34_routes.py`: target = first token of line 2's last word after the line-2
  prefix. Cells: edit off/on x original/edited prefix; persistence with the
  original prefix split into retrieval-only (in-between positions given clean
  states) and relay-only (anchor clean, in-between positions given edited-run
  states). Outcome: log mass on donor rhymes minus log mass on original rhymes
  (single-token rhyme words), and TV.

## Results

### Qwen3-1.7B (2026-09-25)

Step 2 (100 couplets): without the edit line 2 rhymes with the original 83% and
with the donor 0%; with the edit, 1% and 27%. The edit changed the last word on
every couplet. Unedited generations match the authors' released text exactly on
43% (numerics and generation path), with the rhyme behaviour intact.

Steps 3-4 (100 couplets, median 17 positions between anchor and rhyme):

| Component | Rhyme preference shift [95% CI] |
|---|---|
| Total | +14.4 [13.1, 15.8] |
| Emission (edited line-2 words, edit off) | +2.8 [1.5, 4.1] |
| Persistence (edit on, words fixed) | +11.7 [10.4, 12.9] |
| Persistence, original words | +12.1 [10.6, 13.6] |
| Direct retrieval | +11.5 [10.1, 12.8] |
| Relay | +0.8 [0.5, 1.1] |
| Additivity gap | -0.15 [-0.38, 0.08] |

About 80% of the effect is persistence, and a median 96% of persistence per
couplet is direct retrieval of the anchor; relay through the in-between
positions is about 4% (small but above zero). The 27 couplets where the edit
produced a donor rhyme show the same pattern (retrieval +12.3, relay +1.0).

### Null control: same-rhyme donors, Qwen3-1.7B (2026-09-25)

`--control same_rhyme`: the donor is another couplet from the same rhyme group
(83 couplets with a partner; seed 20260925). Rhyme preference barely moves
(total -0.05 [-1.04, 0.91]; persistence with original words +0.09, retrieval
+0.14, relay -0.02), while the next-word distribution still changes a lot (TV of
persistence 0.38 vs 0.74 in the main run). The rhyme-preference measure therefore
tracks rhyme information, not generic disruption. Part of this null is by
construction (donor and original rhyme sets overlap).

Caveat: the all-layer anchor edit is disruptive. With a same-rhyme donor it
still cuts rhyming from 87% to about 35%, so step-2 rhyme rates mix steering and
disruption. The route split concerns where the rhyme-specific influence travels
and is not affected, but the milder published intervention (Hanna & Ameisen's
rhyme-feature steering, plan step 6) should be run to confirm.

### Qwen3-4B (2026-09-25)

Step 2 (100 couplets): without the edit line 2 rhymes with the original 92%,
donor 0%; with the edit, 2% and 43%. Unedited generations match the authors'
released text exactly on 59%.

Steps 3-4 (100 couplets, median 16 positions between anchor and rhyme):

| Component | 1.7B | 4B [95% CI] |
|---|---|---|
| Total | +14.4 | +19.5 [18.0, 21.0] |
| Emission | +2.8 | +3.1 [1.8, 4.4] |
| Persistence (words fixed) | +11.7 | +16.5 [15.0, 18.0] |
| Persistence, original words | +12.1 | +15.3 [13.5, 17.1] |
| Direct retrieval | +11.5 | +14.0 [12.3, 15.7] |
| Relay | +0.8 | +1.6 [1.2, 2.1] |
| Median relay share of persistence | 4% | 8% |
| Additivity gap | -0.15 | -0.29 [-0.69, 0.09] |

Direct retrieval of the anchor still dominates at 4B. Relay is small but about
twice as large as at 1.7B, which raises the hypothesis that relay grows with
scale (two sizes only; the M1 caps us at 4B). The 43 couplets where the edit
produced a donor rhyme show the same pattern (retrieval +15.2, relay +1.5).

### Null control: same-rhyme donors, Qwen3-4B (2026-09-25)

81 couplets with a same-group partner. Persistence with the original words is
+1.2 [0.6, 2.0] (retrieval +1.1, relay -0.03), against +15.3 in the main run;
the rhyme-specific signal is more than ten times the same-rhyme edit's. The edit
is less disruptive than at 1.7B: rhyming with the original falls from 91% to 51%
(1.7B: 87% to 35%).

### Converging evidence from the authors' released results

- Attention intervention (`couplets/results/attention_intervention/`): blocking
  the heads that read the line-1 anchor near the end of line 2 leaves line 2
  rhyming in only 30% / 20% / 13% / 32% of couplets at 1.7B / 4B / 8B / 14B
  (all rhymed before). Direct retrieval is necessary up to 14B; relay does not
  replace it.
- Their rhyme-feature steering (`rhyme_intervention_sample/`) is weak at small
  scale: donor-rhyme rate 11% / 12% / 35% / 33% at 1.7B / 4B / 8B / 14B. Step 6
  (the route split under their steering) will therefore have a small effect at
  4B and serves mainly as a check on the direction of the split.
