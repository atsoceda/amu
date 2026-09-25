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

### Step 6: route split under the authors' rhyme-feature steering, Qwen3-4B (2026-09-25)

Rhyme-feature selection (`step6_features.py`, the authors' `_is_rhyme_feature`
rule on all features active at the anchor, transcoders streamed): mean 19.1
features per couplet vs the authors' 19.2; exact per-couplet count match on
67/100. Steering (`step6_routes.py`): own rhyme features to -3x, donor rhyme
features to 7x the donor's activation, at the anchor.

| Component | State edit | Feature steering [95% CI] |
|---|---|---|
| Total | +19.5 | +14.9 [13.3, 16.4] |
| Emission | +3.1 | +1.4 [0.5, 2.5] |
| Persistence, original words | +15.3 | +11.8 [10.2, 13.3] |
| Direct retrieval | +14.0 | +10.6 [9.2, 12.1] |
| Relay | +1.6 | +1.0 [0.7, 1.3] |
| Median retrieval / relay share | 92% / 8% | 93% / 6% |
| Additivity gap | -0.29 | +0.11 [-0.10, 0.32] |
| Line 2 rhymes with donor | 43% | 16% (authors' released run: 12%) |

The milder published intervention gives the same split, so the state edit's
disruption does not explain the result. Steering shifts the rhyme distribution
strongly but rarely wins the emitted word, matching the authors' low steered
rhyme rate at 4B. The 16 couplets that switched to the donor rhyme show the same
pattern (retrieval +15.8, relay +1.8).

## Anchor specificity (design frozen 2026-09-25 before any result; `anchor_specificity.py`)

The donor's anchor states are placed at one position of the original prompt
(original line-2 words fixed). Prediction: large persistence only at the anchor
(A), possibly at adjacent end-of-line positions (C, E); little at content
positions (M, F).

Qwen3-1.7B, 100 couplets (rhyme-preference shift [95% CI]):

| Position | Persistence | Direct retrieval | Relay |
|---|---|---|---|
| A: last word of line 1 (anchor) | +12.1 [10.6, 13.6] | +11.5 | +0.8 |
| E: end of user turn | +6.0 [5.1, 7.0] | +5.8 | +0.8 |
| C: punctuation after anchor | +5.7 [4.8, 6.7] | +5.6 | +0.7 |
| M: middle word of line 1 | +5.5 [4.4, 6.6] | +4.1 | +0.6 |
| F: first word of line 1 | +3.5 [2.7, 4.3] | +2.7 | +0.6 |

Partly supported: the anchor is retrieved 2-3.5x more than any other position,
but rhyme information at other positions is still partly retrieved (30-50% of
the anchor's effect). The accurate description is retrieval weighted toward
structural anchors, not anchor-only retrieval. Relay is small from every
position (+0.6 to +0.8).

## Cross-scale summary (Mac Studio for 8B; generated from the committed JSON files)

| | 1.7B | 4B | 8B |
|---|---|---|---|
| Rhyme with original off -> on / with donor on | 83% -> 1% / 27% | 92% -> 2% / 43% | 96% -> 1% / 61% |
| Total | +14.4 [13.1, 15.8] | +19.5 [18.0, 21.0] | +26.0 [24.2, 27.8] |
| Emission | +2.7 [1.5, 4.1] | +3.1 [1.8, 4.4] | +3.6 [2.1, 5.2] |
| Persistence, original words | +12.1 [10.6, 13.6] | +15.3 [13.5, 17.1] | +22.6 [20.9, 24.4] |
| Direct retrieval | +11.5 [10.1, 12.8] | +14.0 [12.3, 15.7] | +20.4 [18.8, 22.1] |
| Relay | +0.8 [0.5, 1.1] | +1.6 [1.2, 2.0] | +2.4 [1.9, 2.9] |
| Median relay share of persistence | 4% | 8% | 8% |
| Same-rhyme null: persistence | +0.1 [-0.6, 0.6] | +1.2 [0.6, 2.0] | +2.0 [1.1, 3.2] |

Anchor specificity at 8B (persistence): A +22.6 [20.9, 24.4], M +8.6 [6.9, 10.4], F +5.1 [3.8, 6.5], C +10.7 [8.9, 12.5], E +10.6 [8.8, 12.4]; the anchor is again about twice as effective as any other position.

Reading: direct retrieval dominates at all three sizes. Relay grows in absolute size with the whole effect, but its median share goes 4% -> 8% -> 8%: no further growth from 4B to 8B. 14B and 8B step 6 are pending.

## Relay-tail analysis (2026-09-25; `relay_tail.py`, existing data)

Couplets with relay share above 30%: 5/88 (1.7B), 8/89 (4B), 5/98 (8B). No
systematic relay regime:

- At 4B, 4 of the 8 are couplets whose original line 2 first repeats line 1 (vs
  4 of 81 among the rest). The copy puts the rhyme word among the "in-between"
  positions, so the target can retrieve it there: retrieval from a second anchor
  that the split counts as relay (a definitional artifact).
- At every size the tail couplets have weaker edits (median persistence 9-17 vs
  14-24), so part of the tail is the share ratio being noisy.
- No other difference: no donor- or original-rhyme words elsewhere in the prefix,
  and the same number of in-between positions (except the 4B copies).

Conclusion: relay stays small for visible rhyme information; the derived-value
task is the real test of whether relay appears when the information must be
computed.

## Updates (2026-09-25, late evening)

- **4B anchor specificity:** anchor +15.3 [13.5, 17.1]; end of turn +7.0, punctuation
  +6.8, mid-line +4.3, first word +2.7; relay +0.4 to +1.6 from every position.
- **8B step 6 (authors' feature steering, Mac Studio):** donor-rhyme rate 27% (authors'
  released 35%), original 2%. Persistence with original words +20.4 [18.8, 21.9] =
  direct retrieval +17.9 [16.4, 19.4] + relay +1.9 [1.5, 2.3] (gap +0.5); emission
  +2.7. Where the published intervention genuinely moves the rhyme, the plan is still
  about 88% retrieval.

Full tables: `results/report.md` (regenerated).
