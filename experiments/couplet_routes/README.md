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
