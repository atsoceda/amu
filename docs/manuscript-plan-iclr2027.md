# Manuscript plan: ICLR 2027 main track (draft, 2026-09-26)

Status: plan only. It does not edit `paper.qmd` or `manuscript/`; the frozen NeurIPS
workshop version (v35) is untouched. Numbers come from the experiment READMEs and
`results/report.md` files named below and should be re-read from there when
writing. Rules that apply: `AGENTS.md` (ICLR target, no length trimming, render with
`bin/render-paper iclr submission`, 9 pages of main text).

## 1. Proposed thesis

Future-relevant information crosses token positions in two ways. By **emission**,
it is written into a mediator token and re-read, scaled by that token's leverage.
By **retrieval**, a later position reads an earlier position that holds it. In
every task tested up to 32B, in two model families, information is **not carried
privately through the generated text in between** (relay through the generated
line is about 0%). With scale, a plan gains a **second storage site at the line
boundary** (the token that ends the line), which later positions also read. A route
accounting validated on a known relay mechanism (induction) makes these claims
measurable.

Working title options: "Looked up, not carried: how latent plans cross positions in
language models"; or keep "Closing the Causal Loop" with a subtitle on routes.

## 2. Claims, evidence and status

| # | Claim | Evidence (experiment directory) | Status |
|---|---|---|---|
| C1 | The route accounting separates emission, retrieval and relay, with sufficiency and necessity versions and an additivity check | Method; `experiments/couplet_routes` (step 3/4, `relay_positions.py`, `relay_necessity.py`) | Done |
| C2 | The accounting detects relay where relay is the known mechanism | `experiments/relay_positive_control` (induction): retrieval about 0-2% and relay carrying the effect via the key position in Qwen3 1.7B-14B and Gemma 12B/27B; at Qwen3-32B relay is 89% but direct retrieval is 31% (the largest model also reads the first token directly) | Done |
| C3 | Emission follows leverage | `experiments/qwen3_planning_six_cell` (a/an planning features, 4B: emission 92-96%; leverage law); `ab_qwen_gate`, `ab_qwen_route` | Done (from earlier work) |
| C4 | In the flagship planning task (rhyming couplets) the plan reaches the rhyme by retrieval of the rhyme word; carrying through line 2 is about 0 | `couplet_routes`: Qwen3 1.7B-32B, Gemma 3 1B-27B, chat and plain formats, instruct and base, state edit and the authors' feature steering; nulls; anchor specificity | Done (32B plain, Gemma 27B plain running) |
| C5 | With scale a second storage site appears at the line-ending token | Position, necessity and per-token splits: from Qwen3 14B and Gemma 12B (plain) / 27B (chat); the comma ending line 1 in both families; present in base checkpoints (Gemma 12B PT; Qwen 14B-Base running); relay-share trend +6.2 pp per decade [2.8, 9.6] | Done, with replication runs in progress |
| C6 | C5 reconciles published accounts | Hanna & Ameisen (retrieval circuit), Lindsey et al. (line-end planning in Claude), Ma & Rui (hand-off in Gemma-3-27B only), Jacopin (no newline effect at 0.6-2.6B), Maar et al. | Writing task |
| C7 | Derived values are recomputed from visible sources at the answer, not carried | `experiments/derived_value_carry`: two-digit sums (1.7B-32B) and variable chains (1.7B-8B; 14B/32B running); accuracy on chains collapses with length, while no downstream position holds the running value, even as a block | Mostly done |
| C8 | Implication for monitoring: information that matters stays anchored to visible tokens; what to watch as models scale is boundary storage, not generated-text carrying | Discussion | Writing task |

Demoted or removed from the workshop version: mediator-relative routing (a
vector-construction artifact, retracted in `matched_triads_construction`);
Gemma 270M results become background or appendix.

## 3. Figures and tables (main text)

1. Route schematic: emission, retrieval, relay; sufficiency vs necessity (new).
2. Positive control next to couplets: induction (retrieval about 0, relay at the key
   position) vs couplets (retrieval dominant); same axes, both families.
3. Couplet route shares by size, both families and formats (from `scale_summary.py`).
4. `couplet_routes/results/scale_summary.png`: boundary, late and early line-2 relay
   by size; add necessity markers when all necessity runs are in.
5. Per-token boundary table: the comma carries the storage (Qwen3-14B, Gemma 27B).
6. Derived value: accuracy vs chain length and the block edit (about 0) vs start-digit
   edit.
Appendix: anchor specificity, nulls, feature-steering replication, a/an leverage,
additivity gaps and Shapley summaries, per-run tables.

## 4. Remaining work before writing

- Running on the Mac Studio: Qwen3-32B plain format, necessity and boundary tokens;
  Gemma 27B plain and PT plain; Qwen3-14B-Base plain; chain task 14B/32B; positive
  control at 32B and Gemma 27B.
- Optional: the authors' feature steering on Gemma (Gemma Scope 2 has per-layer
  transcoders) for a second-family replication of the steering result.
- Decide the title and whether the ICLR paper keeps the public/private framing as
  its lead or moves route accounting to the front.
- Adapt the NeurIPS checklist material to ICLR requirements (see `AGENTS.md`).
