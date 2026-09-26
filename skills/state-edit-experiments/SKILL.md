---
name: state-edit-experiments
description: >-
  Writes and runs AMU route-accounting experiments that edit hidden states (all-layer
  state patches, route splits into emission / direct retrieval / relay, sufficiency and
  necessity, nulls and positive controls) on Qwen3 and Gemma 3 models, including
  batching for speed and the pitfalls that broke earlier runs. Use when creating or
  modifying scripts under experiments/couplet_routes, derived_value_carry,
  relay_positive_control or hidden_choice, or any new state-edit experiment.
license: MIT
compatibility: >-
  PyTorch with MPS (Apple Silicon), transformers 4.57, the repo's
  experiments/couplet_routes/models.py and batching.py.
metadata:
  version: "1.0"
---

# State-edit route experiments

Vocabulary: `docs/glossary/route-accounting-glossary.html` (route axis: emission,
persistence, direct retrieval, relay; observability axis: visible vs hidden). Record
rules: `docs/research-record.md`.

## Workflow

1. **Freeze the design first.** Write the question, measurements and pre-registered
   readings in the experiment README, commit, then run. Additions after seeing any
   result are labelled post hoc with the date; never relabel a failed criterion.
2. **Smoke-test on 2-3 items**, then run fully. Run smoke tests on a model name no
   experiment *in that directory* uses (the batching check uses `Qwen3-0.6B` under
   `derived_value_carry` and `hidden_choice` only). Delete test outputs by exact path,
   never with a glob across `experiments/*/results/…`: older experiments have
   committed results under the same model names (a glob deleted Qwen3-0.6B results of
   `ab_qwen_gate` and `qwen3_planning_six_cell` on 2026-09-26; restored from git).
3. **Controls:** a same-rhyme / same-choice null (disruption vs content) and, for
   any relay claim, the induction positive control (`experiments/relay_positive_control`).
4. Record results in the README the same hour, commit and push.
5. **Checkpoint any script that can run longer than a few minutes** with
   `experiments/couplet_routes/checkpoint.py` (append each finished row; skip finished
   rows on restart). A stopped unchekpointed job loses everything (a 19-minute 32B chain
   run was lost this way on 2026-09-26). Test resume with `jobs/test_resume.sh`.

## Models and positions (`experiments/couplet_routes/models.py`)

- `load(model)` handles Qwen3 (`Qwen/…`) and Gemma 3 (`google/…`). Gemma 3 4B-27B load
  as `Gemma3ForConditionalGeneration`; decoder layers are
  `model.language_model.layers` (`decoder_layers` finds them). Gemma has no
  `/no_think`; its end of turn is `<end_of_turn>`.
- Find anchors by **character offsets** (`return_offsets_mapping=True`), not by counting
  tokens back from a template marker: tokenizers split punctuation and newlines
  differently (Qwen `,\n` is one token; Gemma `,` and `\n` are two).
- A model name ending in `-plain` switches to the plain couplet prompt.
- `output_hidden_states[1:]` gives per-layer outputs; the last entry is after the
  final norm (harmless when patching a position whose own logits are not read).

## Numerics

- **bf16 logits resolve only about 0.125 nats.** For small effects, run the output
  projection in float32 as a *separate copy* (`F32Head` in
  `derived_value_carry/chain_routes.py`); small Qwen3 models tie it to the input
  embeddings, so converting in place changes the whole network.
- Report shares of persistence across families (log-probability scales differ).

## Batching (`experiments/couplet_routes/batching.py`)

- `run_edits`: one sequence, many edits, one forward pass; replacement is one indexed
  write per layer (a Python loop over positions per layer is slow enough to cancel
  the gain). Use `logits_to_keep=1`.
- **Never left-pad a plain forward pass.** It shifts positions and fully masked rows
  can become NaN that leaks into real tokens. `last_logprobs` batches only texts of
  equal token length. `generate_batch` may left-pad: `generate` builds positions from
  the attention mask.
- Batched bf16 can flip greedy near-ties (about 1 generation in 8); numbers agree to
  about 0.001-0.03 where text is identical. Always compare a new batched script with
  its one-at-a-time path (`experiments/couplet_routes/jobs/verify_batching.py`)
  before trusting it.

## Interpretation pitfalls met so far

- Groups of positions are not a partition of paths: positions outside an edited group
  are recomputed and can read it. Report sufficiency (group alone) and necessity
  (reset the group with the source edited); a large additivity gap means redundant
  storage that sufficiency understates.
- Editing token positions changes the visible source (like changing the question);
  compare it with a text swap. A state edit at list positions can move the outcome
  in either sign because the original embeddings still feed layer 0.
- A positive effect of a donor's stored state can mean a stored choice or a stored
  copy of the donor's inputs: add a specificity contrast (donor pick vs the donor's
  other items).
- Models restate operands when free ("17 + 25 = 42"); prefill the answer prefix.
