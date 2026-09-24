# Archived discussed feature vectors

Small, exact exports of encoder vectors, downstream decoder vectors,
encoder biases, and JumpReLU thresholds for the 270M Gemma Scope 2 CLT
features discussed in the manuscript. The manifest records the model and
CLT snapshots, feature roles, tensor keys, and checksums for the locally
preserved attribution graphs.

These exports are not a complete CLT and cannot regenerate arbitrary
features. They preserve the named mechanisms while the full downloadable
weight cache is removed to make room for the matched Gemma 1B CLT.

## Gemma 3 1B

`export_1b_discussed.py` archives exact encoder, decoder, encoder-bias and
threshold slices for every 1B Gemma Scope 2 affine CLT feature used in the 1B
experiments (S1–S4 sets, the 32-feature calibration panel, the effect-matched
pools and runs, and the sparse-frontier feature L18/F5015). It also archives a
frozen noun-only candidate pool: features ranked by mean future-noun direct
attribution at the pre-article position alone, with no article term, from the
saved 1B selection graphs (84 features with at least three selection prompts).
The manifest records source-file hashes for all 26 CLT layers.

Per-layer affine-skip matrices and decoder biases for the archived layers are
written to `gemma_scope_2_1b_pt_affine_layer_skips.local.safetensors`. That file
is about 117 MB, is gitignored, and is needed only for replacement-model
reconstruction audits, not for steering the archived features.
