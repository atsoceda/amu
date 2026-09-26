# Recurrent carry: do hybrid models carry plans through recurrent memory? (pilot)

Pilot design frozen 2026-09-26, before any result.

## Question (H1, architecture)

In every model so far (Qwen3, Gemma 3: attention everywhere), a plan reaches its target
by direct retrieval or by storage at structural positions, never by relay through
generated text; full attention can always read the source, so relay is never needed.
Qwen3.5 is a hybrid: 3 of every 4 layers are Gated DeltaNet (recurrent linear
attention with a fixed-size memory updated position by position), 1 in 4 is full
attention. In the recurrent layers, the only way information from the source reaches
a later position is by being carried forward through the recurrent state: relay by
construction, and invisible in the text. Does the rhyme plan travel that way?

## Pilot

- Model: Qwen3.5-4B (32 layers: 24 Gated DeltaNet, 8 full attention), separate
  environment (Python 3.12, current transformers) on the Mac Studio.
- Couplets: the authors' prompt (chat, thinking off), 40 couplets from the rhyme screen
  (`couplet_routes/screen_rhymes.py`), donors of a different rhyme group; state edit
  and line-2 generation from `couplet_routes/step2_state_edit.py`.
- Cells, with the original line-2 words fixed and R = log mass on donor rhymes - log
  mass on original rhymes at the rhyme position:
  - persistence: donor anchor state (all layers);
  - **attention-only**: donor anchor state, but every Gated DeltaNet layer receives the
    clean anchor as input (the edit can reach later positions only through the
    full-attention layers);
  - **recurrent-only**: donor anchor state, but every full-attention layer receives the
    clean anchor as input (the edit can reach later positions only through the
    recurrent memory);
  - both blocked (sanity: about 0);
  - the standard direct-retrieval and relay cells for comparison.
- **Stop/go rule (fixed in advance):** go to a full study (more couplets, 9B/27B,
  distance, hidden choice) if recurrent-only is at least 25% of persistence with its
  95% CI above zero; stop if it is below 10%; in between, extend the pilot to 100
  couplets before deciding.

**Amendment (2026-09-26, before any pilot result).** The pilot runs at Qwen3.5-4B, 9B and
27B (dense hybrid, 64 layers: 48 Gated DeltaNet, 16 full attention). Qwen3.5-4B rhymes
rarely (about 1 in 6 first lines), and in Qwen3 storage effects appeared only from 14B,
so a small-model null would not be informative. The stop/go rule applies to the
**largest model piloted**; smaller sizes are reported as the scale trend.

**Fast path for the 27B pilot (2026-09-26, before any result):** 24 couplets, and the
unedited line 2 is the rhyme screen's own greedy line (same prompt; identical to step 2's
unedited generation), so step 2 is skipped (`run_pilot_fast.sh`, `pilot.py --from-screen`).
The stop/go rule is unchanged; an in-between result extends the pilot.
