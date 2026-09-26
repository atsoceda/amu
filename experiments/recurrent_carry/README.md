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

## Results

### Supporting sizes (2026-09-26, Mac Studio; 40 couplets each)

| | Persistence | Attention-only | **Recurrent-only** | Both blocked | Direct retrieval | Relay |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | +12.0 [10.2, 13.6] | +11.6 | **+0.28 [0.07, 0.49]** (3%) | 0.00 | +10.5 | +1.6 |
| Qwen3.5-9B | +14.7 [13.3, 16.2] | +14.1 | **+0.55 [0.31, 0.79]** (4%) | 0.00 | +12.x | see rows |

In both, the plan reaches the rhyme position almost entirely through the full-attention
layers (1 in 4); the recurrent memory (24 of 32 layers) carries 3-4%, below the 10% stop
line at these sizes. The both-blocked sanity cell is exactly zero. The decisive 27B
pilot (stop/go on the largest model) is running.

### Decisive size: Qwen3.5-27B (2026-09-26, fast path, 24 couplets)

Persistence +18.77 [16.34, 21.27]; attention-only +17.90 [15.47, 20.43] (95%);
**recurrent-only +0.41 [0.18, 0.64]** (median share 2%); both blocked 0.00; direct
retrieval +15.82, relay +2.95.

**Decision by the frozen rule: stop** (recurrent-only far below 10% at the largest
model). In a hybrid where 48 of 64 layers can pass information forward only through a
recurrent memory, the rhyme plan still reaches its target through the 16 full-attention
layers by direct retrieval; the recurrent memory carries 2-4% at 4B, 9B and 27B and does
not take over when attention is denied the edit. "Retrieved, not relayed" extends to a
recurrent hybrid architecture. The open form of H1 is distance (attention that cannot
reach the source, e.g. Gemma 3's 1,024-token local layers).

Deadline plan (2026-09-26): Qwen3.5 - 27B - recurrent carry - pilot at 100 couplets and the
Qwen3.5 - 35B-A3B pilot were dropped (not run). The decision rests on the 24-couplet 27B pilot.
