# Architecture generalization: do the key results hold on newer designs?

Design frozen 2026-09-26, before any result.

## Why

Model choices so far (Qwen3 dense full attention; Gemma 3 sliding-window hybrid; Qwen3.5
recurrent hybrid) were strategic. The paper's key results should also hold on the
architectures the field is moving to: the newest Gemma generation and mixture-of-experts.

## Core suite (per model; `core_suite.sh`, Python 3.12 environment, resumable by step)

1. Couplet routes (chat prompt): rhyme screen (100), state edit and generation, route
   split, position split, necessity split.
2. Relay positive control: induction.
3. Hidden choice: out-of-list control with pick specificity (fruits), pre-registered
   replication (animals), forced route split (fruits).

Models: Qwen3.5 - 27B (recurrent hybrid, dense), Gemma 4 - 31B (newest Gemma, dense),
Qwen3.5 - 35B-A3B (recurrent hybrid, mixture-of-experts).

## Predictions fixed in advance (from the Qwen3 / Gemma 3 results)

- Early line-2 relay about 0 (under 5% of persistence); direct retrieval the largest path.
- Induction control: direct retrieval under 5% of persistence, relay large.
- Hidden choice: relay through the generated sentence about 0.
- Stored pick specificity: family-like (Gemma 4 like Gemma 3, i.e. clearly positive;
  Qwen3.5 like Qwen3 at its size, i.e. near zero); a reversal would be reported as a
  family-by-generation difference.

**Trimmed suite (2026-09-26, before any result; time budget about 1 hour per model).**
The full couplet chain (backbone; expected; already broad) is dropped. Kept, because the
surprise-led framing rests on them: hidden choice (out-of-list specificity, animal
replication, forced route split), the induction control, and for Qwen3.5 - 35B-A3B the
fast recurrent-carry pilot (24 couplets). `core_suite_lite.sh`. Predictions unchanged.

## Results

### Qwen3.5 - 27B - hidden choice - out-of-list control (fruits; 100 pairs; 2026-09-26)

Text swap +17.91 (pick specificity +1.66); post-list block: donor pick +0.16, donor's other
fruits +0.11, **pick specificity +0.05 [0.03, 0.08]** (3% of the text swap's). As predicted
(family-like): Qwen3.5 stores almost none of the hidden pick after the list, like Qwen3 at
14B-32B (+0.00 / +0.04), unlike Gemma 3 (27B: +2.48, 31%).

### Qwen3.5 - 27B - hidden choice - replication (animals; pre-registered specificity)

Text swap specificity +1.27; post-list pick specificity **+0.05 [0.01, 0.08]** (4%). As
predicted: Qwen3.5 like Qwen3 at its size (Qwen3-32B 7%), unlike Gemma 3 27B (40%).

### Qwen3.5 - 27B - hidden choice - forced route split (fruits; 100 pairs; 2026-09-26)

Text swap +2.28; post-list block +0.19 [0.13, 0.25] (8%), direct retrieval +0.16; **relay
through the sentence +0.03 [0.02, 0.04]** (1% of the text swap; small but above zero, as in
Qwen3-32B); **sentence block +0.15 [0.12, 0.18]** (7% of the text swap). Prediction ("relay
through the generated sentence about 0") holds for relay of the stored part (1%). The sentence
positions themselves carry a small part of the pick (7%, the largest of any model; Qwen3-4B 5%,
Qwen3-32B 1%, Gemma 3 0), expected in a recurrent hybrid where each position's recurrent
state summarizes the prefix, including the list.

### Qwen3.5 - 27B - relay positive control - induction (100 sequences)

p(B) clean 0.984; persistence -4.34 [-4.55, -4.12]; **direct retrieval -0.72 [-0.85, -0.61]
(17%)**; relay -2.41 (55%); key position alone -3.35 (77%). Prediction ("direct retrieval under
5%, relay large"): relay is the main path, but direct retrieval (17%) exceeds 5%: a partial
pass, like Qwen3-32B (31%). The largest models also read the first A directly.

### Gemma 4 - 31B - hidden choice - out-of-list control (fruits; 100 pairs; 2026-09-26)

Text swap: donor pick +34.90, donor's other fruits +31.31 (pick specificity +3.59). Post-list
block: donor pick +9.19 [8.88, 9.50], other donor fruits +6.26, **pick specificity +2.93 [2.56,
3.30]** (82% of the text swap's; Gemma 3 27B 31%, Qwen3.5-27B 3%). Prediction (family-like:
Gemma 4 like Gemma 3, clearly positive) holds; the stored hidden pick is the largest of any
model.

## Deadline plan (2026-09-26, 10:30 CEST; user: all results by 19:30 JST = 12:30 CEST)

Only one job of 60 GB or more fits the memory budget at a time, so the queue was cut before any
of the affected results. Kept: Gemma 4 - 31B hidden choice animal replication (running) and
induction control; Qwen3.5 - 9B hidden choice forced route split and induction (small, run
alongside). Dropped (moved to `~/amu_jobs/queue.later.txt`, not run): Gemma 4 - 31B hidden
choice forced route split (about 90 min at this model's speed), and the whole Qwen3.5 - 35B-A3B
core suite. The trimmed-suite predictions for these remain untested.
