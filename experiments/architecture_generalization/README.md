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
