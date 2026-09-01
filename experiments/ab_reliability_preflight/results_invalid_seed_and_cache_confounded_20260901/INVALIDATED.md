# Invalidated exploratory run

These 48 completed variants are retained for audit only and must not be used as
confirmatory evidence.

The run was stopped on 2026-09-01 for two reasons:

1. label-role complements and order repetitions used different compound random
   seeds, so they changed more than the intended factor;
2. long-prefix cached inference differed from ordinary full-prompt inference by
   as much as 0.125 target logit (0.022 pair-conditional probability) on the
   validation panel.

The corrected protocol independently freezes bank assignment, choice layout,
demonstration order, and literal A/B role, and uses uncached full-prompt
inference for the decisive 50% endpoint.
