# A/B reliability preflight

This experiment redesigns the artificial A/B mediator assay after the original
four-family pilot showed measurable code-to-term leverage and a much larger
fixed-code context effect, but failed to provide an ordered reliability knob.

The workflow is deliberately staged:

1. `generate_candidates.py` creates a broad lexical candidate pool without A/B
   labels or route outcomes.
2. `screen_candidates.py` tests whether Gemma 3 1B PT itself robustly prefers
   the formal-versus-everyday log odds move in the requested direction when
   `Register: EVERYDAY` changes to `Register: FORMAL`, across prompt paraphrases
   after counterbalancing choice-order nuisance. Absolute lexical priors are not
   required to cross zero.
3. `freeze_split.py` creates disjoint demonstration, development, and untouched
   confirmatory family sets before any A/B route measurement.
4. `run_preflight.py` (added after the lexical gate succeeds) evaluates
   randomized reliability banks, label complements, and demonstration orders.

No residual intervention is authorized by this stage. The behavioral gate must
first establish A/B support, token-role reversal, order robustness, near-zero
code leverage at 50% reliability, strong leverage at 100%, monotone policy
movement, a fixed-code register effect, and held-out generalization.

The primary statistical unit is the semantic family. Random banks, label swaps,
choice orders, register paraphrases, and demonstration orders are repeated
measurements within family.
