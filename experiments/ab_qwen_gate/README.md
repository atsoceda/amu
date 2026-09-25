# A/B learnability gate on Qwen3

Behavioral gate deciding whether the in-context A/B code becomes a mediator on
Qwen3, before any route assay is built on it. It reuses the frozen A/B few-shot
preflight config (`experiments/ab_fewshot_preflight/config.json`). Gate
thresholds were fixed before any Qwen3 A/B result (see `run.py`).

| Model | Code follows context (100% bank) | Forced-code leverage 100% / 75% / 50% | A/B mass | Gate |
|---|---:|---|---:|---|
| Qwen3-0.6B | +0.019 | -0.008 / +0.026 / +0.050 | ≥0.93 | fail (G1, G2, G3) |
| Qwen3-1.7B | +0.999 | +0.243 / +0.192 / +0.098 | ≥0.998 | fail (G3 only: 0.098 > 0.081) |
| Qwen3-4B | +0.998 | +0.642 / +0.366 / -0.196 | 1.000 / 0.999 / 0.864 | fail (G4 only: 50%-bank A/B mass 0.864 < 0.90) |

At 1.7B the model learns the code: it writes `B` for formal contexts almost
always, and forcing the code moves the term. Leverage falls with reliability but
not below the prespecified one-third cutoff at 50%. With the code fixed, the
visible context still moves the term more (0.42-0.57).

```bash
/Users/anthony/miniconda3/bin/python experiments/ab_qwen_gate/run.py Qwen3-4B
```

At 4B the code is a strong in-context mediator whose leverage tracks reliability
(0.64, 0.37, then negative at 50%), passing G1-G3 by wide margins. The gate
fails only on support in the uninformative 50% bank, where the model sometimes
emits a token other than `A`/`B`. Under the prespecified rule no route assay is
built on this paradigm unless that rule is revisited.
