# Fixed-Pair Generalization

This is the held-out test of the pair discovered on:

```text
Someone who treats eye diseases is
```

The intervention is frozen as `L13/F10304 + L14/F1949`. The experiment applies
those same feature identities at the final token of every other
occupation-style prompt in the paper's released `a/an` dataset. There is no
attribution, feature selection, or pair selection on the held-out prompts.

The original ophthalmologist sentence is excluded. Expected-`an` occupations
test transfer; expected-`a` occupations are controls that expose indiscriminate
article flipping.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/fixed_pair_generalization/run.py
```

To regenerate it, prompt an agent with:

> Regenerate the `fixed_pair_generalization` results.

## Result

The fixed pair did not generalize as a content-preserving preparation control:

- 10 of 21 expected-`an` prompts changed from generated `a` to generated `an`;
- 22 of 86 expected-`a` controls also changed to generated `an`;
- the first generated content word changed on 37 of 107 prompts; and
- exact listed-word completions fell from 53 to 44.

The generated phrases remained grammatically coordinated because the noun often
changed with the article, for example:

```text
a pilot        -> an aviator
a lawyer       -> an attorney
a physicist    -> an astronomer
```

This suggests the pair controls a broader `a`/consonant versus `an`/vowel
response pathway. It does not cleanly expose a fixed future answer that was
already planned. The source `an ophthalmologist` correction therefore cannot
yet be claimed as a reusable concealed-planning intervention.
