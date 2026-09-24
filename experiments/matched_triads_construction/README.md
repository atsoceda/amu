# Matched triads: vector construction and reverse-order decomposition (Workstreams C and D)

Frozen 2026-09-25, before any result was computed. Uses the 14 admissible
triads, layer 17, strengths {0.5, 1, 1.5} and temperatures {0.1, 0.25, 0.5, 1}
frozen in `experiments/matched_semantic_triads_repaired/`.

## C: model or construction?

Named-donor directions \(\delta\) are read at the position that predicts the
article, so a cross-class \(\delta\) may carry article information directly.
Let \(g\) be the gradient of logit(`an`) − logit(`a`) at the neutral prompt's
final position with respect to the layer-17 residual there, and
\(\hat g = g/\|g\|\). Constructions:

- `named`: \(\delta\) (the paper's construction; must reproduce the committed rows)
- `article_removed`: \(\delta - (\delta\cdot\hat g)\hat g\)
- `article_only`: \((\delta\cdot\hat g)\hat g\) (diagnostic)

Primary comparison: the paired interaction \(R_{\rm cross}-R_{\rm within}\) at
\(s=1,\tau=1\) for `article_removed` against `named`, together with local
efficacy (fixed-target-article effect) under each construction. Also reported:
the first-order article push \(s\,\delta\cdot g\) by arm, and the actual
article-logit change. Removing \(\hat g\) removes only the first-order,
local article direction; later layers may rebuild article information, so a
surviving interaction shows that this component is not necessary, not that the
vector carries no article information at all.

An "unnamed donor" construction is not possible within matched triads: the three
words share one definition, so no description-only donor separates the targets.

## D: reverse order and log-odds

Every cell records the forward decomposition (public = policy change under the
intervention-off branches; private = remainder under the treated policy), the
reverse order (private at the baseline policy first; public under the treated
branches), and target-vs-source log-odds at every endpoint.

## Run

```bash
/Users/anthony/miniconda3/bin/python experiments/matched_triads_construction/run.py
```

## Results (2026-09-25)

`named` reproduces the committed matched-triad assay exactly (paired interaction
0.1196; maximum per-triad difference 0.0). Paired interaction
\(R_{\rm cross}-R_{\rm within}\) at \(s=1,\tau=1\), 14 triads:

| Construction | Interaction | Positive | Cross-arm target-article policy change | Fixed-article efficacy (cross / within) |
|---|---|---:|---:|---|
| `named` | 0.120 [0.079, 0.161] | 13/14 | +0.099 | 0.353 / 0.415 |
| `article_removed` | 0.018 [-0.000, 0.038] | 8/14 | +0.002 | 0.344 / 0.397 |
| `article_only` | 0.100 [0.059, 0.142] | 12/14 | +0.097 | 0.022 / 0.005 |

The article-readout component is about 4% of the vector norm (31 of about 700
in the cross arm). Removing it leaves lexical efficacy essentially unchanged but
removes nearly all of the route shift and of the cross-arm article recruitment;
the component alone reproduces most of the route shift with almost no lexical
efficacy. The reverse-order decomposition gives the same picture (0.125, 0.019,
0.101).

Reading: in this assay, mediator-relative public recruitment is carried by
article information placed directly in the named-donor direction, not by the
model converting the lexical target into an article choice downstream of layer
17. The paper's mediator-relative routing claim must be revised accordingly.
The first-order caveat above applies: this shows the component is sufficient
and, to first order, necessary for the route shift; it does not exclude article
information elsewhere in the vector that later layers could use but here do not.
