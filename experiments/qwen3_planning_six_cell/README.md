# Six-cell route assay on published Qwen3 a/an planning features (Workstream A)

Question: when a future-token intervention is selected **without any article
criterion**, where does its effect on the later noun travel: through the
generated article (public), through context that survives a fixed article
(private), or both?

Handles are Hanna & Ameisen's a/an planning nodes (commit `993cff5`,
`a_an/planning_node_intervention.py`), reproduced for Qwen3-0.6B, 1.7B and 4B
with their per-layer transcoders (`mwhanna/qwen3-*-transcoders*`). A planning
node is a transcoder feature active at the pre-article position whose feature
card represents the planned word. The rule has no article term.

## Design (frozen before any six-cell result was computed, 2026-09-24)

- Prompts: all 349 rows of the authors' released a/an intervention table for
  each model; the assay uses every prompt with at least one planning node.
- Selection: `select_features.py`, which copies the authors' word-feature test
  unchanged. Transcoders are never stored: encoders are streamed one layer at a
  time by HTTP range request, and only active features' cards and selected
  features' decoder rows are fetched. Validation compares our per-prompt node
  counts with the authors' released `selected_nodes_count`.
- Conditions (co-primary, as in the authors' paper): selected nodes **zeroed**,
  and selected nodes **multiplied to 5x** their activation. Controls: the same
  number of randomly chosen features active at the same position, zeroed and 5x
  (seed 20260924).
- Intervention: add \((\text{target}-a_f)W_{\rm dec}[f]\) to that layer's MLP
  output at the pre-article position, kept active during noun prediction, with
  full recomputation. This matches circuit-tracer's unconstrained
  `feature_intervention` for per-layer transcoders with live error terms.
- Six cells: intervention off/on \(\times\) {free, do(`a`), do(`an`)}.
  Estimands: total, treated-article replay (public), fixed treated article
  (private), reverse-order private at the baseline article, and \(\tau=1\)
  policy-weighted versions over {`a`,`an`}; planned-word log-probability in each
  decomposition; top-1 nouns; `a/an` support.
- Reported regardless of sign or size. A null (no interpretable noun effect) is
  a result, not a reason to change the rule.

## Run

```bash
/Users/anthony/miniconda3/bin/python experiments/qwen3_planning_six_cell/select_features.py Qwen3-0.6B
/Users/anthony/miniconda3/bin/python experiments/qwen3_planning_six_cell/six_cell.py Qwen3-0.6B
```

Outputs go to `results/<model>/`. Streaming the encoders takes about 25 minutes
for 0.6B and 75 minutes for 4B at about 7 MB/s.

## Results

### Qwen3-0.6B (2026-09-24)

Selection matched the authors' released node counts exactly on 344/349 prompts
(115 prompts with nodes, 166 nodes). Six-cell assay on the 115 prompts:

- No condition switched the greedy article on any prompt.
- Noun-distribution TV is at random-control level: zeroed 0.016 vs random
  0.018; 5x 0.033 vs random 0.037. All of it sits in the fixed-article (private)
  cell, because the article never changed.
- The only target-specific effect: at 5x, the planned word's log-probability
  rises by 0.068 at \(\tau=1\), mostly through the public route (0.059 [0.020,
  0.105]; random control -0.007 [-0.018, 0.004]); the private part is 0.009
  [-0.016, 0.034]. Zeroing has no effect (-0.003).

Reading: at 0.6B the published planning nodes have no interpretable effect on
the noun; the small effect of amplifying them travels through the article.
This matches the authors' report that planning is weak at this scale. Full
table: `results/report.md`; rows: `results/Qwen3-0.6B/six_cell_rows.jsonl`.

### Qwen3-1.7B (2026-09-25)

Selection matched the authors' counts exactly on 338/349 prompts (237 vs 235
prompts with nodes; 559 vs 560 nodes). Six-cell assay on 237 prompts:

- Greedy article switches: at most 1/237 in any condition; noun TV again at
  random-control level (zeroed 0.017 vs 0.019; 5x 0.035 vs 0.041).
- Target-specific effects on the planned word's log-probability at \(\tau=1\)
  are public: 5x +0.045 [0.022, 0.073] (random +0.009 [-0.003, 0.023]); zeroed
  -0.010 [-0.018, -0.005] (random -0.002). Private parts are about zero
  (5x -0.002 [-0.019, 0.013]; zeroed -0.004 [-0.013, 0.006]).

Reading: as at 0.6B, published planning nodes change the planned word only
slightly, in both directions, and only through the article.

### Qwen3-4B (2026-09-25)

Selection: exact per-prompt count match on 297/349 prompts; totals 3371 vs
3373 nodes, 336 vs 335 prompts with nodes (about 10 nodes per prompt). Six-cell
assay on 336 prompts:

| Condition | Article switches | Noun TV total / public / private | \(\tau=1\) Δlog p(planned): total | public | private |
|---|---:|---|---|---|---|
| Planning 5x | 9.5% | 0.100 / 0.076 / 0.026 | +0.199 | +0.187 [0.112, 0.269] | +0.012 [0.000, 0.024] |
| Random 5x | 2.7% | 0.045 / 0.011 / 0.036 | -0.030 | -0.028 [-0.056, -0.003] | -0.003 [-0.019, 0.013] |
| Planning zeroed | 5.7% | 0.057 / 0.042 / 0.018 | -0.188 | -0.173 [-0.230, -0.120] | -0.015 [-0.022, -0.007] |
| Random zeroed | 0.3% | 0.015 / 0.000 / 0.015 | -0.004 | -0.003 [-0.011, 0.004] | -0.001 [-0.008, 0.005] |

(Noun TV columns use greedy articles over prompts with article support; the
log-probability columns are \(\tau=1\) mixtures over all 336 prompts.)

Reading: at 4B, where the authors report planning, the published planning
nodes have an interpretable, feature-specific, bidirectional effect on the
planned noun, and about 92-94% of it travels through the generated article.
Fixed-article noun changes are at random-control level; the only private
effect is a small decrease when the nodes are zeroed. Across 0.6B, 1.7B and 4B
the planning-node effect grows with scale and stays public.
