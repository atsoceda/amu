# Gemma 1B effect-matched sparse intervention

Development-only top-k/gain selection followed by a frozen held-out six-cell
assay. The operating point targets the 270M S1 mean article-margin movement
(3.3125 logits); top-four at 5x remains the unmatched reference.

`sweep.py` also records intervention-off article margins, forced-a/forced-an
noun TV, article probabilities across temperatures, held-out feature
activations, decoder norms, and graph-cap coverage.

