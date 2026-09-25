# Vector construction and reverse-order decomposition: report

_Sources: `summary.json`, `rows.json`, `independent_family_summary.json`. Design (frozen before running): `../README.md`._

## Motivation

The earlier paper's central claim was mediator-relative routing: when the
article distinguishes two synonyms, a steering vector's effect goes public; when
it does not, it stays private. The vectors were built from donor prompts that
name the target word, read at the position that predicts the article. Such a
vector may carry the article decision directly, which would make the routing an
artifact of construction.

## What was measured

The frozen matched triads (14) and independent families (6 + 8), layer 17,
Gemma 3 1B. Three constructions of the same vector: `named` (the paper's);
`article_removed` (projected off the local article-readout direction,
g = d(logit an - logit a)/dh); `article_only` (that component alone, about 4% of
the norm). Also the reverse-order decomposition and log-odds at every endpoint.

## Results

| Assay | Construction | Route interaction | Lexical efficacy |
|---|---|---|---|
| Matched triads | named | 0.120 [0.079, 0.161], 13/14 positive | 0.353 / 0.415 |
| Matched triads | article removed | 0.018 [-0.000, 0.038], 8/14 | 0.344 / 0.397 |
| Matched triads | article only | 0.100 [0.059, 0.142], 12/14 | 0.022 / 0.005 |
| Independent families | named | 0.264 [0.137, 0.399]; cross-class emission-dominant 6/6 | 0.411 |
| Independent families | article removed | 0.057 [0.017, 0.093]; 1/6 | 0.388 |
| Independent families | article only | 0.191 [0.067, 0.320]; 6/6 | 0.004 |

`named` reproduces the published values exactly. The reverse-order
decomposition gives the same pattern.

## Reading

Removing a component of about 4% of the vector's norm leaves the effect on the
target word unchanged but removes nearly all of the route shift; the component
alone reproduces it. The published routing reversal came from article
information injected by the construction, not from the model routing lexical
information into the article. The claim is retracted.

## Caveats

The projection removes a local, first-order direction; article information
encoded nonlinearly could remain, but it does not produce the route shift.

## Place in the thesis

The **construction artifact**: a validity result showing that route claims made
with named-donor steering vectors can be counterfeited.
