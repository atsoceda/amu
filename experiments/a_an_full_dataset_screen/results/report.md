# Full A/An Dataset Screen

Generated: 2026-07-23T08:22:04.158432+00:00
Model: `google/gemma-3-270m`

## Question

Across every `an`-target prompt released with *Latent Planning Emerges with Scale*, how often does Gemma 3 270M select `a` and then generate a vowel-initial answer?

## Source

- Repository: `https://github.com/hannamw/model-planning-public`
- Commit: `993cff51450f43e8e86c254f761cc48c008b8dab`
- Dataset: `a_an/data/a_an_examples.csv`
- SHA-256: `fdc65578873d8f3ed1f364c221a32f5c75c64c109f0b25a1a4660d35c5df134c`
- Released rows: 457 (`a`: 352, `an`: 105)

## Demonstrations

- `a` demonstration: `Someone who studies living organisms is a biologist.`
- `an` demonstration: `Someone who studies ancient objects and sites is an archaeologist.`

## Short Answer

The screen found some behavioral mismatch candidates, but fewer than the ten distinct targets set as the threshold for a broad mechanistic study.

- Unique `an` targets screened: 105
- Total evaluations: 210
- `an` article recall: 21.9%
- Candidate mismatch rows: 3
- Unique candidate targets: 2
- Candidates reproduced under both demonstrations: 1

A candidate is an orthographic screen: `a` followed by a word beginning with a vowel letter. Candidate words must be manually checked for pronunciation before mechanistic follow-up.

## Candidate Mismatches

| Target Prompt | Listed Word | Demo | Natural Continuation | Classification |
| --- | --- | --- | --- | --- |
| `Someone who treats eye diseases is` | `ophthalmologist` | `a` | `a ophthalmologist.` | `a_plus_listed_word_mismatch` |
| `Someone who treats eye diseases is` | `ophthalmologist` | `an` | `a ophthalmologist.` | `a_plus_listed_word_mismatch` |
| `Someone who studies stars and planets is` | `astronomer` | `an` | `a astronomer. Someone` | `a_plus_listed_word_mismatch` |

## Outcome Counts

| Classification | Count |
| --- | ---: |
| `a_plus_consonant_word` | 102 |
| `a_plus_listed_word_mismatch` | 3 |
| `article_an` | 46 |
| `other` | 59 |

Full prompt-level results are stored in `results/examples.jsonl`. The report intentionally foregrounds candidate mismatches rather than presenting a 210-row table.

## Interpretation Boundary

This screen identifies behavioral preparation/content mismatches. It does not establish that the generated answer was represented before the article or that a competing article circuit concealed latent planning.
