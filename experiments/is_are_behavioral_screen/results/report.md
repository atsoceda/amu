# Is/Are Behavioral Screen

Generated: 2026-07-23T08:25:35.302602+00:00
Model: `google/gemma-3-270m`

## Question

On the paper's released subtraction task, does Gemma 3 270M produce the specific preparation/content mismatch `are 1`?

## Source

- Repository: `https://github.com/hannamw/model-planning-public`
- Commit: `993cff51450f43e8e86c254f761cc48c008b8dab`
- Dataset: `is_are/data/animals_dataset.csv`
- SHA-256: `e2167d1a6ed8ca0849fcda75a790f1f2d41c594113cbd342560edcc05e6eeccd`

## Short Answer

Gemma produced no `are 1` mismatches, so this task does not provide the required behavioral planning failures.

- Prompts screened: 360
- Singular-answer prompts: 80
- Exact `are 1` mismatches: 0
- Unique animals with an `are 1` mismatch: 0
- Overall number accuracy: 4.2%
- Overall verb accuracy: 77.8%

## Exact `are 1` Mismatches

| Prompt | Natural Continuation |
| --- | --- |

## Outcome Counts

| Gold Verb | Predicted Verb | Number Correct | Count |
| --- | --- | --- | ---: |
| `are` | `are` | `false` | 265 |
| `are` | `are` | `true` | 15 |
| `is` | `are` | `false` | 80 |

Full prompt-level results are in `results/examples.jsonl`.

## Interpretation Boundary

An `are 1` continuation establishes a behavioral mismatch. It does not prove that a representation of `1` was active before the verb or causally supported the later number.
