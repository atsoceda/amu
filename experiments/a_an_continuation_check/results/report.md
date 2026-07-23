# A/An Continuation Check Report

Generated: 2026-07-23T08:02:27.254832+00:00
Model: `google/gemma-3-270m`

## Question

When Gemma incorrectly selects `a` for a prompt whose listed target requires `an`, does it continue with that listed target anyway, as in `a accountant`, or choose a different continuation?

## Demonstrations

- `a` demonstration: `Someone who studies living organisms is a biologist.`
- `an` demonstration: `Someone who studies ancient objects and sites is an archaeologist.`

## Short Answer

Gemma produced 2 preparation/content mismatch(es), where `a` was immediately followed by a vowel-initial answer. These cases justify a targeted mechanistic follow-up.

- Evaluations: 24
- Next-token `a` predictions: 21
- Exact `a` + listed-word mismatches: 0
- `a` + other vowel-initial word mismatches: 2
- `a` followed by a consonant-initial word: 19

An exact mismatch means the first generated lexical word after `a` equals the listed planned word. The broader mismatch count also includes a different vowel-initial answer, such as `a ophthalmologist`. This is a behavioral test only; the generated word has not yet been established as an internal plan.

## Mismatch Cases

These are the rows counted as preparation/content mismatches:

| Target Prompt | Demonstration | Natural Continuation | Why It Counts |
| --- | --- | --- | --- |
| `Someone who examines eyes and prescribes corrective lenses is` | `a` | `a ophthalmologist.` | The model selected `a`, then generated a different vowel-initial answer. |
| `Someone who examines eyes and prescribes corrective lenses is` | `an` | `a ophthalmologist.` | The model selected `a`, then generated a different vowel-initial answer. |

The 2 rows come from 1 unique target prompt(s); the same target was evaluated under both demonstrations.

## Every Continuation

| Target Prompt | Listed Word | Demo | Predicted Article | Natural Continuation | After Forced `a` | After Forced `an` | Listed Word LogP After `a` | After `an` | Classification |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `Someone who handles financial records is` | `accountant` | `a` | `a` | `a financial analyst.` | `financial analyst.` | `accountant. Someone` | -3.347 | -0.102 | `a_plus_consonant_word` |
| `Someone who handles financial records is` | `accountant` | `an` | `a` | `a financial analyst.` | `financial analyst.` | `accountant. Someone` | -4.441 | -0.072 | `a_plus_consonant_word` |
| `Someone who designs buildings and structures is` | `architect` | `a` | `a` | `a designer. Someone` | `designer. Someone` | `architect. Someone` | -1.398 | -0.084 | `a_plus_consonant_word` |
| `Someone who designs buildings and structures is` | `architect` | `an` | `a` | `a designer. Someone` | `designer. Someone` | `architect. Someone` | -1.278 | -0.038 | `a_plus_consonant_word` |
| `Someone who installs and repairs electrical systems is` | `electrician` | `a` | `a` | `a technician. Someone` | `technician. Someone` | `electrician. Someone` | -2.600 | -0.261 | `a_plus_consonant_word` |
| `Someone who installs and repairs electrical systems is` | `electrician` | `an` | `a` | `a technician. Someone` | `technician. Someone` | `electrician. Someone` | -3.412 | -0.313 | `a_plus_consonant_word` |
| `Someone who designs and builds technical systems is` | `engineer` | `a` | `a` | `a computer scientist.` | `computer scientist.` | `engineer. Someone` | -3.844 | -0.122 | `a_plus_consonant_word` |
| `Someone who designs and builds technical systems is` | `engineer` | `an` | `a` | `a designer. Someone` | `designer. Someone` | `engineer. Someone` | -4.082 | -0.367 | `a_plus_consonant_word` |
| `Someone who studies financial systems and markets is` | `economist` | `a` | `a` | `a financial analyst.` | `financial analyst.` | `economist. Someone` | -2.581 | -0.053 | `a_plus_consonant_word` |
| `Someone who studies financial systems and markets is` | `economist` | `an` | `an` | `an economist. Someone` | `financial analyst.` | `economist. Someone` | -4.474 | -0.220 | `article_not_a` |
| `Someone who reviews and corrects written material is` | `editor` | `a` | `a` | `a scientist. Someone` | `scientist. Someone` | `editor. Someone` | -5.864 | -0.903 | `a_plus_consonant_word` |
| `Someone who reviews and corrects written material is` | `editor` | `an` | `a` | `a historian. Someone` | `historian. Someone` | `archaeologist. Someone` | -8.171 | -2.784 | `a_plus_consonant_word` |
| `Someone who creates pictures for books and magazines is` | `illustrator` | `a` | `a` | `a scientist. Someone` | `scientist. Someone` | `illustrator. Someone` | -5.535 | -0.947 | `a_plus_consonant_word` |
| `Someone who creates pictures for books and magazines is` | `illustrator` | `an` | `a` | `a historian. Someone` | `historian. Someone` | `artist. Someone` | -6.987 | -2.204 | `a_plus_consonant_word` |
| `Someone who translates spoken language in real time is` | `interpreter` | `a` | `a` | `a linguist.` | `linguist.` | `interpreter. Someone` | -9.169 | -1.154 | `a_plus_consonant_word` |
| `Someone who translates spoken language in real time is` | `interpreter` | `an` | `a` | `a linguist.` | `linguist.` | `interpreter. Someone` | -8.681 | -0.649 | `a_plus_consonant_word` |
| `Someone who examines eyes and prescribes corrective lenses is` | `optometrist` | `a` | `a` | `a ophthalmologist.` | `ophthalmologist.` | `ophthalmologist.` | -2.038 | -0.998 | `a_plus_other_vowel_word_mismatch` |
| `Someone who examines eyes and prescribes corrective lenses is` | `optometrist` | `an` | `a` | `a ophthalmologist.` | `ophthalmologist.` | `ophthalmologist.` | -2.328 | -1.221 | `a_plus_other_vowel_word_mismatch` |
| `Someone who corrects the alignment of teeth is` | `orthodontist` | `a` | `a` | `a dentist. Someone` | `dentist. Someone` | `orthodontist.` | -2.429 | -0.085 | `a_plus_consonant_word` |
| `Someone who corrects the alignment of teeth is` | `orthodontist` | `an` | `a` | `a dentist. Someone` | `dentist. Someone` | `orthodontist.` | -4.768 | -0.405 | `a_plus_consonant_word` |
| `Someone who prepares deceased people for funerals is` | `undertaker` | `a` | `a` | `a priest. Someone` | `priest. Someone` | `archaeologist. Someone` | -9.446 | -5.733 | `a_plus_consonant_word` |
| `Someone who prepares deceased people for funerals is` | `undertaker` | `an` | `a` | `a necromancer` | `necromancer` | `archaeologist. Someone` | -11.099 | -6.376 | `a_plus_consonant_word` |
| `Someone who officially examines financial accounts is` | `auditor` | `a` | `an` | `an accountant. Someone` | `financial analyst.` | `accountant. Someone` | -7.545 | -4.054 | `article_not_a` |
| `Someone who officially examines financial accounts is` | `auditor` | `an` | `an` | `an accountant. Someone` | `financial analyst.` | `accountant. Someone` | -7.129 | -2.736 | `article_not_a` |

## Interpretation Boundary

If Gemma selects `a` and then generates the listed vowel-initial word, that is the preparation/content mismatch needed to motivate a mechanistic planning test. If it selects a grammatically compatible alternative, the article may be shaping the later answer rather than failing to prepare for an existing plan.
