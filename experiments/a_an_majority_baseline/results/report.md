# A/An Majority Baseline Report

Generated: 2026-07-23T07:43:00.144716+00:00
Model: `google/gemma-3-270m`

## Question

Does Gemma 3 270M reproduce the paper's behavioral result that small Qwen-3 models default to the majority article `a` on prompts whose planned word requires `an`?

This experiment measures only the immediate next-token article. It does not establish that Gemma internally planned the listed noun, and it does not test an intervention.

## Demonstrations

Each target prompt was evaluated twice, once after each completed example:

- `a` demonstration: `Someone who studies living organisms is a biologist.`
- `an` demonstration: `Someone who studies ancient objects and sites is an archaeologist.`

For example, the target `Someone who handles financial records is` was appended after each demonstration, and the next-token probabilities of `a` and `an` were measured.

## Result

Gemma shows a meaningful minority-class disadvantage, but not a complete collapse to `a`.

- `an` recall: 12.5% (3/24)
- `a` recall: 100.0% (40/40)
- Other-token predictions: 0/64

Each target was evaluated after both an `a` demonstration and an `an` demonstration. This exposes whether the one-shot example itself controls the answer.

## Recall By Demonstration

| Expected | Demonstration | N | Recall | Mean P(a) | Mean P(an) | Mean an-a Logit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `a` | `a` | 20 | 100.0% | 0.7206 | 0.1958 | -1.343 |
| `a` | `an` | 20 | 100.0% | 0.7129 | 0.2213 | -1.228 |
| `an` | `a` | 12 | 8.3% | 0.6505 | 0.2458 | -1.083 |
| `an` | `an` | 12 | 16.7% | 0.6368 | 0.2865 | -0.890 |

## Every Prediction

| Target Prompt | Planned Word | Expected | Demonstration | Predicted | P(a) | P(an) | an-a Logit |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `Someone who handles financial records is` | `accountant` | `an` | `a` | `a` | 0.7198 | 0.2380 | -1.106 |
| `Someone who handles financial records is` | `accountant` | `an` | `an` | `a` | 0.6746 | 0.2951 | -0.827 |
| `Someone who designs buildings and structures is` | `architect` | `an` | `a` | `a` | 0.5695 | 0.2060 | -1.017 |
| `Someone who designs buildings and structures is` | `architect` | `an` | `an` | `a` | 0.5843 | 0.2344 | -0.913 |
| `Someone who installs and repairs electrical systems is` | `electrician` | `an` | `a` | `a` | 0.7054 | 0.2013 | -1.254 |
| `Someone who installs and repairs electrical systems is` | `electrician` | `an` | `an` | `a` | 0.5377 | 0.4081 | -0.276 |
| `Someone who designs and builds technical systems is` | `engineer` | `an` | `a` | `a` | 0.6473 | 0.2280 | -1.043 |
| `Someone who designs and builds technical systems is` | `engineer` | `an` | `an` | `a` | 0.6358 | 0.2874 | -0.794 |
| `Someone who studies financial systems and markets is` | `economist` | `an` | `a` | `a` | 0.5158 | 0.3860 | -0.290 |
| `Someone who studies financial systems and markets is` | `economist` | `an` | `an` | `an` | 0.4580 | 0.4711 | 0.028 |
| `Someone who reviews and corrects written material is` | `editor` | `an` | `a` | `a` | 0.7385 | 0.2011 | -1.301 |
| `Someone who reviews and corrects written material is` | `editor` | `an` | `an` | `a` | 0.7304 | 0.2255 | -1.175 |
| `Someone who creates pictures for books and magazines is` | `illustrator` | `an` | `a` | `a` | 0.6771 | 0.2381 | -1.045 |
| `Someone who creates pictures for books and magazines is` | `illustrator` | `an` | `an` | `a` | 0.7504 | 0.1942 | -1.352 |
| `Someone who translates spoken language in real time is` | `interpreter` | `an` | `a` | `a` | 0.6794 | 0.1192 | -1.741 |
| `Someone who translates spoken language in real time is` | `interpreter` | `an` | `an` | `a` | 0.6541 | 0.1473 | -1.491 |
| `Someone who examines eyes and prescribes corrective lenses is` | `optometrist` | `an` | `a` | `a` | 0.5342 | 0.4221 | -0.236 |
| `Someone who examines eyes and prescribes corrective lenses is` | `optometrist` | `an` | `an` | `a` | 0.5682 | 0.4108 | -0.324 |
| `Someone who corrects the alignment of teeth is` | `orthodontist` | `an` | `a` | `a` | 0.7122 | 0.1443 | -1.596 |
| `Someone who corrects the alignment of teeth is` | `orthodontist` | `an` | `an` | `a` | 0.7277 | 0.1854 | -1.367 |
| `Someone who prepares deceased people for funerals is` | `undertaker` | `an` | `a` | `a` | 0.8558 | 0.0737 | -2.452 |
| `Someone who prepares deceased people for funerals is` | `undertaker` | `an` | `an` | `a` | 0.8418 | 0.0929 | -2.203 |
| `Someone who officially examines financial accounts is` | `auditor` | `an` | `a` | `an` | 0.4513 | 0.4916 | 0.086 |
| `Someone who officially examines financial accounts is` | `auditor` | `an` | `an` | `an` | 0.4789 | 0.4858 | 0.014 |
| `Someone who heals sick pets is` | `veterinarian` | `a` | `a` | `a` | 0.8271 | 0.1259 | -1.882 |
| `Someone who heals sick pets is` | `veterinarian` | `a` | `an` | `a` | 0.6936 | 0.2774 | -0.916 |
| `Someone who educates children in schools is` | `teacher` | `a` | `a` | `a` | 0.8206 | 0.1305 | -1.838 |
| `Someone who educates children in schools is` | `teacher` | `a` | `an` | `a` | 0.8000 | 0.1622 | -1.596 |
| `Someone who prepares meals in restaurants is` | `chef` | `a` | `a` | `a` | 0.8430 | 0.0866 | -2.275 |
| `Someone who prepares meals in restaurants is` | `chef` | `a` | `an` | `a` | 0.8506 | 0.0852 | -2.301 |
| `Someone who extinguishes fires and rescues people is` | `firefighter` | `a` | `a` | `a` | 0.6743 | 0.2539 | -0.977 |
| `Someone who extinguishes fires and rescues people is` | `firefighter` | `a` | `an` | `a` | 0.7951 | 0.1710 | -1.537 |
| `Someone who flies airplanes is` | `pilot` | `a` | `a` | `a` | 0.6570 | 0.3142 | -0.738 |
| `Someone who flies airplanes is` | `pilot` | `a` | `an` | `a` | 0.5122 | 0.4609 | -0.105 |
| `Someone who treats teeth and gums is` | `dentist` | `a` | `a` | `a` | 0.7308 | 0.1695 | -1.461 |
| `Someone who treats teeth and gums is` | `dentist` | `a` | `an` | `a` | 0.7282 | 0.1648 | -1.486 |
| `Someone who manages books and helps people find information is` | `librarian` | `a` | `a` | `a` | 0.7612 | 0.1951 | -1.361 |
| `Someone who manages books and helps people find information is` | `librarian` | `a` | `an` | `a` | 0.8268 | 0.1198 | -1.932 |
| `Someone who takes professional pictures is` | `photographer` | `a` | `a` | `a` | 0.7098 | 0.2457 | -1.061 |
| `Someone who takes professional pictures is` | `photographer` | `a` | `an` | `a` | 0.7439 | 0.2237 | -1.202 |
| `Someone who fixes cars and engines is` | `mechanic` | `a` | `a` | `a` | 0.6841 | 0.2464 | -1.021 |
| `Someone who fixes cars and engines is` | `mechanic` | `a` | `an` | `a` | 0.6926 | 0.2650 | -0.961 |
| `Someone who performs operations on patients is` | `surgeon` | `a` | `a` | `a` | 0.7897 | 0.1783 | -1.488 |
| `Someone who performs operations on patients is` | `surgeon` | `a` | `an` | `a` | 0.8382 | 0.1460 | -1.748 |
| `Someone who dispenses medications is` | `pharmacist` | `a` | `a` | `a` | 0.8043 | 0.1409 | -1.742 |
| `Someone who dispenses medications is` | `pharmacist` | `a` | `an` | `a` | 0.8066 | 0.1668 | -1.576 |
| `Someone who writes news articles is` | `journalist` | `a` | `a` | `a` | 0.8070 | 0.1537 | -1.659 |
| `Someone who writes news articles is` | `journalist` | `a` | `an` | `a` | 0.7688 | 0.2027 | -1.333 |
| `Someone who fixes pipes and water systems is` | `plumber` | `a` | `a` | `a` | 0.7558 | 0.1842 | -1.412 |
| `Someone who fixes pipes and water systems is` | `plumber` | `a` | `an` | `a` | 0.6717 | 0.2864 | -0.852 |
| `Someone who represents clients in legal matters is` | `lawyer` | `a` | `a` | `a` | 0.6747 | 0.2946 | -0.828 |
| `Someone who represents clients in legal matters is` | `lawyer` | `a` | `an` | `a` | 0.6442 | 0.3355 | -0.652 |
| `Someone who makes bread and pastries is` | `baker` | `a` | `a` | `a` | 0.7632 | 0.1233 | -1.823 |
| `Someone who makes bread and pastries is` | `baker` | `a` | `an` | `a` | 0.7259 | 0.1626 | -1.496 |
| `Someone who builds things with wood is` | `carpenter` | `a` | `a` | `a` | 0.7258 | 0.1771 | -1.411 |
| `Someone who builds things with wood is` | `carpenter` | `a` | `an` | `a` | 0.6938 | 0.2286 | -1.110 |
| `Someone who teaches and conducts research at universities is` | `professor` | `a` | `a` | `a` | 0.7666 | 0.1865 | -1.414 |
| `Someone who teaches and conducts research at universities is` | `professor` | `a` | `an` | `a` | 0.7714 | 0.1976 | -1.362 |
| `Someone who studies matter and energy is` | `physicist` | `a` | `a` | `a` | 0.4015 | 0.1910 | -0.743 |
| `Someone who studies matter and energy is` | `physicist` | `a` | `an` | `a` | 0.3893 | 0.2752 | -0.347 |
| `Someone who studies chemical reactions and compounds is` | `chemist` | `a` | `a` | `a` | 0.5292 | 0.3169 | -0.513 |
| `Someone who studies chemical reactions and compounds is` | `chemist` | `a` | `an` | `a` | 0.5733 | 0.3302 | -0.552 |
| `Someone who studies and writes about past events is` | `historian` | `a` | `a` | `a` | 0.6867 | 0.2022 | -1.223 |
| `Someone who studies and writes about past events is` | `historian` | `a` | `an` | `a` | 0.7320 | 0.1641 | -1.495 |

## Interpretation Boundary

Low `an` recall would reproduce the paper's majority-class behavioral collapse. It would not by itself show that Gemma would generate an ungrammatical phrase such as `a accountant`, nor that a grammar circuit conceals latent planning. Those require continuation and causal tests.
