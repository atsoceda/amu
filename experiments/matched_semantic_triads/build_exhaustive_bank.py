#!/usr/bin/env python3
"""Build an outcome-blind triad bank from the pre-existing synonym lexicon."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/correctness_preserving_aan/config.json"
OUTPUT = Path(__file__).resolve().parent / "config_batch3.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    families = defaultdict(lambda: {"definitions": [], "words": {}})
    for pair in source["pairs"]:
        family = pair["id"].split("_", 1)[0]
        families[family]["definitions"].append(pair["definition"])
        families[family]["words"][pair["source_word"]] = pair["source_article"]
        families[family]["words"][pair["target_word"]] = pair["target_article"]

    triads = []
    for family, record in sorted(families.items()):
        words = record["words"]
        if len(set(words.values())) < 2:
            continue
        definition = Counter(record["definitions"]).most_common(1)[0][0]
        for source_word, source_article in sorted(words.items()):
            same = sorted(w for w, a in words.items() if a == source_article and w != source_word)
            cross = sorted(w for w, a in words.items() if a != source_article)
            for same_word in same:
                for cross_word in cross:
                    triads.append({
                        "id": f"{family}__{source_word}__{same_word}__{cross_word}",
                        "family": family,
                        "definition": definition,
                        "source_word": source_word,
                        "source_article": source_article,
                            "within_word": same_word,
                            "within_article": source_article,
                        "cross_word": cross_word,
                        "cross_article": words[cross_word],
                    })

    config = {
        "experiment_name": "matched_semantic_triads_exhaustive_discovery_batch3",
        "frozen_before_screening": "2026-08-31T14:05:00+09:00",
        "construction": (
            "Outcome-blind exhaustive enumeration within semantic-family labels and lexical "
            "alternatives frozen in correctness_preserving_aan/config.json. The modal existing "
            "definition is selected deterministically per family; no new logits or causal "
            "outcomes enter construction."
        ),
        "model": source["models"]["gemma_1b"]["model"],
        "model_snapshot": source["models"]["gemma_1b"]["model_snapshot"],
        "dtype": source["dtype"],
        "prompt_template": source["source_template"],
        "screening_rule": (
            "Identical to frozen batches 1-2: all nouns are single-token readouts; each prompt "
            "preserves its intended article; the source prompt favors source over both targets; "
            "each target prompt favors target over source. At most one triad per family may enter "
            "inference, selected by the frozen behavioral-only score."
        ),
        "family_selection_score": (
            "maximize the minimum of the four positive lexical logit margins: source over same, "
            "source over cross, same over source, and cross over source"
        ),
        "fixed_layer": 18,
        "primary_strength": 1.0,
        "robustness_strengths": [0.5, 1.5],
        "primary_temperature": 1.0,
        "temperatures": [0.1, 0.25, 0.5, 1.0],
        "triads": triads,
    }
    OUTPUT.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(triads)} candidates across {len(set(t['family'] for t in triads))} families")


if __name__ == "__main__":
    main()
