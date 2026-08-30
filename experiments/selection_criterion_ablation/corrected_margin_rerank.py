#!/usr/bin/env python3
"""Re-rank the frozen 270M selection graphs by positive (an-a) and future effects."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from circuit_tracer.graph import Graph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.lib.aan_protocol import load_tokenizer, slugify, token_id_for_text, write_json
from experiments.lib.core import feature_effect_map

EXP = Path(__file__).resolve().parent
GRAPHS = EXP / "results/graphs"
CONFIG = json.loads((EXP / "config.json").read_text())


def main():
    tok = load_tokenizer(CONFIG)
    a_id, an_id = token_id_for_text(tok, " a"), token_id_for_text(tok, " an")
    stats = defaultdict(lambda: {"count": 0, "margin": 0.0, "future": 0.0})
    for sentence in CONFIG["selection_sentences"]:
        slug = slugify(sentence)
        meta = json.loads((GRAPHS / f"{slug}__meta.json").read_text())
        content_id = token_id_for_text(tok, meta["content_token_text"])
        pos = len(tok(meta["article_prompt"], add_special_tokens=True).input_ids) - 1
        article = Graph.from_pt(str(GRAPHS / f"{slug}__article.pt"))
        future = Graph.from_pt(str(GRAPHS / f"{slug}__future.pt"))
        a, an, fut = feature_effect_map(article, a_id), feature_effect_map(article, an_id), feature_effect_map(future, content_id)
        for key in set(a) & set(an) & set(fut):
            layer, position, feature_idx = key
            if position != pos:
                continue
            margin = an[key]["direct_effect"] - a[key]["direct_effect"]
            target = fut[key]["direct_effect"]
            if margin > 0 and target > 0:
                row = stats[(layer, feature_idx)]
                row["count"] += 1; row["margin"] += margin; row["future"] += target
    ranked = []
    for (layer, feature_idx), row in stats.items():
        count = row["count"]
        margin, future = row["margin"] / count, row["future"] / count
        ranked.append({"layer": layer, "feature_idx": feature_idx, "prompt_count": count,
                       "mean_margin_attribution": margin, "mean_future_attribution": future,
                       "score": min(margin, future)})
    ranked.sort(key=lambda r: (r["prompt_count"], r["score"]), reverse=True)
    canonical = {(12,6229),(10,2930),(13,10231),(11,793)}
    eligible = [r for r in ranked if r["prompt_count"] >= CONFIG["min_selection_prompt_count"]]
    top4 = {(r["layer"], r["feature_idx"]) for r in eligible[:4]}
    out = {"criterion": "positive direct attribution to (an-a) and future target; rank by minimum mean effect",
           "top4": eligible[:4],
           "canonical_s1": sorted(canonical), "top4_overlap": len(top4 & canonical), "ranked": ranked[:50]}
    write_json(EXP / "results/corrected_margin_rerank.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
