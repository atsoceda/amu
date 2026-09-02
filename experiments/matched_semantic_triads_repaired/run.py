#!/usr/bin/env python3
"""Route-blind screening and paired assay for repaired neutral semantic triads."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, cosine, first_id, stats
from experiments.lib.aan_protocol import token_id_for_text

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def article_metrics(logits: torch.Tensor, article_ids: dict[str, int]) -> dict:
    probs = torch.softmax(logits.float(), -1)
    top_id = int(logits.argmax())
    return {
        "top_token": top_id,
        "top_text": None,
        "article_mass": float(probs[list(article_ids.values())].sum()),
        "article_top": top_id in article_ids.values(),
        "a_logit": float(logits[article_ids["a"]]),
        "an_logit": float(logits[article_ids["an"]]),
    }


def capture(rm: ResidualModel, text: str) -> tuple[int, list[torch.Tensor]]:
    position = len(rm.tokenizer(text, add_special_tokens=True).input_ids) - 1
    return position, rm.states(text, position)


def screen(cfg: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    article_ids = {a: token_id_for_text(tok, f" {a}") for a in ("a", "an")}
    rows = load(RESULTS / "screen_rows.json") if (RESULTS / "screen_rows.json").exists() else []
    complete = {row["triad_id"] for row in rows}
    for index, triad in enumerate(cfg["triads"], 1):
        if triad["id"] in complete:
            print(f"screen checkpoint {index}/{len(cfg['triads'])}: {triad['id']}", flush=True)
            continue
        neutral = cfg["neutral_template"].format(definition=triad["definition"])
        neutral_position, neutral_states = capture(rm, neutral)
        lexical_ids = {role: first_id(tok, triad[f"{role}_word"]) for role in ("source", "within", "cross")}
        single_token = {
            role: tok.decode([lexical_ids[role]]).strip().lower() == triad[f"{role}_word"].lower()
            for role in lexical_ids
        }
        donor_states = {}
        for role in ("source", "within", "cross"):
            donor = cfg["donor_template"].format(word=triad[f"{role}_word"], definition=triad["definition"])
            _, states = capture(rm, donor)
            donor_states[role] = states[cfg["fixed_layer"]]
        off_logits = rm.logits(neutral)
        off_support = article_metrics(off_logits, article_ids)
        off_support["top_text"] = tok.decode([off_support["top_token"]]).strip()
        arms = {}
        for role in ("within", "cross"):
            article = triad[f"{role}_article"]
            source_id, target_id = lexical_ids["source"], lexical_ids[role]
            delta = donor_states[role] - donor_states["source"]
            replacement = neutral_states[cfg["fixed_layer"]] + cfg["primary_strength"] * delta
            patch = (cfg["fixed_layer"], neutral_position, replacement)
            baseline = stats(rm.logits(neutral + tok.decode([article_ids[article]])), source_id, target_id)
            treated = stats(rm.logits(neutral + tok.decode([article_ids[article]]), patch), source_id, target_id)
            on_logits = rm.logits(neutral, patch)
            on_support = article_metrics(on_logits, article_ids)
            on_support["top_text"] = tok.decode([on_support["top_token"]]).strip()
            arms[role] = {
                "fixed_target_article": article,
                "local_target_minus_source_effect": treated["target_minus_source"] - baseline["target_minus_source"],
                "off_support": off_support,
                "on_support": on_support,
            }
        admissible = (
            all(single_token.values())
            and off_support["article_top"]
            and off_support["article_mass"] >= cfg["minimum_article_mass"]
            and all(arms[role]["local_target_minus_source_effect"] > 0 for role in ("within", "cross"))
            and all(arms[role]["on_support"]["article_top"] for role in ("within", "cross"))
            and all(arms[role]["on_support"]["article_mass"] >= cfg["minimum_article_mass"] for role in ("within", "cross"))
        )
        rows.append({
            "triad_id": triad["id"],
            "definition": triad["definition"],
            "words": {role: triad[f"{role}_word"] for role in ("source", "within", "cross")},
            "articles": {role: triad[f"{role}_article"] for role in ("source", "within", "cross")},
            "single_token": single_token,
            "arms": arms,
            "admissible": admissible,
            "route_outcomes_computed_during_screening": False,
        })
        atomic_json(RESULTS / "screen_rows.json", rows)
        print(f"screened {index}/{len(cfg['triads'])}: {triad['id']} admissible={admissible}", flush=True)
    admissible_ids = [row["triad_id"] for row in rows if row["admissible"]]
    summary = {
        "experiment": cfg["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screening_rule": cfg["screening_rule"],
        "candidate_triads": len(rows),
        "admissible_triads": len(admissible_ids),
        "minimum_required": cfg["minimum_admissible_triads"],
        "proceed_to_assay": len(admissible_ids) >= cfg["minimum_admissible_triads"],
        "admissible_ids": admissible_ids,
    }
    atomic_json(RESULTS / "screen_summary.json", summary)
    print(json.dumps(summary, indent=2))


def vector(effect: torch.Tensor, source_id: int, target_id: int) -> dict:
    desired = torch.zeros_like(effect)
    desired[target_id] = 1
    desired[source_id] = -1
    return {
        "tv": float(.5 * effect.abs().sum()),
        "target_minus_source": float(effect[target_id] - effect[source_id]),
        "desired_cosine": cosine(effect, desired),
    }


def bootstrap(values: list[float], seed: int, resamples: int) -> dict:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": draws[math.floor(.025 * (resamples - 1))],
        "hi": draws[math.ceil(.975 * (resamples - 1))],
        "method": "paired semantic-triad bootstrap",
        "resamples": resamples,
    }


def exact_sign_flip(values: list[float]) -> dict:
    observed = sum(values) / len(values)
    null = []
    for signs in itertools.product((-1, 1), repeat=len(values)):
        null.append(sum(s * v for s, v in zip(signs, values)) / len(values))
    return {
        "observed": observed,
        "one_sided_p": sum(v >= observed - 1e-12 for v in null) / len(null),
        "two_sided_p": sum(abs(v) >= abs(observed) - 1e-12 for v in null) / len(null),
        "assignments": len(null),
    }


def assay(cfg: dict) -> None:
    screen_summary = load(RESULTS / "screen_summary.json")
    if not screen_summary["proceed_to_assay"]:
        raise RuntimeError(f"Only {screen_summary['admissible_triads']} admissible triads; refusing to weaken the frozen gate")
    selected = set(screen_summary["admissible_ids"])
    triads = [t for t in cfg["triads"] if t["id"] in selected]
    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    article_ids = {a: token_id_for_text(tok, f" {a}") for a in ("a", "an")}
    rows = load(RESULTS / "rows.json") if (RESULTS / "rows.json").exists() else []
    completed = {(r["triad_id"], r["arm"], r["strength"]) for r in rows}
    for index, triad in enumerate(triads, 1):
        neutral = cfg["neutral_template"].format(definition=triad["definition"])
        neutral_position, neutral_states = capture(rm, neutral)
        lexical_ids = {role: first_id(tok, triad[f"{role}_word"]) for role in ("source", "within", "cross")}
        donor_states = {}
        for role in ("source", "within", "cross"):
            donor = cfg["donor_template"].format(word=triad[f"{role}_word"], definition=triad["definition"])
            _, states = capture(rm, donor)
            donor_states[role] = states[cfg["fixed_layer"]]
        off_article = rm.logits(neutral)
        off_branches = {a: rm.logits(neutral + tok.decode([aid])) for a, aid in article_ids.items()}
        for role in ("within", "cross"):
            delta = donor_states[role] - donor_states["source"]
            for strength in cfg["strengths"]:
                key = (triad["id"], role, strength)
                if key in completed:
                    continue
                patch = (cfg["fixed_layer"], neutral_position, neutral_states[cfg["fixed_layer"]] + strength * delta)
                on_article = rm.logits(neutral, patch)
                on_branches = {a: rm.logits(neutral + tok.decode([aid]), patch) for a, aid in article_ids.items()}
                stochastic = {}
                for tau in cfg["temperatures"]:
                    q0 = torch.softmax(off_article[list(article_ids.values())].float() / tau, -1)
                    q1 = torch.softmax(on_article[list(article_ids.values())].float() / tau, -1)
                    y0 = {a: torch.softmax(v.float(), -1) for a, v in off_branches.items()}
                    y1 = {a: torch.softmax(v.float(), -1) for a, v in on_branches.items()}
                    off = q0[0] * y0["a"] + q0[1] * y0["an"]
                    public = q1[0] * y0["a"] + q1[1] * y0["an"]
                    treated = q1[0] * y1["a"] + q1[1] * y1["an"]
                    stochastic[str(tau)] = {
                        "public": vector(public - off, lexical_ids["source"], lexical_ids[role]),
                        "private": vector(treated - public, lexical_ids["source"], lexical_ids[role]),
                        "total": vector(treated - off, lexical_ids["source"], lexical_ids[role]),
                        "off_article_mass": float(torch.softmax(off_article.float() / tau, -1)[list(article_ids.values())].sum()),
                        "on_article_mass": float(torch.softmax(on_article.float() / tau, -1)[list(article_ids.values())].sum()),
                        "reconstruction_l1": float((treated - off - (public - off) - (treated - public)).abs().sum()),
                    }
                target_article = triad[f"{role}_article"]
                base = stats(off_branches[target_article], lexical_ids["source"], lexical_ids[role])
                fixed = stats(on_branches[target_article], lexical_ids["source"], lexical_ids[role])
                rows.append({
                    "triad_id": triad["id"], "arm": role, "strength": strength,
                    "source_word": triad["source_word"], "target_word": triad[f"{role}_word"],
                    "source_article": triad["source_article"], "target_article": target_article,
                    "layer": cfg["fixed_layer"],
                    "fixed_target_article_effect": fixed["target_minus_source"] - base["target_minus_source"],
                    "greedy_article_off": tok.decode([int(off_article.argmax())]).strip(),
                    "greedy_article_on": tok.decode([int(on_article.argmax())]).strip(),
                    "stochastic": stochastic,
                })
                atomic_json(RESULTS / "rows.json", rows)
        atomic_json(RESULTS / "progress.json", {"completed_triads": index, "total_triads": len(triads), "last_triad": triad["id"]})
        print(f"assayed and checkpointed {index}/{len(triads)}: {triad['id']}", flush=True)
    primary = [r for r in rows if r["strength"] == cfg["primary_strength"]]
    paired = []
    for triad in triads:
        arm_rows = {r["arm"]: r for r in primary if r["triad_id"] == triad["id"]}
        contrasts = {}
        for role in ("within", "cross"):
            cell = arm_rows[role]["stochastic"][str(cfg["primary_temperature"])]
            contrasts[role] = cell["public"]["target_minus_source"] - cell["private"]["target_minus_source"]
        paired.append({
            "triad_id": triad["id"],
            "within_route_contrast": contrasts["within"],
            "cross_route_contrast": contrasts["cross"],
            "paired_interaction": contrasts["cross"] - contrasts["within"],
            "predicted_double_dissociation": contrasts["cross"] > 0 and contrasts["within"] < 0,
        })
    interactions = [p["paired_interaction"] for p in paired]
    summary = {
        "experiment": cfg["experiment_name"],
        "primary_setting": {"layer": cfg["fixed_layer"], "strength": cfg["primary_strength"], "temperature": cfg["primary_temperature"]},
        "n_triads": len(paired),
        "paired_rows": paired,
        "paired_interaction": bootstrap(interactions, cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
        "paired_interaction_exact_sign_flip": exact_sign_flip(interactions),
        "double_dissociation_n": sum(p["predicted_double_dissociation"] for p in paired),
        "route_outcomes_used_for_selection": False,
    }
    atomic_json(RESULTS / "summary.json", summary)
    atomic_json(RESULTS / "progress.json", {"completed_triads": len(triads), "total_triads": len(triads), "status": "complete"})
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("screen", "assay"))
    args = parser.parse_args()
    cfg = load(EXP / "config.json")
    if args.phase == "screen":
        screen(cfg)
    else:
        assay(cfg)


if __name__ == "__main__":
    main()
