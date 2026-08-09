#!/usr/bin/env python3
"""Fixed-b positive control + near-boundary screening.

Validates the fixed-b noun assay with oracle activation patching, screens
same-article-class pairs for near-boundary logit gaps, and reports continuous
metrics with uncertainty (not only top-1 zeros).
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from nnsight import save

from experiments.lib.aan_protocol import (
    article_and_word,
    first_content_token_text,
    vowel_initial,
    write_json,
)
from experiments.lib.core import (
    load_replacement_model,
    logits_for_prompt,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def pre_article_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def word_token_id(tokenizer, word: str) -> int:
    return token_id_for_text(tokenizer, first_content_token_text(tokenizer, word))


def legal_for_article(article: str, word: str) -> bool:
    if not word or article not in {"a", "an"}:
        return False
    return (article == "an") == vowel_initial(word)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[X <= k] for X~Binom(n,p); stable enough for small n."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    # recursive PMF sum
    # start from 0
    log_term = n * math.log(1 - p)  # P(X=0)
    total = math.exp(log_term) if k >= 0 else 0.0
    for i in range(0, k):
        # P(X=i+1)/P(X=i) = (n-i)/(i+1) * p/(1-p)
        log_term += math.log(n - i) - math.log(i + 1) + math.log(p) - math.log(1 - p)
        total += math.exp(log_term)
    return min(1.0, max(0.0, total))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Two-sided 95% Clopper–Pearson upper bound (no scipy)."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    # Find smallest p such that cdf(k; n, p) <= alpha/2  ... actually upper is
    # the p where Pr(X <= k | p) = alpha/2
    target = alpha / 2
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def mean_ci(xs: list[float], alpha: float = 0.05) -> dict[str, float]:
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = sum(xs) / n
    if n == 1:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    # normal approx; fine for workshop reporting of continuous effects
    z = 1.96
    return {"mean": m, "lo": m - z * se, "hi": m + z * se, "n": n}


def mlp_in_at(model, prompt: str, layer: int, position: int) -> torch.Tensor:
    with model.trace(prompt):
        loc = model.get_feature_input_loc(layer)
        act = save(loc.output)
    return act[0, position].detach().float().cpu()


def apply_mlp_in_replace(
    model,
    prompt: str,
    *,
    position: int,
    source_acts: dict[int, torch.Tensor],
    target_acts: dict[int, torch.Tensor],
    mix: float,
) -> torch.Tensor:
    """loc = (1-mix)*source + mix*target at position (oracle patch)."""
    with model.trace(prompt):
        for layer in source_acts:
            loc = model.get_feature_input_loc(int(layer))
            src = source_acts[layer].to(device=loc.output.device, dtype=loc.output.dtype)
            tgt = target_acts[layer].to(device=loc.output.device, dtype=loc.output.dtype)
            blended = (1.0 - mix) * src + mix * tgt
            loc.output[:, position, :] = blended
        logits = save(model.output.logits)
    return logits


def apply_mlp_in_add(
    model,
    prompt: str,
    *,
    position: int,
    directions: dict[int, torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    with model.trace(prompt):
        for layer, direction in directions.items():
            loc = model.get_feature_input_loc(int(layer))
            delta = (alpha * direction).to(device=loc.output.device, dtype=loc.output.dtype)
            loc.output[:, position, :] = loc.output[:, position, :] + delta
        logits = save(model.output.logits)
    return logits


def free_generate(model, prompt: str, max_new_tokens: int) -> str:
    ids: list[int] = []
    current = prompt
    for _ in range(max_new_tokens):
        with model.trace(current):
            logits = save(model.output.logits)
        tid = int(torch.argmax(logits[0, -1]).item())
        ids.append(tid)
        piece = model.tokenizer.decode([tid])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    return model.tokenizer.decode(ids)


def generate_force_b_with_replace(
    model,
    prompt: str,
    *,
    native_article_id: int,
    position: int,
    source_acts: dict[int, torch.Tensor] | None,
    target_acts: dict[int, torch.Tensor] | None,
    mix: float,
    max_new_tokens: int,
) -> str:
    article_text = model.tokenizer.decode([native_article_id])
    current = prompt + article_text
    generated = [native_article_id]
    for _ in range(max(0, max_new_tokens - 1)):
        if source_acts is not None and target_acts is not None and mix > 0:
            logits = apply_mlp_in_replace(
                model,
                current,
                position=position,
                source_acts=source_acts,
                target_acts=target_acts,
                mix=mix,
            )
        else:
            with model.trace(current):
                logits = save(model.output.logits)
        tid = int(torch.argmax(logits[0, -1]).item())
        generated.append(tid)
        piece = model.tokenizer.decode([tid])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    return model.tokenizer.decode(generated)


def noun_logits_replace(
    model,
    prompt_with_article: str,
    *,
    position: int,
    source_acts: dict[int, torch.Tensor] | None,
    target_acts: dict[int, torch.Tensor] | None,
    mix: float,
    token_ids: list[int],
) -> dict[int, float]:
    if source_acts is not None and target_acts is not None and mix > 0:
        logits = apply_mlp_in_replace(
            model,
            prompt_with_article,
            position=position,
            source_acts=source_acts,
            target_acts=target_acts,
            mix=mix,
        )[0, -1].detach().float().cpu()
    else:
        with model.trace(prompt_with_article):
            logits_t = save(model.output.logits)
        logits = logits_t[0, -1].detach().float().cpu()
    return {tid: float(logits[tid]) for tid in token_ids}


def screen_pair(model, pair: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    demo = config["demonstration"]
    source_prompt = f"{demo} {pair['source_sentence']}"
    target_prompt = f"{demo} {pair['target_sentence']}"
    source_id = word_token_id(tokenizer, pair["source_word"])
    target_id = word_token_id(tokenizer, pair["target_word"])

    src_cont = free_generate(model, source_prompt, int(config["max_new_tokens"]))
    tgt_cont = free_generate(model, target_prompt, int(config["max_new_tokens"]))
    src_art, src_word = article_and_word(src_cont)
    tgt_art, tgt_word = article_and_word(tgt_cont)

    # Prefer model's actual articles when legal; else expected.
    article = src_art if src_art in {"a", "an"} else pair["expected_article"]
    article_id = a_id if article == "a" else an_id
    prompt_plus = source_prompt + tokenizer.decode([article_id])
    logits = noun_logits_replace(
        model,
        prompt_plus,
        position=pre_article_pos(tokenizer, source_prompt),
        source_acts=None,
        target_acts=None,
        mix=0.0,
        token_ids=[source_id, target_id],
    )
    gap = logits[target_id] - logits[source_id]
    return {
        "pair_id": pair["id"],
        "source_sentence": pair["source_sentence"],
        "target_sentence": pair["target_sentence"],
        "source_word": pair["source_word"],
        "target_word": pair["target_word"],
        "source_free_continuation": src_cont,
        "target_free_continuation": tgt_cont,
        "source_free_word": src_word,
        "target_free_word": tgt_word,
        "source_matches_listed": src_word == pair["source_word"].lower(),
        "target_matches_listed": tgt_word == pair["target_word"].lower(),
        "article_used": article,
        "baseline_delta_target_minus_source": gap,
        "near_boundary": gap >= float(config["near_boundary_threshold"]),
        "same_article_class": legal_for_article(article, pair["target_word"]),
    }


def evaluate_oracle_patch(
    model,
    pair: dict[str, Any],
    config: dict[str, Any],
    *,
    mix: float,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    demo = config["demonstration"]
    source_prompt = f"{demo} {pair['source_sentence']}"
    target_prompt = f"{demo} {pair['target_sentence']}"
    position = pre_article_pos(tokenizer, source_prompt)
    # Align positions: collect acts at each prompt's own pre-article pos
    pos_s = pre_article_pos(tokenizer, source_prompt)
    pos_t = pre_article_pos(tokenizer, target_prompt)
    layers = [int(x) for x in config["patch_layers"]]

    source_acts = {L: mlp_in_at(model, source_prompt, L, pos_s) for L in layers}
    target_acts = {L: mlp_in_at(model, target_prompt, L, pos_t) for L in layers}

    source_id = word_token_id(tokenizer, pair["source_word"])
    target_id = word_token_id(tokenizer, pair["target_word"])

    src_cont = free_generate(model, source_prompt, int(config["max_new_tokens"]))
    src_art, src_word = article_and_word(src_cont)
    article = src_art if src_art in {"a", "an"} else pair["expected_article"]
    article_id = a_id if article == "a" else an_id
    prompt_plus = source_prompt + tokenizer.decode([article_id])

    logits_off = noun_logits_replace(
        model, prompt_plus, position=position, source_acts=None, target_acts=None, mix=0.0,
        token_ids=[source_id, target_id],
    )
    logits_on = noun_logits_replace(
        model, prompt_plus, position=position,
        source_acts=source_acts, target_acts=target_acts, mix=mix,
        token_ids=[source_id, target_id],
    )
    force_off = generate_force_b_with_replace(
        model, source_prompt, native_article_id=article_id, position=position,
        source_acts=None, target_acts=None, mix=0.0,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    force_on = generate_force_b_with_replace(
        model, source_prompt, native_article_id=article_id, position=position,
        source_acts=source_acts, target_acts=target_acts, mix=mix,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    _, off_word = article_and_word(force_off)
    _, on_word = article_and_word(force_on)

    d_off = logits_off[target_id] - logits_off[source_id]
    d_on = logits_on[target_id] - logits_on[source_id]
    return {
        "condition": "oracle_activation_patch",
        "mix": mix,
        "pair_id": pair["id"],
        "article_used": article,
        "baseline_word": src_word,
        "force_off_word": off_word,
        "force_on_word": on_word,
        "force_on_continuation": force_on,
        "matched_target_on": on_word == pair["target_word"].lower(),
        "noun_switched": bool(on_word) and on_word != src_word,
        "within_class_on": legal_for_article(article, on_word),
        "logit_source_off": logits_off[source_id],
        "logit_target_off": logits_off[target_id],
        "logit_source_on": logits_on[source_id],
        "logit_target_on": logits_on[target_id],
        "delta_target_minus_source_off": d_off,
        "delta_target_minus_source_on": d_on,
        "delta_delta": d_on - d_off,
        "delta_target_logit": logits_on[target_id] - logits_off[target_id],
        "delta_source_logit": logits_on[source_id] - logits_off[source_id],
    }


def evaluate_random_patch(
    model,
    pair: dict[str, Any],
    config: dict[str, Any],
    *,
    mix: float,
    seed: int,
) -> dict[str, Any]:
    """Matched-norm random target activations as control."""
    tokenizer = model.tokenizer
    demo = config["demonstration"]
    source_prompt = f"{demo} {pair['source_sentence']}"
    pos_s = pre_article_pos(tokenizer, source_prompt)
    layers = [int(x) for x in config["patch_layers"]]
    source_acts = {L: mlp_in_at(model, source_prompt, L, pos_s) for L in layers}
    g = torch.Generator().manual_seed(int(seed))
    random_acts: dict[int, torch.Tensor] = {}
    for L, vec in source_acts.items():
        rnd = torch.randn(vec.shape, generator=g, dtype=torch.float32)
        # match target-ish scale: use same norm as source act
        sn = float(vec.norm().item())
        rn = float(rnd.norm().item())
        random_acts[L] = rnd * (sn / max(rn, 1e-8))

    # Reuse oracle evaluator path by temporarily swapping target acts
    # Inline minimal copy:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    source_id = word_token_id(tokenizer, pair["source_word"])
    target_id = word_token_id(tokenizer, pair["target_word"])
    src_cont = free_generate(model, source_prompt, int(config["max_new_tokens"]))
    src_art, src_word = article_and_word(src_cont)
    article = src_art if src_art in {"a", "an"} else pair["expected_article"]
    article_id = a_id if article == "a" else an_id
    prompt_plus = source_prompt + tokenizer.decode([article_id])
    position = pos_s
    logits_off = noun_logits_replace(
        model, prompt_plus, position=position, source_acts=None, target_acts=None, mix=0.0,
        token_ids=[source_id, target_id],
    )
    logits_on = noun_logits_replace(
        model, prompt_plus, position=position,
        source_acts=source_acts, target_acts=random_acts, mix=mix,
        token_ids=[source_id, target_id],
    )
    force_on = generate_force_b_with_replace(
        model, source_prompt, native_article_id=article_id, position=position,
        source_acts=source_acts, target_acts=random_acts, mix=mix,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    _, on_word = article_and_word(force_on)
    d_off = logits_off[target_id] - logits_off[source_id]
    d_on = logits_on[target_id] - logits_on[source_id]
    return {
        "condition": "random_activation_patch",
        "mix": mix,
        "seed": seed,
        "pair_id": pair["id"],
        "matched_target_on": on_word == pair["target_word"].lower(),
        "noun_switched": bool(on_word) and on_word != src_word,
        "delta_delta": d_on - d_off,
        "delta_target_logit": logits_on[target_id] - logits_off[target_id],
        "force_on_word": on_word,
    }


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    t0 = time.time()
    logging.info("Loading model for fixed_b_positive_control")
    model = load_replacement_model(config)

    # --- Phase 1: screen gaps ---
    screens: list[dict[str, Any]] = []
    for pair in config["candidate_pairs"]:
        row = screen_pair(model, pair, config)
        screens.append(row)
        logging.info(
            "screen %s gap=%.3f near=%s src_free=%r tgt_free=%r",
            row["pair_id"],
            row["baseline_delta_target_minus_source"],
            row["near_boundary"],
            row["source_free_continuation"].strip(),
            row["target_free_continuation"].strip(),
        )

    screens_sorted = sorted(screens, key=lambda r: -r["baseline_delta_target_minus_source"])
    # Prefer pairs where both free gens match listed nouns and same article class
    usable = [
        r for r in screens_sorted
        if r["same_article_class"]
        and r["source_matches_listed"]
        and r["target_matches_listed"]
    ]
    if not usable:
        usable = [r for r in screens_sorted if r["same_article_class"]]
    near = [r for r in usable if r["near_boundary"]]
    # Always keep top-4 closest gaps for oracle tests even if not near-boundary
    top_ids = {r["pair_id"] for r in usable[:4]}
    near_ids = {r["pair_id"] for r in near} | top_ids
    pair_by_id = {p["id"]: p for p in config["candidate_pairs"]}
    test_pairs = [pair_by_id[i] for i in near_ids if i in pair_by_id]

    # --- Phase 2: oracle patches ---
    oracle_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for pair in test_pairs:
        for mix in config["patch_mixes"]:
            row = evaluate_oracle_patch(model, pair, config, mix=float(mix))
            oracle_rows.append(row)
            logging.info(
                "oracle %s mix=%.2f ΔΔ=%.3f on=%r matched=%s",
                pair["id"], mix, row["delta_delta"], row["force_on_word"], row["matched_target_on"],
            )
            for seed in config["control_seeds"]:
                rnd = evaluate_random_patch(
                    model, pair, config, mix=float(mix), seed=int(seed)
                )
                random_rows.append(rnd)

    # Summaries
    def summarize_oracle(rows: list[dict[str, Any]], mix: float | None = None) -> dict[str, Any]:
        sub = [r for r in rows if mix is None or abs(r["mix"] - mix) < 1e-9]
        n = len(sub)
        k_match = sum(1 for r in sub if r["matched_target_on"])
        k_switch = sum(1 for r in sub if r["noun_switched"])
        dd = [float(r["delta_delta"]) for r in sub]
        dt = [float(r["delta_target_logit"]) for r in sub]
        try:
            upper_match = clopper_pearson_upper(k_match, n) if n else float("nan")
            upper_switch = clopper_pearson_upper(k_switch, n) if n else float("nan")
        except Exception:
            # scipy may be missing; fallback
            upper_match = float("nan")
            upper_switch = float("nan")
        return {
            "n": n,
            "match_rate": k_match / n if n else float("nan"),
            "switch_rate": k_switch / n if n else float("nan"),
            "match_rate_cp_upper95": upper_match,
            "switch_rate_cp_upper95": upper_switch,
            "delta_delta": mean_ci(dd),
            "delta_target_logit": mean_ci(dt),
        }

    by_mix = {float(m): summarize_oracle(oracle_rows, float(m)) for m in config["patch_mixes"]}
    rnd_by_mix: dict[str, Any] = {}
    for m in config["patch_mixes"]:
        sub = [r for r in random_rows if abs(r["mix"] - float(m)) < 1e-9]
        dd = [float(r["delta_delta"]) for r in sub]
        k_match = sum(1 for r in sub if r["matched_target_on"])
        n = len(sub)
        rnd_by_mix[str(m)] = {
            "n": n,
            "match_rate": k_match / n if n else float("nan"),
            "delta_delta": mean_ci(dd),
        }

    best_mix = max(by_mix.items(), key=lambda kv: kv[1]["delta_delta"]["mean"])[0]
    assay_validated = by_mix[best_mix]["delta_delta"]["mean"] > 0.5 or by_mix[best_mix]["match_rate"] > 0.0

    summary = {
        "experiment": config["experiment_name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
        "n_candidates_screened": len(screens),
        "n_near_boundary": sum(1 for r in screens if r["near_boundary"]),
        "n_usable_matched_pairs": len(usable),
        "test_pair_ids": [p["id"] for p in test_pairs],
        "closest_gaps": [
            {
                "pair_id": r["pair_id"],
                "gap": r["baseline_delta_target_minus_source"],
                "near_boundary": r["near_boundary"],
            }
            for r in screens_sorted[:8]
        ],
        "oracle_by_mix": {str(k): v for k, v in by_mix.items()},
        "random_by_mix": rnd_by_mix,
        "best_mix": best_mix,
        "assay_validated": assay_validated,
        "interpretation": (
            "Oracle MLP-in activation patching under fixed native article "
            + (
                "moves target-minus-source logits and/or noun identity; "
                "fixed-b protocol can transmit noun-level effects."
                if assay_validated
                else "does not clearly move nouns; fixed-b assay may be insensitive "
                "or model may lack separable noun control at these sites."
            )
        ),
    }

    write_json(RESULTS_DIR / "screens.json", screens_sorted)
    write_json(RESULTS_DIR / "oracle_rows.json", oracle_rows)
    write_json(RESULTS_DIR / "random_rows.json", random_rows)
    write_json(RESULTS_DIR / "summary.json", summary)

    lines = [
        "# Fixed-b positive control + near-boundary screen",
        "",
        f"- Candidates screened: {summary['n_candidates_screened']}",
        f"- Near-boundary (gap ≥ {config['near_boundary_threshold']}): {summary['n_near_boundary']}",
        f"- Test pairs: {', '.join(summary['test_pair_ids'])}",
        f"- Best mix: {best_mix}",
        f"- Assay validated: {assay_validated}",
        "",
        "## Closest baseline gaps (target − source under fixed b)",
        "",
    ]
    for r in screens_sorted[:8]:
        lines.append(
            f"- `{r['pair_id']}`: gap={r['baseline_delta_target_minus_source']:.3f} "
            f"near={r['near_boundary']} free_src={r['source_free_continuation'].strip()!r} "
            f"free_tgt={r['target_free_continuation'].strip()!r}"
        )
    lines += ["", "## Oracle patch by mix", ""]
    for m, s in by_mix.items():
        lines.append(
            f"- mix={m}: n={s['n']} match={s['match_rate']:.2f} "
            f"(CP95 upper={s['match_rate_cp_upper95']:.2f}) "
            f"ΔΔ mean={s['delta_delta']['mean']:.3f} "
            f"[{s['delta_delta']['lo']:.3f},{s['delta_delta']['hi']:.3f}]"
        )
    lines += ["", "## Random patch controls", ""]
    for m, s in rnd_by_mix.items():
        lines.append(
            f"- mix={m}: n={s['n']} match={s['match_rate']:.2f} "
            f"ΔΔ mean={s['delta_delta']['mean']:.3f} "
            f"[{s['delta_delta']['lo']:.3f},{s['delta_delta']['hi']:.3f}]"
        )
    lines += ["", summary["interpretation"], ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))
    logging.info("Done. assay_validated=%s elapsed=%.1fs", assay_validated, time.time() - t0)


if __name__ == "__main__":
    main()
