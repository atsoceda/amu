#!/usr/bin/env python3
"""Synthetic ground-truth validation of generated-token mediation estimands.

Two known autoregressive mechanisms share the same vocabulary and prompts:

1. mediated  — intervention changes only article logits; nouns depend on the
   generated/forced article (and cue), not on a persistent plan bit.
2. direct    — intervention also injects a persistent plan bias that continues
   to move the noun under any forced article / filler prefix.

The assay must return near-zero token-clamped residual control for (1) and
large residual control for (2), including a k-token residual-control curve.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import write_json
from experiments.lib.core import setup_file_logging
from experiments.lib.mediation_estimands import (
    controlled_direct_effects,
    distribution_distance,
    logit_contrast,
    mean_and_interval,
    mediation_decomposition,
    residual_control_curve,
)


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


ARTICLES = ("a", "an")
VOWEL_NOUNS = {
    "aviator",
    "attorney",
    "oculist",
    "astronomer",
    "analyst",
    "agriculturist",
    "educator",
    "admiral",
}


class SyntheticVocab:
    def __init__(self, cues: list[dict[str, str]], fillers: list[str]) -> None:
        tokens = ["<bos>", "<cue>"]
        tokens.extend(ARTICLES)
        tokens.extend(fillers)
        nouns: list[str] = []
        for cue in cues:
            nouns.append(cue["source"])
            nouns.append(cue["target"])
        # Stable unique order.
        for noun in nouns:
            if noun not in tokens:
                tokens.append(noun)
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = tokens
        self.article_ids = {a: self.token_to_id[a] for a in ARTICLES}
        self.filler_ids = [self.token_to_id[f] for f in fillers]
        self.noun_ids = [
            self.token_to_id[n]
            for n in tokens
            if n not in {"<bos>", "<cue>", *ARTICLES, *fillers}
        ]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode_id(self, token_id: int) -> str:
        return self.id_to_token[int(token_id)]


class GroundTruthAR:
    """Explicit next-token scorer with known mediation structure."""

    def __init__(
        self,
        vocab: SyntheticVocab,
        *,
        mechanism: str,
        article_bias_on: float,
        direct_plan_bias: float,
        mediated_plan_bias: float,
        noun_article_coupling: float,
        filler_bias: float,
        temperature: float = 1.0,
    ) -> None:
        if mechanism not in {"mediated", "direct"}:
            raise ValueError(mechanism)
        self.vocab = vocab
        self.mechanism = mechanism
        self.article_bias_on = float(article_bias_on)
        self.plan_bias = (
            float(direct_plan_bias) if mechanism == "direct" else float(mediated_plan_bias)
        )
        self.noun_article_coupling = float(noun_article_coupling)
        self.filler_bias = float(filler_bias)
        self.temperature = float(temperature)

    def _blank(self) -> torch.Tensor:
        return torch.full((len(self.vocab),), -8.0, dtype=torch.float64)

    def next_logits(
        self,
        prefix: list[str],
        *,
        cue: dict[str, str],
        intervention: int,
    ) -> torch.Tensor:
        logits = self._blank()
        stage = self._stage(prefix)
        if stage == "article":
            logits[self.vocab.article_ids["a"]] = 1.5
            logits[self.vocab.article_ids["an"]] = -0.5
            if intervention:
                logits[self.vocab.article_ids["an"]] += self.article_bias_on
                logits[self.vocab.article_ids["a"]] -= 0.5 * self.article_bias_on
            return logits / self.temperature

        if stage == "filler":
            for filler_id in self.vocab.filler_ids:
                logits[filler_id] = self.filler_bias
            # Weak uniform leakage so distances are well-defined.
            for noun_id in self.vocab.noun_ids:
                logits[noun_id] = -1.0
            return logits / self.temperature

        # Noun stage.
        source_id = self.vocab.token_to_id[cue["source"]]
        target_id = self.vocab.token_to_id[cue["target"]]
        article = self._last_article(prefix)
        for noun_id in self.vocab.noun_ids:
            logits[noun_id] = -1.0
        logits[source_id] = 0.5
        logits[target_id] = 0.5
        if article == "a":
            logits[source_id] += self.noun_article_coupling
            logits[target_id] -= 0.5 * self.noun_article_coupling
        elif article == "an":
            logits[target_id] += self.noun_article_coupling
            logits[source_id] -= 0.5 * self.noun_article_coupling
        if intervention and self.plan_bias != 0.0:
            logits[target_id] += self.plan_bias
            logits[source_id] -= 0.25 * self.plan_bias
        return logits / self.temperature

    def greedy_token(
        self,
        prefix: list[str],
        *,
        cue: dict[str, str],
        intervention: int,
    ) -> str:
        logits = self.next_logits(prefix, cue=cue, intervention=intervention)
        return self.vocab.decode_id(int(torch.argmax(logits).item()))

    def generate_until_noun(
        self,
        *,
        cue: dict[str, str],
        intervention: int,
        max_fillers: int = 0,
    ) -> list[str]:
        prefix = ["<bos>", "<cue>"]
        # Article.
        prefix.append(self.greedy_token(prefix, cue=cue, intervention=intervention))
        for _ in range(max_fillers):
            nxt = self.greedy_token(prefix, cue=cue, intervention=intervention)
            if nxt in ARTICLES:
                break
            if nxt in {cue["source"], cue["target"]} or nxt in VOWEL_NOUNS:
                break
            if nxt not in self.vocab.token_to_id:
                break
            # Stop if model prefers noun over filler.
            if nxt not in [self.vocab.decode_id(i) for i in self.vocab.filler_ids]:
                break
            prefix.append(nxt)
        return prefix

    @staticmethod
    def _last_article(prefix: list[str]) -> str | None:
        for tok in reversed(prefix):
            if tok in ARTICLES:
                return tok
        return None

    def _stage(self, prefix: list[str]) -> str:
        if self._last_article(prefix) is None:
            return "article"
        # After article, if the latest tokens are only fillers (or just article),
        # caller decides whether this query is for filler or noun by convention:
        # we treat any request after article as noun unless the prefix ends with
        # article and we explicitly ask for filler via trailing marker.
        if prefix and prefix[-1] == "<ask_filler>":
            return "filler"
        # If prefix ends with article or filler, noun is the default outcome
        # position used by the factorial / residual curve.
        return "noun"


def force_prefix(
    article: str,
    fillers: list[str],
) -> list[str]:
    return ["<bos>", "<cue>", article, *fillers]


def evaluate_prompt(
    model: GroundTruthAR,
    cue: dict[str, str],
    *,
    fillers: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    vocab = model.vocab
    source_id = vocab.token_to_id[cue["source"]]
    target_id = vocab.token_to_id[cue["target"]]

    baseline_prefix = model.generate_until_noun(cue=cue, intervention=0)
    treated_prefix = model.generate_until_noun(cue=cue, intervention=1)
    b0 = model._last_article(baseline_prefix)
    b1 = model._last_article(treated_prefix)
    assert b0 in ARTICLES and b1 in ARTICLES

    baseline_free = model.next_logits(baseline_prefix, cue=cue, intervention=0)
    treated_free = model.next_logits(treated_prefix, cue=cue, intervention=1)
    article_only = model.next_logits(force_prefix(b1, []), cue=cue, intervention=0)
    residual_on = model.next_logits(force_prefix(b1, []), cue=cue, intervention=1)
    residual_off = model.next_logits(force_prefix(b1, []), cue=cue, intervention=0)

    decomp = mediation_decomposition(
        baseline_free=baseline_free,
        treated_free=treated_free,
        article_only=article_only,
        residual_on=residual_on,
        residual_off=residual_off,
    )

    on_a = model.next_logits(force_prefix("a", []), cue=cue, intervention=1)
    off_a = model.next_logits(force_prefix("a", []), cue=cue, intervention=0)
    on_an = model.next_logits(force_prefix("an", []), cue=cue, intervention=1)
    off_an = model.next_logits(force_prefix("an", []), cue=cue, intervention=0)
    cdes = controlled_direct_effects(
        on_a=on_a,
        off_a=off_a,
        on_an=on_an,
        off_an=off_an,
        target_id=target_id,
        source_id=source_id,
    )

    def curve_for(article: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for k in k_values:
            forced_fillers = fillers[:k]
            # k=0 clamps only the article; k>0 adds the first k fillers.
            on_logits = model.next_logits(
                force_prefix(article, forced_fillers), cue=cue, intervention=1
            )
            off_logits = model.next_logits(
                force_prefix(article, forced_fillers), cue=cue, intervention=0
            )
            dist = distribution_distance(on_logits, off_logits)
            points.append(
                {
                    "k": int(k),
                    "forced_article": article,
                    "forced_prefix": [article, *forced_fillers],
                    "residual_tv": dist["tv"],
                    "residual_js": dist["js"],
                    "target_minus_source_delta": logit_contrast(
                        on_logits, target_id, source_id
                    )
                    - logit_contrast(off_logits, target_id, source_id),
                }
            )
        return residual_control_curve(points)

    # Primary k-curve uses force-a: under an, article coupling can saturate the
    # noun distribution and hide residual TV even when logit CDE is large.
    curve_a = curve_for("a")
    curve_an = curve_for("an")

    return {
        "cue": cue["id"],
        "mechanism": model.mechanism,
        "baseline_article": b0,
        "treated_article": b1,
        "article_switched": b0 != b1,
        "baseline_top1": vocab.decode_id(int(torch.argmax(baseline_free).item())),
        "treated_top1": vocab.decode_id(int(torch.argmax(treated_free).item())),
        "article_only_top1": vocab.decode_id(int(torch.argmax(article_only).item())),
        "decomposition": decomp,
        "controlled_direct_effects": cdes,
        "residual_control_curve": curve_a,
        "residual_control_curve_by_article": {"a": curve_a, "an": curve_an},
        "total_target_source_delta": logit_contrast(treated_free, target_id, source_id)
        - logit_contrast(baseline_free, target_id, source_id),
        "article_only_target_source_delta": logit_contrast(article_only, target_id, source_id)
        - logit_contrast(baseline_free, target_id, source_id),
        "residual_target_source_delta": logit_contrast(residual_on, target_id, source_id)
        - logit_contrast(residual_off, target_id, source_id),
    }


def summarize_mechanism(
    rows: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    total_tv = [float(r["decomposition"]["total"]["tv"]) for r in rows]
    article_tv = [float(r["decomposition"]["mediator_only"]["tv"]) for r in rows]
    residual_tv = [float(r["decomposition"]["residual"]["tv"]) for r in rows]
    ratio = [float(r["decomposition"]["residual_over_total_tv"]) for r in rows]
    repro = [
        1.0 if r["decomposition"]["top1_mediator_reproduces_treated"] else 0.0 for r in rows
    ]
    interaction = [
        float(r["controlled_direct_effects"]["interaction_an_minus_a"]["target_minus_source_delta"])
        for r in rows
    ]
    cde_a = [
        float(r["controlled_direct_effects"]["cde_a"]["target_minus_source_delta"]) for r in rows
    ]
    cde_an = [
        float(r["controlled_direct_effects"]["cde_an"]["target_minus_source_delta"]) for r in rows
    ]
    cde_a_tv = [float(r["controlled_direct_effects"]["cde_a"]["tv"]) for r in rows]
    cde_an_tv = [float(r["controlled_direct_effects"]["cde_an"]["tv"]) for r in rows]

    def summarize_curve(article: str, offset: int) -> list[dict[str, Any]]:
        ks = sorted(
            {
                int(pt["k"])
                for r in rows
                for pt in r["residual_control_curve_by_article"][article]
            }
        )
        out = []
        for k in ks:
            tvs = []
            deltas = []
            for r in rows:
                for pt in r["residual_control_curve_by_article"][article]:
                    if int(pt["k"]) == k:
                        tvs.append(float(pt["residual_tv"]))
                        deltas.append(float(pt["target_minus_source_delta"]))
            out.append(
                {
                    "k": k,
                    "forced_article": article,
                    "residual_tv": mean_and_interval(
                        tvs, n_boot=n_boot, seed=seed + offset + k
                    ),
                    "target_minus_source_delta": mean_and_interval(
                        deltas, n_boot=n_boot, seed=seed + offset + 100 + k
                    ),
                }
            )
        return out

    curve_a = summarize_curve("a", 0)
    curve_an = summarize_curve("an", 200)

    return {
        "n": len(rows),
        "article_switch_rate": sum(1 for r in rows if r["article_switched"]) / max(len(rows), 1),
        "total_tv": mean_and_interval(total_tv, n_boot=n_boot, seed=seed),
        "article_only_tv": mean_and_interval(article_tv, n_boot=n_boot, seed=seed + 1),
        "residual_tv": mean_and_interval(residual_tv, n_boot=n_boot, seed=seed + 2),
        "residual_over_total_tv": mean_and_interval(ratio, n_boot=n_boot, seed=seed + 3),
        "top1_article_reproduction": mean_and_interval(repro, n_boot=0, seed=seed),
        "cde_a_delta": mean_and_interval(cde_a, n_boot=n_boot, seed=seed + 4),
        "cde_an_delta": mean_and_interval(cde_an, n_boot=n_boot, seed=seed + 5),
        "cde_a_tv": mean_and_interval(cde_a_tv, n_boot=n_boot, seed=seed + 7),
        "cde_an_tv": mean_and_interval(cde_an_tv, n_boot=n_boot, seed=seed + 8),
        "interaction_an_minus_a": mean_and_interval(interaction, n_boot=n_boot, seed=seed + 6),
        "residual_control_curve": curve_a,
        "residual_control_curve_by_article": {"a": curve_a, "an": curve_an},
    }


def acceptance_checks(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    acc = config["acceptance"]
    med = summary["mediated"]
    direct = summary["direct"]
    med_curve_max = max(
        pt["residual_tv"]["mean"]
        for article in ("a", "an")
        for pt in med["residual_control_curve_by_article"][article]
    )
    direct_curve0_a = next(
        pt for pt in direct["residual_control_curve_by_article"]["a"] if pt["k"] == 0
    )
    checks = {
        "mediated_residual_ratio_small": med["residual_over_total_tv"]["mean"]
        <= acc["mediated_max_residual_over_total"],
        "direct_cde_a_tv_large": direct["cde_a_tv"]["mean"]
        >= acc["direct_min_cde_a_tv"],
        "mediated_k_curve_near_zero": med_curve_max <= acc["mediated_max_k_residual_tv"],
        "direct_force_a_k0_residual_large": direct_curve0_a["residual_tv"]["mean"]
        >= acc["direct_min_k0_residual_tv"],
        "mediated_cde_near_zero": max(med["cde_a_tv"]["mean"], med["cde_an_tv"]["mean"])
        <= acc["mediated_max_k_residual_tv"],
        "mediated_top1_reproduction": med["top1_article_reproduction"]["mean"] >= 0.999,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def fmt_ci(stat: dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}]"


def write_report(summary: dict[str, Any], checks: dict[str, Any], runtime_s: float) -> str:
    lines = [
        "# Synthetic token-mediation validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Runtime: {runtime_s:.1f}s",
        "",
        "Ground-truth mechanisms share vocabulary and cues. The assay must",
        "return near-zero token-clamped residual control for the mediated",
        "mechanism and large residual control for the direct-planning mechanism.",
        "",
    ]
    for name in ("mediated", "direct"):
        block = summary[name]
        lines.extend(
            [
                f"## Mechanism: `{name}`",
                "",
                f"- N cues: {block['n']}; article-switch rate: {block['article_switch_rate']:.2f}",
                f"- Total TV: {fmt_ci(block['total_tv'])}",
                f"- Article-only TV: {fmt_ci(block['article_only_tv'])}",
                f"- Residual TV: {fmt_ci(block['residual_tv'])}",
                f"- Residual/total TV: {fmt_ci(block['residual_over_total_tv'])}",
                f"- Article-only top-1 reproduction: {block['top1_article_reproduction']['mean']:.2f}",
                f"- CDE_a TV / Δ(target−source): {fmt_ci(block['cde_a_tv'])} / {fmt_ci(block['cde_a_delta'])}",
                f"- CDE_an TV / Δ(target−source): {fmt_ci(block['cde_an_tv'])} / {fmt_ci(block['cde_an_delta'])}",
                f"- Interaction (an−a) logit ΔΔ: {fmt_ci(block['interaction_an_minus_a'])}",
                "",
                "Primary residual-control curve under `do(a)` (unsaturated article):",
                "",
                "| k | Residual TV | Target−source ΔΔ |",
                "| ---: | ---: | ---: |",
            ]
        )
        for pt in block["residual_control_curve"]:
            lines.append(
                f"| {pt['k']} | {fmt_ci(pt['residual_tv'])} | {fmt_ci(pt['target_minus_source_delta'])} |"
            )
        lines.append("")
        lines.append("Secondary curve under `do(an)`:")
        lines.append("")
        lines.append("| k | Residual TV | Target−source ΔΔ |")
        lines.append("| ---: | ---: | ---: |")
        for pt in block["residual_control_curve_by_article"]["an"]:
            lines.append(
                f"| {pt['k']} | {fmt_ci(pt['residual_tv'])} | {fmt_ci(pt['target_minus_source_delta'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Acceptance checks",
            "",
        ]
    )
    for key, value in checks.items():
        if key == "all_passed":
            continue
        lines.append(f"- `{key}`: **{value}**")
    lines.append(f"- all_passed: **{checks['all_passed']}**")
    lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The mediated model is a positive control for generated-token relay:",
            "  free-generation TE is large, article-only reproduces it, and the",
            "  token-clamped residual / k-curve stay near zero.",
            "- The direct model is a positive control for residual control: the",
            "  same estimands remain large after article and filler clamping.",
            "- The Gemma full-residual reference remains necessary for stack",
            "  sensitivity; this synthetic check validates the estimands themselves.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR / "run.log")
    config = json.loads(CONFIG_PATH.read_text())
    torch.manual_seed(int(config["seed"]))

    t0 = time.time()
    vocab = SyntheticVocab(config["cues"], config["fillers"])
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for mechanism in config["mechanisms"]:
        model = GroundTruthAR(
            vocab,
            mechanism=mechanism,
            article_bias_on=float(config["article_bias_on"]),
            direct_plan_bias=float(config["direct_plan_bias"]),
            mediated_plan_bias=float(config["mediated_plan_bias"]),
            noun_article_coupling=float(config["noun_article_coupling"]),
            filler_bias=float(config["filler_bias"]),
            temperature=float(config["temperature"]),
        )
        mech_rows = [
            evaluate_prompt(
                model,
                cue,
                fillers=list(config["fillers"]),
                k_values=list(config["k_values"]),
            )
            for cue in config["cues"]
        ]
        rows.extend(mech_rows)
        summary[mechanism] = summarize_mechanism(
            mech_rows,
            n_boot=int(config["bootstrap_samples"]),
            seed=int(config["seed"]),
        )

    checks = acceptance_checks(summary, config)
    runtime_s = time.time() - t0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_s": runtime_s,
        "config": config,
        "summary": summary,
        "acceptance": checks,
        "vocab": vocab.id_to_token,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_json(RESULTS_DIR / "rows.json", {"rows": rows})
    report = write_report(summary, checks, runtime_s)
    (RESULTS_DIR / "report.md").write_text(report)
    logging.info("Acceptance all_passed=%s", checks["all_passed"])
    print(report)
    if not checks["all_passed"]:
        raise SystemExit("Synthetic mediation assay failed acceptance checks")


if __name__ == "__main__":
    main()
