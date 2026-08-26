"""Shared estimands for generated-token mediation assays.

These quantities formalize the Phase-2 method:

- total free-generation effect (TE)
- article-only / mediator-only reproduction
- token-clamped residual / controlled direct effect (CDE)
- intervention–mediator interaction (CDE_an - CDE_a)
- k-token residual-control curve under forced prefixes

Distance ratios are descriptive: TV is not an additive causal decomposition.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch


def js_divergence_from_logits(left: torch.Tensor, right: torch.Tensor) -> float:
    log_p = torch.log_softmax(left, dim=-1)
    log_q = torch.log_softmax(right, dim=-1)
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(torch.finfo(m.dtype).tiny))
    js = 0.5 * torch.sum(p * (log_p - log_m)) + 0.5 * torch.sum(q * (log_q - log_m))
    return float(js)


def total_variation_from_logits(left: torch.Tensor, right: torch.Tensor) -> float:
    p = torch.softmax(left, dim=-1)
    q = torch.softmax(right, dim=-1)
    return float(0.5 * torch.sum(torch.abs(p - q)))


def _unit_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm < 1e-12 or right_norm < 1e-12:
        return 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def effect_vector_decomposition(
    *,
    baseline_off: torch.Tensor,
    treated_article_off: torch.Tensor,
    treated_article_on: torch.Tensor,
) -> dict[str, float]:
    """Additive probability-vector decomposition of a generated-token contrast.

    Let Y(i, b) be the noun-token distribution under intervention i and article
    b. For greedy articles B0, B1 this stores

        E_T = Y(1, B1) - Y(0, B0)
        E_M = Y(0, B1) - Y(0, B0)
        E_R = Y(1, B1) - Y(0, B1)

    with E_T = E_M + E_R before any distance is taken. The TV of a difference
    vector is half its L1 norm; those TVs are not themselves additive.
    """
    y_off_b0 = torch.softmax(baseline_off.float(), dim=-1)
    y_off_b1 = torch.softmax(treated_article_off.float(), dim=-1)
    y_on_b1 = torch.softmax(treated_article_on.float(), dim=-1)
    e_t = y_on_b1 - y_off_b0
    e_m = y_off_b1 - y_off_b0
    e_r = y_on_b1 - y_off_b1
    reconstruction = e_t - (e_m + e_r)

    def tv_norm(delta: torch.Tensor) -> float:
        return float(0.5 * torch.sum(torch.abs(delta)))

    return {
        "total_tv": tv_norm(e_t),
        "mediator_tv": tv_norm(e_m),
        "residual_tv": tv_norm(e_r),
        "l1_total": float(torch.sum(torch.abs(e_t))),
        "l1_mediator": float(torch.sum(torch.abs(e_m))),
        "l1_residual": float(torch.sum(torch.abs(e_r))),
        "cosine_mediator_total": _unit_cosine(e_m, e_t),
        "cosine_residual_total": _unit_cosine(e_r, e_t),
        "cosine_mediator_residual": _unit_cosine(e_m, e_r),
        "reconstruction_l1": float(torch.sum(torch.abs(reconstruction))),
        "residual_tv_over_total_tv": (
            tv_norm(e_r) / tv_norm(e_t) if tv_norm(e_t) > 0 else 0.0
        ),
    }


def top_k_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    left_ids = set(torch.topk(left, k=min(k, left.numel())).indices.tolist())
    right_ids = set(torch.topk(right, k=min(k, right.numel())).indices.tolist())
    return len(left_ids & right_ids) / max(len(left_ids | right_ids), 1)


def logit_contrast(
    logits: torch.Tensor,
    target_id: int,
    source_id: int,
) -> float:
    return float(logits[target_id] - logits[source_id])


def distribution_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    return {
        "js": js_divergence_from_logits(left, right),
        "tv": total_variation_from_logits(left, right),
        "top_k_jaccard": top_k_overlap(left, right, top_k),
    }


def mediation_decomposition(
    *,
    baseline_free: torch.Tensor,
    treated_free: torch.Tensor,
    article_only: torch.Tensor,
    residual_on: torch.Tensor,
    residual_off: torch.Tensor,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare TE, mediator-only, and token-clamped residual distances.

    Arguments are next-token logit vectors for the outcome token position.
    `article_only` is the intervention-off distribution under the treated
    mediator token; `residual_{on,off}` share that same visible prefix.
    """
    total = distribution_distance(treated_free, baseline_free, top_k=top_k)
    mediator_only = distribution_distance(article_only, baseline_free, top_k=top_k)
    residual = distribution_distance(residual_on, residual_off, top_k=top_k)
    total_tv = max(float(total["tv"]), 1e-12)
    return {
        "total": total,
        "mediator_only": mediator_only,
        "residual": residual,
        "residual_over_total_tv": float(residual["tv"]) / total_tv,
        "top1_mediator_reproduces_treated": bool(
            int(torch.argmax(article_only).item()) == int(torch.argmax(treated_free).item())
        ),
    }


def controlled_direct_effects(
    *,
    on_a: torch.Tensor,
    off_a: torch.Tensor,
    on_an: torch.Tensor,
    off_an: torch.Tensor,
    target_id: int,
    source_id: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """Token-clamped CDEs and their article interaction."""
    cde_a_dist = distribution_distance(on_a, off_a, top_k=top_k)
    cde_an_dist = distribution_distance(on_an, off_an, top_k=top_k)
    delta_a = logit_contrast(on_a, target_id, source_id) - logit_contrast(
        off_a, target_id, source_id
    )
    delta_an = logit_contrast(on_an, target_id, source_id) - logit_contrast(
        off_an, target_id, source_id
    )
    return {
        "cde_a": {
            **cde_a_dist,
            "target_minus_source_delta": float(delta_a),
        },
        "cde_an": {
            **cde_an_dist,
            "target_minus_source_delta": float(delta_an),
        },
        "interaction_an_minus_a": {
            "tv": float(cde_an_dist["tv"] - cde_a_dist["tv"]),
            "js": float(cde_an_dist["js"] - cde_a_dist["js"]),
            "target_minus_source_delta": float(delta_an - delta_a),
        },
    }


def mean_and_interval(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    xs = [float(v) for v in values]
    if not xs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean = sum(xs) / len(xs)
    if len(xs) == 1 or n_boot <= 0:
        return {"mean": mean, "lo": mean, "hi": mean, "n": len(xs)}
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randint(0, len(xs), (n_boot, len(xs)), generator=g)
    arr = torch.tensor(xs, dtype=torch.float64)
    boots = arr[idx].mean(dim=1)
    lo = float(torch.quantile(boots, alpha / 2).item())
    hi = float(torch.quantile(boots, 1 - alpha / 2).item())
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(xs)}


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    key: str,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    return mean_and_interval(
        [float(row[key]) for row in rows],
        n_boot=n_boot,
        seed=seed,
    )


def residual_control_curve(
    points: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a list of {k, residual_tv, ...} measurements."""
    out: list[dict[str, Any]] = []
    for point in points:
        item = dict(point)
        item["k"] = int(item["k"])
        item["residual_tv"] = float(item["residual_tv"])
        item["residual_js"] = float(item.get("residual_js", 0.0))
        out.append(item)
    return sorted(out, key=lambda row: row["k"])
