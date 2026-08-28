"""Precision-aware feature interventions for the Gemma replacement model.

The standard circuit-tracer API returns the model's native logits.  With the
current stack those logits are bfloat16, which makes article margins visibly
quantized.  This module mirrors circuit-tracer's unconstrained intervention
path while additionally saving the final residual and recomputing the final
RMS normalization, tied unembedding, and logit soft cap in float32.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch
from nnsight import save


Intervention = tuple[int, int, int, float]


def _float32_head(model, pre_norm: torch.Tensor) -> torch.Tensor:
    """Apply Gemma 3's final normalization and unembedding in float32."""
    norm = model._model.model.norm
    value = pre_norm.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + norm.eps)
    value = value * (1.0 + norm.weight.detach().float())
    logits = torch.nn.functional.linear(
        value,
        model._model.lm_head.weight.detach().float(),
    )
    cap = model.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / float(cap)) * float(cap)
    return logits


@torch.no_grad()
def feature_intervention_precision(
    model,
    inputs: str,
    interventions: Sequence[Intervention],
) -> dict[str, Any]:
    """Run an unconstrained, attention-recomputed intervention with diagnostics.

    Returns native logits, float32-head logits, sparse feature activations, and
    the pre/post-final-normalization residuals.  The intervention semantics are
    copied from circuit-tracer 0.5.0's NNSightReplacementModel implementation.
    """
    activation_matrix, activation_fn = model.get_activation_fn(
        apply_activation_function=True,
        sparse=False,
    )
    by_layer: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for layer, pos, feature_idx, value in interventions:
        by_layer[int(layer)].append((int(pos), int(feature_idx), float(value)))
    intervention_layers = set(by_layer)
    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke(inputs):
            activation_fn(
                barrier=barrier,
                barrier_layers=intervention_layers,
                activation_layers=sorted(intervention_layers),
            )

        with tracer.invoke():
            n_pos = len(model.tokenizer(inputs).input_ids)
            layer_deltas = torch.zeros(
                [model.cfg.n_layers, n_pos, model.cfg.d_model],
                dtype=model.dtype,
                device=model.device,
            )
            for layer in range(model.cfg.n_layers):
                if by_layer[layer]:
                    barrier()
                    current = activation_matrix[layer]
                    if current.is_sparse:
                        current = current.to_dense()
                    activation_deltas = torch.zeros_like(current)
                    for pos, feature_idx, value in by_layer[layer]:
                        activation_deltas[pos, feature_idx] = (
                            value - current[pos, feature_idx]
                        )
                    positions, feature_indices = activation_deltas.nonzero(as_tuple=True)
                    new_values = activation_deltas[positions, feature_indices]
                    decoder_vectors = model.transcoders._module._get_decoder_vectors(
                        layer, feature_indices
                    )
                    if decoder_vectors.ndim == 2:
                        decoded = decoder_vectors * new_values.unsqueeze(1)
                        layer_deltas[layer].index_add_(0, positions, decoded)
                    else:
                        decoded = decoder_vectors * new_values.unsqueeze(-1).unsqueeze(-1)
                        decoded = decoded.transpose(0, 1)
                        n_remaining = decoded.shape[0]
                        layer_deltas[-n_remaining:].index_add_(1, positions, decoded)
                output = model.get_feature_output_loc(layer).output
                output[:] = output + layer_deltas[layer]
                layer_deltas[layer] *= 0

            pre_norm_proxy = model.model.norm.input[0]
            pre_norm = save(pre_norm_proxy)
            float32_logits = save(_float32_head(model, pre_norm_proxy))
            post_norm = save(model.model.norm.output)
            native_logits = save(model.output.logits)

    return {
        "native_logits": native_logits,
        "float32_logits": float32_logits,
        "pre_norm": pre_norm,
        "post_norm": post_norm,
    }


def dtype_audit(model, payload: dict[str, Any]) -> dict[str, str]:
    """Return the recorded dtype at every requested computation boundary."""
    return {
        "model_compute": str(model.dtype),
        "sparse_feature_activation": str(model.dtype),
        "transcoder_encoder_parameters": str(model.transcoders._module.W_enc.dtype),
        "transcoder_decoder_vectors": str(
            next(
                parameter.dtype
                for name, parameter in model.transcoders._module.named_parameters()
                if name.startswith("W_dec")
            )
        ),
        "transcoder_reconstruction_delta": str(model.dtype),
        "residual_before_final_norm": str(payload["pre_norm"].dtype),
        "residual_after_native_final_norm": str(payload["post_norm"].dtype),
        "final_norm_parameters": str(model._model.model.norm.weight.dtype),
        "unembedding_parameters_native": str(model._model.lm_head.weight.dtype),
        "stored_native_logits": str(payload["native_logits"].dtype),
        "stored_float32_head_logits": str(payload["float32_logits"].dtype),
    }
