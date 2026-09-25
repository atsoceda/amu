"""Batched forward passes and generation for the route experiments (added 2026-09-26).

The Mac Studio GPU was 100% busy with batch-size-1 jobs, so batching (not more
parallel jobs) is what speeds runs up. Three helpers:

- run_edits: one sequence under many state edits in one forward pass (the same input
  repeated, each row with its own {position: per-layer states} replacement). The
  replacement is one indexed write per layer.
- last_logprobs: next-token log-probabilities for several texts (left-padded).
- generate_batch: greedy generation for several prompts (left-padded), optionally with
  per-row state patches applied during the prompt pass.

Every script that uses these was checked against its unbatched version on a few items
before use (see the experiment READMEs).
"""
from __future__ import annotations

import torch


def _hooks(layers, reps, offsets=None, prompt_len=None):
    """Register hooks writing reps[b][pos][layer] into row b (pos shifted by offsets[b])."""
    pairs = [(b, p + (offsets[b] if offsets else 0), sts) for b, rep in enumerate(reps) for p, sts in rep.items()]
    if not pairs:
        return []
    dev = None
    b_idx = torch.tensor([b for b, _, _ in pairs])
    p_idx = torch.tensor([p for _, p, _ in pairs])
    hs = []
    for li, layer in enumerate(layers):
        vals = torch.stack([sts[li] for _, _, sts in pairs])

        def fn(mod, inp, out, vals=vals):
            nonlocal dev
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1 or (prompt_len is not None and h.shape[1] != prompt_len):
                return out
            h = h.clone()
            h[b_idx.to(h.device), p_idx.to(h.device)] = vals.to(h.device, h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        hs.append(layer.register_forward_hook(fn))
    return hs


@torch.no_grad()
def run_edits(m, layers, ids, reps, want_states=False, chunk=16, device="mps"):
    """ids: (1, T). reps: list of {pos: [per-layer state]}. Returns (log-probs at the last
    position per edit, per-edit list of per-layer states or None)."""
    outs, states = [], []
    for c in range(0, len(reps), chunk):
        part = reps[c:c + chunk]
        x = ids.to(device).expand(len(part), -1).contiguous()
        hs = _hooks(layers, part)
        try:
            o = m(x, output_hidden_states=want_states, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        lp = torch.log_softmax(o.logits[:, -1].float(), -1).cpu()
        outs.extend(lp[b] for b in range(len(part)))
        if want_states:
            layer_states = o.hidden_states[1:]
            states.extend([[h[b] for h in layer_states] for b in range(len(part))])
    return outs, (states if want_states else None)


def _left_pad(tok, texts, device):
    enc = [tok(t, add_special_tokens=False).input_ids for t in texts]
    T = max(len(e) for e in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[pad] * (T - len(e)) + e for e in enc], device=device)
    mask = torch.tensor([[0] * (T - len(e)) + [1] * len(e) for e in enc], device=device)
    return ids, mask, [T - len(e) for e in enc]


@torch.no_grad()
def last_logprobs(m, tok, texts, chunk=16, device="mps"):
    """Next-token log-probabilities after each text (texts may differ in length)."""
    out = []
    for c in range(0, len(texts), chunk):
        ids, mask, _ = _left_pad(tok, texts[c:c + chunk], device)
        o = m(ids, attention_mask=mask, logits_to_keep=1)
        out.extend(torch.log_softmax(o.logits[:, -1].float(), -1).cpu())
    return out


@torch.no_grad()
def generate_batch(m, tok, layers, texts, max_new_tokens, patches=None, chunk=16, device="mps"):
    """Greedy continuations of texts. patches: optional list (one per text) of
    {pos: [per-layer state]} applied during the prompt pass (positions in unpadded
    coordinates)."""
    outs = []
    for c in range(0, len(texts), chunk):
        part = texts[c:c + chunk]
        ids, mask, offs = _left_pad(tok, part, device)
        hs = _hooks(layers, patches[c:c + chunk], offs, prompt_len=ids.shape[1]) if patches else []
        try:
            g = m.generate(ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        finally:
            for h in hs:
                h.remove()
        outs.extend(tok.decode(g[b, ids.shape[1]:], skip_special_tokens=True) for b in range(len(part)))
    return outs
