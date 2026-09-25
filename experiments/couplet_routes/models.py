"""Model loading, prompt and anchor for the couplet route experiments (Qwen3 and Gemma 3).

Qwen3 keeps Hanna & Ameisen's exact prompt and anchor (two tokens before
<|im_end|>: the last word of line 1). Gemma 3 has no thinking switch, so the
instruction drops "/no_think"; its anchor is found by character offsets as the
token that ends the last word of line 1 (the same token the Qwen rule picks when
line 1 ends in one punctuation token).
"""
from __future__ import annotations

import re

import torch

END_OF_USER = ("<|im_end|>", "<end_of_turn>")


def is_gemma(model: str) -> bool:
    return model.lower().startswith("gemma")


def hf_id(model: str) -> str:
    return f"google/{model}" if is_gemma(model) else f"Qwen/{model}"


def decoder_layers(m):
    """The decoder block ModuleList (text-only and multimodal Gemma 3 classes differ)."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder layers found")


def load(model: str, device: str = "mps"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id(model))
    m = AutoModelForCausalLM.from_pretrained(hf_id(model), dtype=torch.bfloat16).to(device).eval()
    return tok, m, decoder_layers(m)


def end_of_user(toks) -> int:
    return next(i for i, t in enumerate(toks) if t in END_OF_USER)


def prompt_ids(tok, first_line: str):
    """Chat prompt ids, the line-1 anchor position, and the prompt text."""
    line = first_line.strip()
    if "<|im_end|>" in tok.get_vocab():
        msgs = [{"role": "user", "content": f"/no_think Write only the next line of this rhyming couplet: {line}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids, tok.convert_ids_to_tokens(ids[0]).index("<|im_end|>") - 2, text
    msgs = [{"role": "user", "content": f"Write only the next line of this rhyming couplet: {line}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    words = list(re.finditer(r"[A-Za-z']+", line))
    end = text.rindex(line) + words[-1].end() - 1  # last character of line 1's last word
    anchor = next(i for i, (s, e) in enumerate(enc.offset_mapping[0].tolist()) if s <= end < e)
    return enc.input_ids, anchor, text
