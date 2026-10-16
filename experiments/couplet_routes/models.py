"""Model loading, prompt and anchor for the couplet route experiments (Qwen3 and Gemma 3).

Qwen3 keeps Hanna & Ameisen's exact prompt and anchor (two tokens before
<|im_end|>: the last word of line 1). Gemma 3 has no thinking switch, so the
instruction drops "/no_think"; its anchor is found by character offsets as the
token that ends the last word of line 1 (the same token the Qwen rule picks when
line 1 ends in one punctuation token).

Plain format (model names ending in "-plain", results in results/<model>-plain/):
Ma & Rui's (2026) prompt "A rhyming couplet:\n{line 1}\n" without a chat template,
so line 1 ends at an explicit newline boundary and line 2 is a plain continuation
(generation is cut at the next newline). Same checkpoints as the chat runs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from model_refs import revision  # noqa: E402  (torch-free; reads env/models.json)

END_OF_USER = ("<|im_end|>", "<end_of_turn>")
PLAIN = "-plain"
FORMAT = {"plain": False}  # set by load(); read by prompt_ids()


def is_gemma(model: str) -> bool:
    return model.lower().startswith("gemma")


def base_name(model: str) -> str:
    return model[: -len(PLAIN)] if model.endswith(PLAIN) else model


def hf_id(model: str) -> str:
    b = base_name(model)
    return f"google/{b}" if is_gemma(b) else f"Qwen/{b}"


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
    """Tokenizer, model and decoder layers. A model whose revision is recorded in
    env/models.json loads at exactly that revision; others load the Hub's current one."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    FORMAT["plain"] = model.endswith(PLAIN)
    pin = {"revision": revision(hf_id(model))} if revision(hf_id(model)) else {}
    tok = AutoTokenizer.from_pretrained(hf_id(model), **pin)
    m = AutoModelForCausalLM.from_pretrained(hf_id(model), dtype=torch.bfloat16, **pin).to(device).eval()
    return tok, m, decoder_layers(m)


def end_of_user(toks, anchor: int = 0) -> int:
    """End of the user turn (chat) or the newline ending line 1 (plain format)."""
    if FORMAT["plain"]:
        return next(i for i, t in enumerate(toks) if i > anchor and ("\n" in t or "Ċ" in t))
    return next(i for i, t in enumerate(toks) if t in END_OF_USER)


def cut_line(text: str) -> str:
    """Plain format: line 2 ends at the first newline."""
    return text.lstrip("\n").split("\n")[0].strip() if FORMAT["plain"] else text


def prompt_ids(tok, first_line: str):
    """Chat prompt ids, the line-1 anchor position, and the prompt text."""
    line = first_line.strip()
    if FORMAT["plain"]:
        text = (tok.bos_token or "") + f"A rhyming couplet:\n{line}\n"
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        words = list(re.finditer(r"[A-Za-z']+", line))
        end = text.rindex(line) + words[-1].end() - 1
        anchor = next(i for i, (s, e) in enumerate(enc.offset_mapping[0].tolist()) if s <= end < e)
        return enc.input_ids, anchor, text
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
