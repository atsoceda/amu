#!/usr/bin/env python3
"""Hidden choice, stage 0 probe: does the model follow the format, and does its secret
pick depend on the list order (so that list-order donors change the pick)?

Prompt: "Secretly choose one fruit from this list: ... Do not write your choice yet.
First write one sentence about the weather. Then write the fruit you chose." Reports
the free generation and the reveal after a forced neutral sentence, for random list
orders. No route measurement here.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couplet_routes"))
from models import load  # noqa: E402

OPTS = ["apple", "banana", "cherry", "grape", "lemon", "mango", "peach", "plum"]
FORCED = "The weather is calm and mild today.\nThe fruit I chose is"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"
    tok, m, _ = load(model)
    rng = random.Random(1)
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 12):
        lst = rng.sample(OPTS, 4)
        q = (f"Secretly choose one fruit from this list: {', '.join(lst)}. Do not write your choice yet. "
             "First write one sentence about the weather. Then write the fruit you chose.")
        base = tok.apply_chat_template([{"role": "user", "content": "/no_think " + q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        outs = []
        for pre, n in (("", 40), (FORCED, 4)):
            ids = tok(base + pre, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
            o = m.generate(ids, max_new_tokens=n, do_sample=False)
            outs.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " / "))
        print(lst, "| free:", outs[0][:100], "| forced reveal:", outs[1], flush=True)


if __name__ == "__main__":
    main()
