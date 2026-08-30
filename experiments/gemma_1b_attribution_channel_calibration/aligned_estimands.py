#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.six_cell_family_sweep.run import activations_at_position, next_logits

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
TEMPERATURES = (0.1, 0.25, 0.5, 1.0)


def load(path: Path):
    return json.loads(path.read_text())


def first_id(tokenizer, word):
    ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(word)
    return int(ids[0])


def tv_probs(left, right):
    return float(0.5 * (left-right).abs().sum())


def article_q(margin, tau):
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, margin/tau))))


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); ranks = [0.0]*len(values); cursor = 0
    while cursor < len(order):
        end = cursor+1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor+end-1)/2+1
        for index in order[cursor:end]: ranks[index] = rank
        cursor = end
    return np.asarray(ranks)


def rho(left, right):
    x, y = rankdata(left), rankdata(right); x -= x.mean(); y -= y.mean()
    denominator = float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if denominator == 0 else float(np.dot(x,y)/denominator)


def bootstrap(left, right, seed, resamples=10000):
    rng, n, draws = random.Random(seed), len(left), []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        value = rho([left[i] for i in indices], [right[i] for i in indices])
        if value is not None: draws.append(value)
    draws.sort()
    return {"rho": rho(left,right), "lo": draws[int(.025*(len(draws)-1))],
            "hi": draws[int(.975*(len(draws)-1))], "n": n, "resamples": resamples}


def cross_validated_gain(rows, features, tau):
    # Leave one feature out. Fit local margin pressure from attribution on the
    # remaining features, then compose it with baseline susceptibility and
    # continuation leverage. This keeps the held-out feature entirely unseen.
    observed, attr_only, susceptibility, full = [], [], [], []
    feature_ids = sorted({row["feature_index"] for row in rows})
    for heldout in feature_ids:
        train = [row for row in rows if row["feature_index"] != heldout]
        test = [row for row in rows if row["feature_index"] == heldout]
        x_train = np.asarray([[1.0, row["article_attribution"]] for row in train])
        y_train = np.asarray([row["article_margin_effect"] for row in train])
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        mean_leverage = sum(row["branch_leverage_tv"] for row in train)/len(train)
        a_train = np.asarray([[1.0, abs(row["article_attribution"])] for row in train])
        p_train = np.asarray([row["public_tv"][str(tau)] for row in train])
        beta_attr = np.linalg.lstsq(a_train, p_train, rcond=None)[0]
        for row in test:
            predicted_dm = float(beta[0] + beta[1]*row["article_attribution"])
            predicted_dq = article_q(row["baseline_margin"]+predicted_dm, tau)-article_q(row["baseline_margin"], tau)
            observed.append(row["public_tv"][str(tau)])
            attr_only.append(max(0.0, float(beta_attr[0]+beta_attr[1]*abs(row["article_attribution"]))))
            susceptibility.append(abs(predicted_dq)*mean_leverage)
            full.append(abs(predicted_dq)*row["branch_leverage_tv"])
    y = np.asarray(observed)
    denominator = float(((y-y.mean())**2).sum())
    def score(prediction):
        p = np.asarray(prediction)
        return 1-float(((y-p)**2).sum())/denominator if denominator else None
    return {"n": len(y), "r2_attribution_only": score(attr_only),
            "r2_attribution_plus_margin_susceptibility": score(susceptibility),
            "r2_full_gain_law": score(full),
            "mae_attribution_only": float(np.mean(np.abs(y-np.asarray(attr_only)))),
            "mae_susceptibility": float(np.mean(np.abs(y-np.asarray(susceptibility)))),
            "mae_full_gain_law": float(np.mean(np.abs(y-np.asarray(full))))}


def main():
    config = load(EXP / "config.json")
    source = load((EXP / config["source_config_path"]).resolve())
    selection = load(RESULTS / "selection.json")
    features = selection["features"]
    RESULTS.mkdir(parents=True, exist_ok=True); setup_file_logging(RESULTS)
    started = time.time(); model = load_replacement_model(config); tokenizer = model.tokenizer
    a_id, an_id = token_id_for_text(tokenizer," a"), token_id_for_text(tokenizer," an")
    gain = float(config["amplify_factor"]); rows = []
    for prompt_index, example in enumerate(source["test_examples"],1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt,add_special_tokens=True).input_ids)-1
        activations = activations_at_position(model,prompt,position)
        off_article = next_logits(model,prompt,[]); baseline_margin = float(off_article[an_id]-off_article[a_id])
        off_branches = {"a":next_logits(model,prompt+tokenizer.decode([a_id]),[]),
                        "an":next_logits(model,prompt+tokenizer.decode([an_id]),[])}
        branch_probs = {key:torch.softmax(value,-1) for key,value in off_branches.items()}
        branch_delta = branch_probs["an"]-branch_probs["a"]
        branch_leverage = float(.5*branch_delta.abs().sum())
        target_id = first_id(tokenizer,example["listed_word"])
        expected = example["expected_article"]
        source_id = int(off_branches[expected].argmax())
        for feature_index,feature in enumerate(features):
            layer,feature_idx = int(feature["layer"]),int(feature["feature_idx"])
            activation = float(activations[layer,feature_idx].detach().float().cpu())
            interventions=[{"layer":layer,"pos":position,"feature_idx":feature_idx,"value":activation*gain}]
            on_article=next_logits(model,prompt,interventions)
            on_expected=next_logits(model,prompt+tokenizer.decode([a_id if expected=="a" else an_id]),interventions)
            margin_effect=float((on_article[an_id]-on_article[a_id])-baseline_margin)
            target_change=float(on_expected[target_id]-off_branches[expected][target_id])
            source_change=float(on_expected[source_id]-off_branches[expected][source_id])
            public_tv={}; exact_error={}; dq={}
            for tau in TEMPERATURES:
                q0=article_q(baseline_margin,tau); q1=article_q(baseline_margin+margin_effect,tau)
                off_mix=(1-q0)*branch_probs["a"]+q0*branch_probs["an"]
                on_public=(1-q1)*branch_probs["a"]+q1*branch_probs["an"]
                observed=tv_probs(on_public,off_mix); predicted=abs(q1-q0)*branch_leverage
                public_tv[str(tau)]=observed; exact_error[str(tau)]=abs(observed-predicted); dq[str(tau)]=q1-q0
            rows.append({"feature_index":feature_index,"prompt_index":prompt_index,"layer":layer,"feature_idx":feature_idx,
                         "stratum":feature["stratum"],"activation":activation,
                         "article_attribution":feature["article_attribution"],"future_attribution":feature["future_attribution"],
                         "baseline_margin":baseline_margin,"article_margin_effect":margin_effect,
                         "target_id":target_id,"source_id":source_id,"expected_article":expected,
                         "target_logit_change_fixed_expected":target_change,
                         "source_logit_change_fixed_expected":source_change,
                         "target_minus_source_change_fixed_expected":target_change-source_change,
                         "branch_leverage_tv":branch_leverage,"delta_q":dq,"public_tv":public_tv,
                         "gain_law_absolute_error":exact_error})
        print(f"completed aligned prompt {prompt_index}/{len(source['test_examples'])}",flush=True)
    feature_rows=[]
    for index,feature in enumerate(features):
        group=[row for row in rows if row["feature_index"]==index]
        mean=lambda key:sum(float(row[key]) for row in group)/len(group)
        feature_rows.append({**feature,"target_logit_change_fixed_expected":mean("target_logit_change_fixed_expected"),
                             "target_minus_source_change_fixed_expected":mean("target_minus_source_change_fixed_expected"),
                             "fixed_target_effect_magnitude":sum(abs(row["target_logit_change_fixed_expected"]) for row in group)/len(group)})
    analyses={
        "signed_future_vs_signed_fixed_target":bootstrap([row["future_attribution"] for row in feature_rows],
            [row["target_logit_change_fixed_expected"] for row in feature_rows],20260830),
        "signed_future_vs_signed_fixed_target_minus_source":bootstrap([row["future_attribution"] for row in feature_rows],
            [row["target_minus_source_change_fixed_expected"] for row in feature_rows],20260831),
        "absolute_future_vs_fixed_target_magnitude":bootstrap([abs(row["future_attribution"]) for row in feature_rows],
            [row["fixed_target_effect_magnitude"] for row in feature_rows],20260832),
    }
    gain_law={str(tau):cross_validated_gain(rows,features,tau) for tau in TEMPERATURES}
    summary={"experiment":"gemma_1b_aligned_attribution_estimands","generated_at":datetime.now(timezone.utc).isoformat(),
             "elapsed_sec":time.time()-started,"n_features":len(features),"n_prompts":len(source["test_examples"]),
             "gain":gain,"analyses":analyses,"gain_law":gain_law,
             "exact_gain_law_max_absolute_error":max(error for row in rows for error in row["gain_law_absolute_error"].values())}
    write_json(RESULTS/"aligned_rows.json",rows);write_json(RESULTS/"aligned_feature_rows.json",feature_rows)
    write_json(RESULTS/"aligned_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
