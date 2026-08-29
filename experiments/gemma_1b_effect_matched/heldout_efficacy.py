#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.lib.aan_protocol import write_json
from experiments.lib.core import load_replacement_model, token_id_for_text
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits
EXP=Path(__file__).resolve().parent; cfg=json.loads((EXP/'config.json').read_text()); source=json.loads((EXP/cfg['selection_config_path']).resolve().read_text())
orig=json.loads((EXP/cfg['selection_path']).resolve().read_text())['sets']['S1_dual_effect']['selected_features']
margin=json.loads((EXP/'results/margin_ranked_features.json').read_text())[:24]
handles={'original_top4_5x':(orig,5.0),'margin_top24_20x':(margin,20.0)}
model=load_replacement_model(cfg); tok=model.tokenizer; a=token_id_for_text(tok,' a'); an=token_id_for_text(tok,' an'); rows=[]
for i,ex in enumerate(source['test_examples'],1):
 prompt=f"{cfg['demonstration']} {ex['sentence']}"; pos=len(tok(prompt,add_special_tokens=True).input_ids)-1
 acts=activations_at_position(model,prompt,pos); base=next_logits(model,prompt,[]); bm=float(base[an]-base[a])
 for name,(features,gain) in handles.items():
  ints,ar=build_interventions(acts,pos,features,gain); out=next_logits(model,prompt,ints); tm=float(out[an]-out[a])
  rows.append({'index':i,'sentence':ex['sentence'],'handle':name,'baseline_margin':bm,'treated_margin':tm,'margin_movement':tm-bm,
               'crossed_to_an':bm<=0 and tm>0,'top_token':tok.decode([int(out.argmax())]),'article_is_top1':int(out.argmax()) in (a,an),
               'active_feature_count':sum(abs(x['activation'])>0 for x in ar),'mean_abs_activation':sum(abs(x['activation']) for x in ar)/len(ar)})
 print(f'heldout efficacy {i}/20',flush=True)
summary={}
for name in handles:
 g=[r for r in rows if r['handle']==name]; summary[name]={'n':len(g),'mean_margin_movement':sum(r['margin_movement'] for r in g)/len(g),
   'crossed_to_an_rate':sum(r['crossed_to_an'] for r in g)/len(g),'article_top1_rate':sum(r['article_is_top1'] for r in g)/len(g),
   'mean_active_feature_count':sum(r['active_feature_count'] for r in g)/len(g),'mean_abs_activation':sum(r['mean_abs_activation'] for r in g)/len(g)}
write_json(EXP/'results/heldout_efficacy_rows.json',rows); write_json(EXP/'results/heldout_efficacy_summary.json',summary); print(json.dumps(summary,indent=2))
