#!/usr/bin/env bash
# Check that a killed written_chain run resumes and reproduces an uninterrupted run
# (Qwen3-0.6B, a name no experiment uses). Run from ~/amu_jobs/bundle.
P=python; S=experiments/derived_value_carry/written_chain.py; R=experiments/derived_value_carry/results/Qwen3-0.6B
rm -rf $R
$P -u $S Qwen3-0.6B --limit 4 --ks 3 --batch 2 > /tmp/resume_full.log 2>&1
cp $R/written_K3_rows.json /tmp/resume_full.json
rm -rf $R
$P -u $S Qwen3-0.6B --limit 4 --ks 3 --batch 2 > /tmp/resume_a.log 2>&1 &
pid=$!
until grep -q "items done" /tmp/resume_a.log 2>/dev/null; do sleep 2; done
kill $pid; sleep 2
echo "partial rows after kill: $(wc -l < $R/written_K3_rows.partial.jsonl)"
$P -u $S Qwen3-0.6B --limit 4 --ks 3 --batch 2 > /tmp/resume_b.log 2>&1
grep -m1 resuming /tmp/resume_b.log
$P - <<'PYEOF'
import json
a=json.load(open('/tmp/resume_full.json')); b=json.load(open('experiments/derived_value_carry/results/Qwen3-0.6B/written_K3_rows.json'))
key=lambda r: (r['v0'], r['dv0'], tuple(r['inc']))
A={key(r): r for r in a}; B={key(r): r for r in b}
same=all(A[k].get('total')==B[k].get('total') and A[k]['gen_off']==B[k]['gen_off'] for k in A)
print('rows', len(a), len(b), 'identical keys', set(A)==set(B), 'identical values', same)
PYEOF
rm -rf $R
