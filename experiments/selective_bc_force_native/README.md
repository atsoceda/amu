# Selective b-step intervention + force-native article

Tests the editable-wrapper planner hypothesis under the selective protocol:

1. Intervene only while predicting the article \(b\)
2. Force the native baseline article into the string
3. Generate content \(c\) with the intervention off

Companion free generation (intervention left on) checks packager behavior.

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python experiments/selective_bc_force_native/run.py
```

Requires E1 `selection.json` for S1/S2/S3 feature IDs.
