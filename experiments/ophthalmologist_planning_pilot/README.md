# Ophthalmologist Planning Pilot

This targeted mechanistic pilot follows the reproducible behavioral continuation:

```text
Someone who studies living organisms is a biologist.
Someone who treats eye diseases is a ophthalmologist.
```

It asks whether a feature active before the incorrect article supports both the
losing `an` logit and the future answer prefix ` ophthalm`.

Run:

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/ophthalmologist_planning_pilot/run.py
```

Outputs:

- `results/graphs/article.pt`
- `results/graphs/future.pt`
- `results/summary.json`
- `results/report.md`
- `results/run.log`

To regenerate it, prompt an agent with:

> Regenerate the `ophthalmologist_planning_pilot` results.
