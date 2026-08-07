# Selection-Criterion Ablation (E1)

Compares four frozen feature-selection rules under the same \(5\times\) gain-of-function
protocol on held-out occupation `a`/`an` prompts.

## Feature sets

| ID | Rule |
| --- | --- |
| S1 Dual-effect | \(+\mathrm{attr}(\texttt{an})\) and \(+\mathrm{attr}(\text{future})\) |
| S2 Article-only | \(+\mathrm{attr}(\texttt{an})\), \(\lvert\mathrm{attr}(\text{future})\rvert \le 0.05\) |
| S3 Content-only | \(+\mathrm{attr}(\text{future})\), \(\lvert\mathrm{attr}(\texttt{an})\rvert \le 0.05\) |
| S4 Competing / a-favoring | \(+\mathrm{attr}(\texttt{a}-\texttt{an})\), \(\lvert\mathrm{attr}(\text{future})\rvert \le 0.05\) |

## Run

```bash
cd /Users/anthony/repos/amu
/Users/anthony/miniconda3/bin/python \
  experiments/selection_criterion_ablation/run.py
```

Optional:

```bash
# Graphs + selection.json only
/Users/anthony/miniconda3/bin/python \
  experiments/selection_criterion_ablation/run.py --selection-only
```

## Outputs

- `results/selection.json` — ranked/selected features per set
- `results/summary.json` — held-out scores + E2 candidate sets
- `results/report.md`
- `results/graphs/` — article + future attribution graphs
- `results/run.log`
