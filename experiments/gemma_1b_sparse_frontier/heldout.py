#!/usr/bin/env python3
from pathlib import Path

import experiments.gemma_1b_effect_matched.run_six_cell as assay

EXP=Path(__file__).resolve().parent
assay.EXP_DIR=EXP
assay.CONFIG_PATH=EXP/"config.json"
assay.RESULTS_DIR=EXP/"results/heldout"

if __name__=="__main__":
    assay.main()
