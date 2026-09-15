# Aggregated Run Metrics

Generated from local `run_summary.json` files. The `metric_split` column records which summary split was used for each run.

- Run count: 1
- Mean model MSE: 0.0092998
- Mean baseline MSE: 0.00688262
- Mean relative improvement: -35.1%
- Positive-improvement patients: 0
- Nonpositive-improvement patients: 1
- Mean tumor-focal relative improvement: -43.0%
- Ground-truth tumor Dice (whole tumor): model 0.053 vs persistence baseline 0.195 (n=1 runs with a real segmentation mask)
- Persistence-baseline Dice, real mask vs real mask (tumor core / enhancing): 0.000 / 0.000

## Top 1 Runs by Relative Improvement

| Patient | Split | Samples | Model MSE | Baseline MSE | Improvement |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Patient-029 | holdout | 1 | 0.0092998 | 0.00688262 | -35.1% |
