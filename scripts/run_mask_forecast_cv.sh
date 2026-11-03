#!/bin/bash
# Step 2 runner: waits for the atlas cache, then trains every (fold, variant). Resumable:
# finished folds are skipped and interrupted folds resume from last_state.pt.
cd "$(dirname "$0")/.."
mkdir -p runs/logs
.venv/bin/python scripts/build_atlas_cache.py >> runs/logs/atlas_cache.log 2>&1  # no-op if already cached
for f in 0 1 2 3 4; do
  for v in ode static; do
    .venv/bin/python scripts/mask_forecast_experiment.py train --variant $v --fold $f >> runs/logs/mask_forecast_${v}_fold$f.log 2>&1
    .venv/bin/python scripts/mask_forecast_experiment.py summarize > runs/logs/mask_forecast_summary.log 2>&1
    .venv/bin/python scripts/visualize_mask_forecasts.py --variant $v --fold $f --weights final >> runs/logs/visualize.log 2>&1 || true
  done
done
.venv/bin/python scripts/plot_step_results.py >> runs/logs/visualize.log 2>&1
touch runs/logs/mask_forecast.DONE
