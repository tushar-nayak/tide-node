#!/usr/bin/env bash
# Follow-up GPU queue (run in tmux; every stage resumable, finished folds are skipped):
#   1. original intensity-forecast model re-tested on held-out patients (intensity_forecast_cv.py)
#   2. step 2 retrained in the registered frame (needs register_step2_pairs.py to have finished)
# Logs: runs/logs/; sentinels: runs/logs/intensity_forecast.DONE, runs/logs/mask_forecast_registered.DONE
#   tmux new -d -s gbm_followups 'bash scripts/run_followup_experiments.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGS=runs/logs
mkdir -p "$LOGS"

[ -f "$LOGS/intensity_cache.DONE" ] || { $PY -u scripts/intensity_forecast_cv.py build-cache > "$LOGS/intensity_cache.log" 2>&1; touch "$LOGS/intensity_cache.DONE"; }
for f in 0 1 2 3 4; do
  $PY -u scripts/intensity_forecast_cv.py train --fold "$f" --workers 4 >> "$LOGS/intensity_fold$f.log" 2>&1
done
$PY scripts/intensity_forecast_cv.py summarize > "$LOGS/intensity_summary.log" 2>&1
touch "$LOGS/intensity_forecast.DONE"

if [ ! -f "$LOGS/register_step2_pairs.DONE" ]; then
  echo "waiting for register_step2_pairs.DONE"
  until [ -f "$LOGS/register_step2_pairs.DONE" ]; do
    grep -q Traceback "$LOGS/register_step2_pairs.log" 2>/dev/null && { echo "pair registration failed"; exit 1; }
    sleep 60
  done
fi
for f in 0 1 2 3 4; do
  for v in ode static; do
    $PY -u scripts/mask_forecast_experiment.py train --frame registered --variant "$v" --fold "$f" >> "$LOGS/mask_forecast_registered_${v}_fold$f.log" 2>&1
  done
done
$PY scripts/mask_forecast_experiment.py summarize --frame registered > "$LOGS/mask_forecast_registered_summary.log" 2>&1
for f in 0 1 2 3 4; do
  $PY scripts/visualize_mask_forecasts.py --frame registered --variant ode --fold "$f" --weights final >> "$LOGS/visualize_registered.log" 2>&1 || true
done
$PY scripts/plot_step_results.py >> "$LOGS/visualize_registered.log" 2>&1 || true
touch "$LOGS/mask_forecast_registered.DONE"
