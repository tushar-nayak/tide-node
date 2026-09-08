#!/usr/bin/env bash
# Full 81-patient cohort validation run for the residual-prediction + real-mask-Dice
# pipeline (see README.md "Model & Evaluation Update" and PUBLICATION_PROTOCOL.md).
#
# This is a long, unattended job. Registration preprocessing alone is estimated at
# roughly 1-1.5 days of wall-clock on a 12-core machine (~9,112 SimpleITK registration
# calls, CPU-bound -- the GPU does not help this stage). Cohort-pretrain + per-patient
# fine-tuning after that is comparatively fast once volumes are pre-registered.
#
# USAGE
#   Run this inside tmux/screen (or with nohup) so it survives a disconnected terminal:
#     tmux new-session -s cohort_validation 'bash scripts/run_full_cohort_validation.sh'
#   or
#     nohup bash scripts/run_full_cohort_validation.sh > /tmp/full_cohort_validation.log 2>&1 &
#
# RESUMABILITY
#   Stage 1 (preprocessing) skips any (patient, target week, moving week) triple whose
#   registered output already exists on disk, so if this script is interrupted, just
#   re-run it -- completed registration work is not redone. Stage 2/3 are not resumable
#   mid-stage; if interrupted there, re-run from Stage 2 by commenting out Stage 1 below.
#
# OUTPUT
#   - data/lumiere_registered/                       registered volumes + manifest
#   - runs/lumiere_residual_v1_cohort_pretrain/       cohort pretraining stage
#   - runs/lumiere_residual_v1_Patient-*/             per-patient fine-tuned runs
#   - results/lumiere_residual_v1_metrics.{csv,json,md}   aggregated holdout comparison
#     (a NEW prefix -- this does not overwrite the existing lumiere_full_v1_metrics
#     checkpoint tables, so both are available to compare side by side)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="data/lumiere/extracted/Imaging"
REGISTERED_DIR="data/lumiere_registered"
RUN_NAME="lumiere_residual_v1"
RESULTS_PREFIX="lumiere_residual_v1_metrics"
LOG_DIR="/tmp/lumiere_residual_v1_logs"
mkdir -p "$LOG_DIR"

echo "=== Stage 1: parallel registration preprocessing ==="
echo "Logs: $LOG_DIR/preprocess_group_*.log"

# 81 valid patients (>=2 usable weeks) split into 8 groups, bin-packed by registration
# workload (pairs = weeks*(weeks-1)/2 per patient) rather than by patient count, since
# workload per patient ranges from 1 pair (2 weeks) to 210 pairs (21 weeks). Each group
# below carries ~285 registration pairs (~1,140 SimpleITK calls). Regenerate this split
# with a different NUM_GROUPS by re-running the bin-packing snippet from the session that
# produced this script if the cohort composition changes.
# NOTE: do not name this array GROUPS -- that is a reserved bash builtin (the current
# user's Unix group-id list), and silently overrides any assignment to it, which caused
# --patients to receive garbage GIDs instead of patient names the first time this ran.
PATIENT_GROUPS=(
  "Patient-067 Patient-060 Patient-055 Patient-022 Patient-086 Patient-075 Patient-049 Patient-038 Patient-083"
  "Patient-031 Patient-007 Patient-024 Patient-004 Patient-064 Patient-011 Patient-013 Patient-074 Patient-084"
  "Patient-073 Patient-033 Patient-025 Patient-061 Patient-068 Patient-048 Patient-017 Patient-001 Patient-039 Patient-087"
  "Patient-078 Patient-043 Patient-052 Patient-018 Patient-037 Patient-009 Patient-003 Patient-058 Patient-045"
  "Patient-006 Patient-042 Patient-054 Patient-077 Patient-012 Patient-035 Patient-008 Patient-089 Patient-010 Patient-080"
  "Patient-015 Patient-072 Patient-051 Patient-066 Patient-002 Patient-034 Patient-091 Patient-040 Patient-014 Patient-005 Patient-079 Patient-090"
  "Patient-019 Patient-023 Patient-059 Patient-028 Patient-062 Patient-071 Patient-050 Patient-041 Patient-021 Patient-027 Patient-081"
  "Patient-029 Patient-032 Patient-063 Patient-030 Patient-065 Patient-085 Patient-070 Patient-046 Patient-036 Patient-069 Patient-082"
)

PIDS=()
for i in "${!PATIENT_GROUPS[@]}"; do
  # shellcheck disable=SC2086
  python3 scripts/preprocess_lumiere_registration.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$REGISTERED_DIR" \
    --patients ${PATIENT_GROUPS[$i]} \
    > "$LOG_DIR/preprocess_group_${i}.log" 2>&1 &
  PIDS+=("$!")
  echo "  launched group $i (pid ${PIDS[-1]}, ${PATIENT_GROUPS[$i]}"
done

echo "Waiting for all ${#PIDS[@]} preprocessing groups to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  WARNING: preprocessing group with pid $pid exited non-zero -- check $LOG_DIR"
    FAILED=1
  fi
done
if [ "$FAILED" -eq 1 ]; then
  echo "One or more preprocessing groups reported an error. Check logs before trusting Stage 2/3."
fi
echo "Stage 1 preprocessing groups finished."

echo ""
echo "=== Stage 1b: consolidating registration manifest ==="
# The 8 parallel invocations above each write their own registration_manifest.json to the
# same --output-dir, so only the last one to finish "wins" that file. Everything is already
# cached on disk at this point, so this single serial pass over all patients just re-verifies
# and reuses the cache -- it should be fast -- and produces one complete, correct manifest.
python3 scripts/preprocess_lumiere_registration.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$REGISTERED_DIR" \
  > "$LOG_DIR/preprocess_manifest_consolidation.log" 2>&1
echo "Manifest consolidated: $REGISTERED_DIR/registration_manifest.json"

echo ""
echo "=== Stage 1c: verifying the cache is complete before starting expensive Stage 2 ==="
# --registered-data-dir makes run_neural_ode_pipeline.py load pre-registered volumes
# directly (see _has_registered_volume_set / _resolve_modality_path) instead of ever
# re-running SimpleITK registration, so registration only happens once, in Stage 1 -- but
# that guarantee is only as good as the cache actually being complete. Fail loudly here
# instead of silently discovering gaps (which would fall back to slow on-the-fly
# registration mid-training) hours into Stage 2.
# Expected pair count for the 81-patient cohort (sum of weeks*(weeks-1)/2 per patient),
# computed from the same dataset discovery used to build PATIENT_GROUPS above. Recompute
# this if the cohort composition changes (e.g. more LUMIERE data becomes available).
EXPECTED_PAIRS=2278

python3 -c "
import json
manifest = json.load(open('$REGISTERED_DIR/registration_manifest.json'))
generated = manifest['generated_job_count']
reused = manifest['reused_job_count']
total = generated + reused
expected = $EXPECTED_PAIRS
print(f'Registration cache: {generated} generated + {reused} reused = {total} total jobs (expected {expected})')
if total != expected:
    raise SystemExit(
        f'Registration cache is incomplete: {total} of {expected} expected pairs are present. '
        'Re-run this script (Stage 1 skips already-cached pairs, so it will only redo the missing ones) '
        'before trusting Stage 2.'
    )
print('Cache is complete. Stage 2 will read from disk and will not re-register anything.')
"
if [ $? -ne 0 ]; then
  echo "FATAL: registration cache verification failed. Not proceeding to Stage 2." >&2
  exit 1
fi

echo ""
echo "=== Stage 2: cohort-pretrain + per-patient fine-tune (matches PUBLICATION_PROTOCOL.md) ==="
python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir "$DATA_DIR" \
  --registered-data-dir "$REGISTERED_DIR" \
  --separate-patient-runs \
  --holdout-last-pair \
  --epochs 40 \
  --batch-size 1 \
  --model-size standard \
  --device cuda \
  --seed 7 \
  --run-name "$RUN_NAME" \
  2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "=== Stage 3: aggregating results ==="
python3 scripts/summarize_run_metrics.py \
  --runs-dir runs \
  --pattern "${RUN_NAME}_Patient-*/run_summary.json" \
  --output-dir results \
  --prefix "$RESULTS_PREFIX" \
  --split holdout \
  2>&1 | tee "$LOG_DIR/summarize.log"

echo ""
echo "=== Done. Compare results/${RESULTS_PREFIX}.md against results/lumiere_full_v1_metrics.md ==="
