#!/usr/bin/env bash
# Bootstrap the LUMIERE dataset on a fresh machine: extract the downloaded archive (if
# not already extracted) and pre-register the full 81-patient cohort, so that future
# training runs (this script, or scripts/run_full_cohort_validation.sh) never have to
# re-run SimpleITK registration -- see DATA_SETUP.md for the full walkthrough.
#
# USAGE
#   bash scripts/setup_lumiere_data.sh [path/to/Imagingv202211.zip]
#
#   The zip argument is optional -- if data/lumiere/extracted/Imaging already exists
#   (e.g. you already extracted it, or copied over an existing "extracted" directory
#   from another machine), this skips straight to registration.
#
#   Run this inside tmux/screen (or with nohup) so it survives a disconnected terminal --
#   registration alone is estimated at ~1-1.5 days of wall-clock CPU-bound work on a
#   12-core machine (~9,112 SimpleITK registration calls across the cohort):
#     tmux new-session -s lumiere_setup 'bash scripts/setup_lumiere_data.sh path/to/Imagingv202211.zip'
#
# RESUMABILITY
#   Both extraction and registration are safe to re-run / interrupt and resume:
#   - Extraction is skipped entirely if data/lumiere/extracted/Imaging already exists.
#   - Registration skips any (patient, target week, moving week) triple whose output
#     already exists on disk (see preprocess_lumiere_registration.py), so if this script
#     is interrupted partway through Stage 2, just re-run it -- nothing already done gets
#     redone.
#
# WHY THIS ONLY NEEDS TO RUN ONCE
#   Registration operates on full 3D volumes and, for every target week, registers every
#   prior week to it -- the complete superset of pairs any history-conditioning strategy
#   (prefix vs sliding history, any context size, any model config) could ever need. Slice
#   extraction happens later, at training time, and is fully decoupled from this step. So
#   once this finishes, any future training run that passes
#   --registered-data-dir data/lumiere_registered reads straight from disk and never
#   re-registers anything, regardless of what else changes about the experiment.
#
#   The one exception: if LUMIERE ever releases more patients or more weeks for existing
#   patients, those new pairs would need registering -- but that is incremental (this
#   script skips everything already cached), not a redo of the whole cohort.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ZIP_PATH="${1:-}"
RAW_DIR="data/lumiere"
EXTRACTED_DIR="data/lumiere/extracted"
IMAGING_DIR="$EXTRACTED_DIR/Imaging"
REGISTERED_DIR="data/lumiere_registered"
LOG_DIR="/tmp/lumiere_setup_logs"
mkdir -p "$LOG_DIR"

echo "=== Stage 0: dataset extraction ==="
if [ -d "$IMAGING_DIR" ] && [ -n "$(ls -A "$IMAGING_DIR" 2>/dev/null)" ]; then
  echo "Found existing $IMAGING_DIR -- skipping extraction."
elif [ -n "$ZIP_PATH" ]; then
  if [ ! -f "$ZIP_PATH" ]; then
    echo "FATAL: zip path '$ZIP_PATH' does not exist." >&2
    exit 1
  fi
  echo "Extracting $ZIP_PATH into $EXTRACTED_DIR (this can take a while for a ~30GB+ archive)..."
  mkdir -p "$EXTRACTED_DIR"
  # unzip/7z are not assumed to be installed; Python's zipfile module always is, since
  # Python itself is a hard requirement for this whole project.
  python3 -c "
import zipfile
import sys

zip_path = sys.argv[1]
dest = sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    total = len(zf.infolist())
    for i, member in enumerate(zf.infolist(), start=1):
        zf.extract(member, dest)
        if i % 500 == 0 or i == total:
            print(f'  extracted {i}/{total}', flush=True)
" "$ZIP_PATH" "$EXTRACTED_DIR" 2>&1 | tee "$LOG_DIR/extract.log"

  if [ ! -d "$IMAGING_DIR" ]; then
    found_imaging=$(find "$EXTRACTED_DIR" -maxdepth 3 -type d -name "Imaging" | head -1)
    if [ -n "$found_imaging" ] && [ "$found_imaging" != "$IMAGING_DIR" ]; then
      echo "Archive extracted to a different layout than expected ($found_imaging)."
      echo "Symlinking $IMAGING_DIR -> $found_imaging"
      ln -s "$(cd "$found_imaging" && pwd)" "$IMAGING_DIR"
    fi
  fi
  if [ ! -d "$IMAGING_DIR" ]; then
    echo "FATAL: extraction finished but $IMAGING_DIR was not found. Check $LOG_DIR/extract.log" >&2
    exit 1
  fi
  echo "Extraction complete: $IMAGING_DIR"
else
  echo "FATAL: $IMAGING_DIR does not exist and no zip path was given." >&2
  echo "Usage: bash scripts/setup_lumiere_data.sh path/to/Imagingv202211.zip" >&2
  exit 1
fi

PATIENT_COUNT=$(find "$IMAGING_DIR" -maxdepth 1 -type d -name "Patient-*" | wc -l)
echo "Found $PATIENT_COUNT patient directories under $IMAGING_DIR."

echo ""
echo "=== Stage 1: parallel registration preprocessing ==="
echo "Logs: $LOG_DIR/preprocess_group_*.log"

# Same load-balanced 8-way split used by scripts/run_full_cohort_validation.sh, computed by
# bin-packing patients by registration workload (pairs = weeks*(weeks-1)/2 per patient),
# not by patient count -- workload per patient ranges from 1 pair (2 weeks) to 210 pairs
# (21 weeks). If the LUMIERE release on the new machine has a different patient/week
# composition than the 81-patient cohort this was computed for, re-run the bin-packing
# (see the session that produced this script) instead of trusting this list blindly --
# the registration script itself will just skip any group members it can't find, but the
# load balance across groups may no longer be even.
#
# NOTE: do not name this array GROUPS -- that is a reserved bash builtin (the current
# user's Unix group-id list), and silently overrides any assignment to it.
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
    --data-dir "$IMAGING_DIR" \
    --output-dir "$REGISTERED_DIR" \
    --patients ${PATIENT_GROUPS[$i]} \
    > "$LOG_DIR/preprocess_group_${i}.log" 2>&1 &
  PIDS+=("$!")
  echo "  launched group $i (pid ${PIDS[-1]})"
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
  echo "One or more preprocessing groups reported an error. Check logs before trusting the cache."
fi
echo "Stage 1 preprocessing groups finished."

echo ""
echo "=== Stage 1b: consolidating registration manifest ==="
# Each of the 8 parallel invocations above wrote its own registration_manifest.json to the
# same --output-dir, so only the last one to finish "wins" that file. Everything is already
# cached on disk at this point, so this single serial pass over every patient just
# re-verifies and reuses the cache (should be fast) and produces one complete manifest.
python3 scripts/preprocess_lumiere_registration.py \
  --data-dir "$IMAGING_DIR" \
  --output-dir "$REGISTERED_DIR" \
  > "$LOG_DIR/preprocess_manifest_consolidation.log" 2>&1
echo "Manifest consolidated: $REGISTERED_DIR/registration_manifest.json"

echo ""
echo "=== Stage 1c: verifying the cache is complete ==="
python3 -c "
import json
manifest = json.load(open('$REGISTERED_DIR/registration_manifest.json'))
generated = manifest['generated_job_count']
reused = manifest['reused_job_count']
total = generated + reused
print(f'Registration cache: {generated} generated + {reused} reused = {total} total jobs')
print('Patients covered:', len(manifest.get('patients', [])))
if total == 0:
    raise SystemExit('No registration jobs recorded -- something is very wrong.')
print('If this total looks low for your dataset release, check the logs in $LOG_DIR before trusting the cache.')
"

echo ""
echo "=== Done. Point future training runs at --registered-data-dir $REGISTERED_DIR ==="
