#!/usr/bin/env bash
# CPU-only local reproduction. Does not alter original data, caches or weights.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 ]]; then
  echo 'Usage: scripts/build_research_release.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
output=$(realpath -m "$1")
if [[ -e "$output" ]]; then
  echo "Refusing existing output: $output" >&2
  exit 2
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tide-release-mpl}"
py="$PWD/.venv/bin/python"
"$py" -m unittest discover -s tests
mkdir -p "$output"
cp release/contracts/*.json "$output/"
"$py" scripts/package_models.py fit-volume-all --contract "$output/volume-gbm-v1.json" \
  --data-dir data/lumiere/extracted/Imaging --clinical-dir data/lumiere/clinical \
  --output "$output/volume-all"
"$py" scripts/package_models.py inventory-masks --runs-dir runs \
  --data-dir data/lumiere/extracted/Imaging --clinical-dir data/lumiere/clinical \
  --output "$output/masks"
"$py" scripts/verify_release_inference.py --source "$PWD" --release "$output" \
  --output "$output/export-equivalence.json"
audit_output=$(mktemp -d /tmp/tide-release-audit.XXXXXXXX)
"$py" scripts/release_audit.py --source "$PWD" --output "$audit_output" --smoke
"$py" scripts/release_reference_audit.py --source "$PWD" --output "$audit_output/references.json"
cp -r "$audit_output" "$output/audit"
echo "Created local research artifacts in $output; inspect audit/audit.json for historical replay failures."
