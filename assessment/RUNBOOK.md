# Running the current-visit baseline

Run from the repository root with the existing `.venv`. Assessment-only dependency
minimums are in `assessment/requirements.txt`; exact executed versions are saved in
`runs/assessment_v1/requirements.freeze.txt`. No GPU, registration, image generation,
new data download or online service is needed.

The completed run used:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u -m assessment.evaluate > runs/assessment_v1/execution.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v > runs/assessment_build/full_tests.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m assessment.audit > runs/assessment_build/artifact_audit.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m assessment \
  --history runs/assessment_v1/real_history_no_current_label.json \
  --model runs/assessment_v1/all_cohort_logistic.pkl --trust-model \
  --output runs/assessment_v1/real_history_smoke
```

The evaluation deliberately rejects an existing `freeze.json`. To reproduce the
same predefined evaluation without replacing the completed evidence:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u -m assessment.evaluate \
  --results runs/assessment_reproduction/results \
  --run runs/assessment_reproduction/local
```

Supply `--data` and `--ratings` only if the original LUMIERE extraction moved.
Recorded input paths/checksums must resolve to the same measurements. Exact
replication requires the recorded environment and source revisions. `assessment.audit`
is a baseline-specific read-only audit, including expected cohort/prediction counts;
use its `--results` and `--run` options for a reproduction. Source snapshots are
saved before fitting. No legacy results or caches are overwritten.

## Offline historical measurements

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m assessment \
  --history assessment/example.json --output runs/assessment-example-v1
```

Add `--model runs/assessment_v1/all_cohort_logistic.pkl --trust-model` for the
research score. Pickle can execute code; load only your trusted local artifact.
Use the saved checksum to verify identity and use the frozen environment. Output
must be a new directory. JSON and Markdown retain the existing report schema,
measurements, provenance hash and explicit missingness/quality status. Model
reports add raw and calibrated probability, training-derived cutoffs, research
band and binary estimate. Missing/invalid measurements and missing required
optional previous ratings produce an insufficient-evidence report without a score.
Malformed JSON/model-loading errors exit explicitly.

The version-1 manifest contains only `version`, `patient_id`, `history`; every
history element contains `week`, `volumes_mm3`, and optionally a **historical**
`rating`. Three compartment keys are required, in mm³. History must have at least
two strictly ordered visits within one surgery-free segment. The assessed scan
is last, and its rating must be omitted (null/empty/unknown also signify withheld).
Keep supervised targets in a separate file. Do not provide future scans, flag
columns, survival or current labels as inference metadata.

The bundled real-history smoke input contains genuine historical/current volumes
and earlier ratings, with the current expert rating removed. It is from the refit
cohort, so it demonstrates CPU execution and evaluation/CLI feature equivalence,
not independent accuracy. The default exported model is automated volumes-only.
Optional-rating models are internally evaluated; no optional model is silently
substituted when an observed previous rating is missing.

No anatomical review is performed by these commands. A score is an exploratory
estimate of expert rating, not a diagnosis. The CLI therefore retains
`assessment.status = insufficient_evidence`, `quality_deferral = true` with an
attached model, even when a research band is lower or higher. Statistical band
coverage and anatomical-quality-admitted coverage must not be conflated.

## Artifact map

- `results/assessment_v1/REPORT.md`, `metrics.json`: completed results and patient-bootstrap intervals.
- `outer_predictions.csv`: all outer test predictions, thresholds, bands and training audit pointers.
- `comparison.png`, `calibration.png`, `calibration_bins.csv`: readable results and fixed-bin calibration with frozen provenance.
- `freeze.json`, `config.json`, `splits.json`: pre-fit definitions, exact hashes and split membership.
- `cohort.json`, `input_manifest.json`, `exclusions.json`: target/input separation, sources and exclusion events.
- `runs/assessment_v1/*_training.json`: all inner OOF values, candidate scores and patient memberships.
- `all_cohort_logistic.pkl`, `all_cohort_refit_training.json`: explicitly all-cohort refit and audit.
- `frozen_source/`, `environment.json`, `requirements.freeze.txt`, `checksums.json`, `execution.log`: reproducibility evidence.
- `runs/assessment_build/`: tests, artifact audit, execution history and final completion report; first implementation run is retained in `attempt1/`.

Local bundles and execution directories are ignored by git and are not release
artifacts. Result tables and plots are versioned. All available patients are
internally evaluated. Independent validation and anatomical review remain unmet.
