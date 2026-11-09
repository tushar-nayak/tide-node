# Longitudinal tumor assessment

The active workflow on `main` estimates the **current expert PD rating** from
current and historical automated tumor-compartment volumes. The first CPU baseline
has been implemented and actually run on existing LUMIERE data. All results are
exploratory internal evaluation; there are **no fresh independent patients**.
These research estimates are not diagnoses or evidence of clinical readiness.

The prior forecasting project is preserved at
`archive/tide-forecasting-2026-11-09`, commit `1ca3176`; see
[LEGACY_README.md](LEGACY_README.md). Its results, data, caches, models and manuscript
remain unchanged. The old manuscript does not validate this assessment workflow.

## Completed baseline

The structurally eligible cohort contains **371 assessments from 79 patients**,
including 233 expert PD ratings. Three predefined seeds each use five outer patient
folds and three inner patient folds. All scaling, modest-grid model selection,
Platt calibration and decision-band cutoffs use outer training patients only.
Uncertainty resamples whole patients 2000 times. All eligible patients contribute
to internal evaluation; `main` is not an untouched independent cohort.

| Volumes-only comparator | AUROC (95% patient-bootstrap CI) | Average precision | Brier | Sensitivity / specificity at training-selected cutoff |
| --- | --- | --- | --- | --- |
| Enhancing versus previous | 0.679 (0.619–0.737) | 0.774 | 0.214 | 0.569 / 0.744 |
| Enhancing versus historical nadir | 0.705 (0.642–0.766) | 0.803 | 0.206 | 0.714 / 0.609 |
| Regularized logistic | 0.772 (0.715–0.827) | 0.844 | 0.184 | 0.621 / 0.790 |
| Small gradient boosting | 0.811 (0.758–0.859) | 0.873 | 0.169 | 0.718 / 0.766 |

The optional previous-rating analysis has 252 visits from 62 patients, with a
matched automated ablation using identical visits and splits. Boosting AUROC is
0.781 without previous rating and 0.839 with it; this is internal evidence, not a
claim of generalizable superiority. The previous-rating rule is also evaluated.
Among 120 visits from 49 patients after SD/PR/CR (58 new PD ratings), full-cohort
volumes-only boosting AUROC is 0.748 (0.654–0.838), with sensitivity 0.701 and
specificity 0.694. The optional-rating boosted model's sensitivity on this subset
is only 0.489. Overall performance does not establish new-progression detection.

Anatomical alignment and segmentation quality are **unverified**. Structural checks
do not certify anatomy. Research-band coverage is 59.5% for automated boosting,
but quality-admitted assessment coverage is **0%**. CLI reports retain an explicit
`insufficient_evidence` status even when displaying a research probability/band.
No formal conformal guarantee is claimed. Independent validation and anatomical
review remain necessary.

See [complete results and intervals](results/assessment_v1/REPORT.md),
[comparison plot](results/assessment_v1/comparison.png),
[calibration plot](results/assessment_v1/calibration.png),
[frozen protocol](assessment/PROTOCOL.md) and [execution runbook](assessment/RUNBOOK.md).
All 65 repository tests passed, including historical timing, surgery boundaries,
patient separation, training-derived thresholds, serialization and CLI equivalence.
A real-history CPU inference with no current rating supplied also passed.

## Exact execution commands

From this repository root, the completed evaluation used the existing `.venv`:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u -m assessment.evaluate > runs/assessment_v1/execution.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v > runs/assessment_build/full_tests.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m assessment.audit
```

An existing frozen evaluation cannot be overwritten. Use fresh `--results` and
`--run` paths to reproduce it; see the runbook. Configuration, feature/source
hashes, exact input hashes and outer/inner splits were saved before fitting.
Results live in `results/assessment_v1/`; local logs, environment, source snapshots,
training audits and bundles live in `runs/assessment_v1/`. Completion details and
commit IDs are in `runs/assessment_build/FINAL_REPORT.md`.

## Offline current-visit report

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m assessment \
  --history runs/assessment_v1/real_history_no_current_label.json \
  --model runs/assessment_v1/all_cohort_logistic.pkl --trust-model \
  --output runs/assessment-new-report
```

Pickle loading requires explicit trust because pickle can execute code. This
bundle is the predefined **all-cohort logistic refit**, with no independent accuracy
claim. The smoke history comes from its training cohort. Use only trusted local
bundles and their recorded environment/checksums. This command writes JSON and
Markdown with measurements, provenance, missingness, quality and research estimates.

For measurements without a model, omit `--model` and `--trust-model`; a synthetic
manifest is in `assessment/example.json`. Inputs require at least two ordered
visits within one surgery-free segment, finite nonnegative volumes in mm³ and no
current rating. Zero reference volumes produce null ordinary percentage changes
and finite model features. The optional previous-rating variant requires an
observed earlier rating; missing ratings are not silently treated as clinical
observations. No registration or expensive image models are needed.
