# Longitudinal tumor assessment — fresh direction

The active work on `main` now focuses on **assessing the scan that has just been
acquired**, using longitudinal tumor measurements, explicit quality checks and
uncertainty. Optional future-volume ranges are secondary. Neural ODEs and future
MRI/mask generation are no longer the organizing approach.

The previous project is preserved at branch `archive/tide-forecasting-2026-11-09`,
commit `1ca3176`. Its code and evidence remain here for reference; see
[LEGACY_README.md](LEGACY_README.md). Existing data, registration caches and model
artifacts have not been deleted. The separate manuscript still describes the prior
study and is not evidence for a newly validated system.

## First working component

```bash
python -m assessment --history assessment/example.json --output runs/assessment-example
```

This standard-library-only command accepts at least two ordered visits in one
surgery-free segment. It reports compartment volumes and changes versus the previous
scan and historical nadir in `report.json` and `report.md`. Input volumes are in mm³;
reports use mL. The example is synthetic. Use a new output directory for each run.

No future image or label is needed. The current rating must be omitted. The tool
checks measurements but does not certify registration or segmentation quality.
It reports insufficient evidence for a progression classification because no new
assessment model is attached yet.

## What comes next

See [assessment/PROTOCOL.md](assessment/PROTOCOL.md). Build geometry and measurement
quality checks, freeze patient manifests and feature definitions, then compare
regularized logistic regression and small boosted trees against explicit volume
rules. Fit calibration and decision thresholds inside training patients only.
Provide clear deferral behavior and patient-level evaluation of both accuracy and
coverage. Current LUMIERE estimates are exploratory; reusing it does not create a
new independent test cohort.

The research goal is a reproducible assessment tool. Clinical utility and external
generalization remain unestablished. No accuracy advantage is promised by the reset.
