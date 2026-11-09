# Assessment-first protocol, version 0.1

This is a new workflow, not a new independent dataset or a prospectively validated model.
Prior analyses and the existing manuscript remain exploratory evidence. The team has
already inspected LUMIERE outcomes; a new split cannot remove that exposure.

## Primary task

Estimate agreement with expert current-visit PD ratings from information available
when that visit is assessed. This differs from predicting the next visit. The model
does not diagnose true biological progression or determine treatment.

## Inputs and quality

Use historical and current automated compartment measurements, acquisition times,
and optional previous expert rating and baseline covariates. Exclude current/future
ratings, survival, future imaging and selectively annotated post-radiotherapy flags.
Require a single surgery-free segment, complete nonnegative finite measurements,
and anatomical review of alignment and segmentations. Do not infer quality simply
because volumes are present. Missing or unacceptable inputs must produce an explicit
insufficient-evidence result. Missing optional rating should be handled by a separately
evaluated automated variant, not silently treated as a clinical observation.

## Evaluation to implement next

1. Freeze a feature dictionary and patient manifest with hashes before new fitting.
2. Compare explicit enhancing-volume rules, regularized logistic regression, and a
   small gradient-boosted model. No Neural ODE or image-generating backbone.
3. Perform all feature selection, hyperparameter tuning, calibration, and decision
   threshold selection inside outer training patients. Evaluate the frozen choices
   only on outer test patients. Keep all visits from a patient together.
4. Report AUROC, precision-recall area, Brier score, calibration, sensitivity and
   specificity at training-selected thresholds, and fraction of scans deferred.
   Bootstrap whole patients, and give uncertainty for non-PD-to-PD assessments.
5. Separate optional-mask and previous-rating ablations. Audit mask geometry and
   registration using only current and historical scans.
6. Treat all LUMIERE estimates as internal/exploratory. Seek an independent cohort
   before broader generalization or clinical claims.

## Secondary task

Optional compartment-volume forecasts with calibrated ranges, evaluated separately.
Do not make accurate future masks or superiority of an ODE a release requirement.

## Milestones

- Implemented: validated measurement manifest, changes versus previous and historical
  nadir, provenance hash, machine-readable and human-readable reports, honest deferral.
- Next: independent current-visit feature extraction and geometry QC; patient split
  manifests; nested assessment benchmark and threshold calibration.
- Then: frozen model bundles, offline prediction, report plots and reproducible release.
- No new model has been trained or validated as part of this initial commit.
