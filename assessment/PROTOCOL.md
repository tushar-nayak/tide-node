# Assessment-first protocol, version 1.0

This workflow estimates agreement with the **current expert PD rating**, versus
SD/PR/CR. It does not diagnose biological progression or determine treatment.
All LUMIERE work is exploratory. Outcomes were inspected in prior work; all eligible
patients are internally evaluated, and no fresh independent patients exist.
The old forecasting analyses and manuscript remain unchanged.

## Structural cohort and quality

`assessment/cohort.py` independently discovers original DeepBraTumIA compartment
measurement JSON files, using only the audited week-name parser from
`scripts/lumiere_weeks.py`. It does not use forecasting cohorts, registrations,
masks, globally selected margins, survival, selective rating flags or rationale.
The expert CSV supplies only patient/visit keys, surgery boundaries, historical
ratings and the supervised current target. Its whole-file hash is provenance,
not a feature. Demographic/clinical variants and mask comparisons are not run.

Choose the highest acquisition suffix per week without requiring MRI modalities
for this volume-only analysis. Log discarded duplicates. Drop preoperative scans;
reset history for any postoperative annotation in that week, **before** checking
volumes. A missing or conflicting postoperative measurement cannot bridge surgery.
Require two strictly ordered valid visits in one segment, with exactly three
finite, nonnegative compartment measurements. Other source JSON fields are ignored.
An unlabeled visit can supply historical measurements, but cannot be a target.
Identical rating duplicates collapse; conflicting current ratings are excluded as
targets and as previous-rating observations. Any surgery annotation still resets
history. Exclusion records are events; one visit can have several reasons.

Alignment, segmentation accuracy, residual confounding, pseudoprogression and
measurement reliability have **not** been anatomically verified. Structural
eligibility permits exploratory benchmarking, not anatomical certification. Every
current output remains `insufficient_evidence` for a quality-admitted assessment.
Research probabilities and statistical bands are shown separately; actual
quality-admitted coverage is zero. No image models or registrations are launched.

## Frozen features

Shared evaluation/inference code is `assessment/features.py`. The input is the
version-1 historical manifest used by the measurement report. Its current scan has
no rating; current expert labels are stored separately in cohort targets. All
historical scans end at the assessed week. Preoperative visits and postoperative
visits after the first history element are rejected. Current label fields and
unsupported metadata are rejected by model extraction.

There are 27 automated features: interval in weeks, elapsed weeks since first
valid history visit, number of prior visits, plus eight features for each of
necrotic/nonenhancing, enhancing core, and edema compartments:

- `log1p` current, previous and historical-nadir volume in mL;
- current minus previous and current minus historical-nadir volume in mL;
- differences in `log1p` volumes versus previous and historical nadir;
- current-minus-previous volume divided by interval in weeks.

Historical nadir excludes the current scan. Zero volumes are valid; `log1p` avoids
infinities, and undefined ordinary percentage changes remain null in reports.
The optional variant adds four previous-rating indicators for SD/PR/CR/PD; it
requires an observed immediately previous rating. Post-Op and unknown are not
silently coded as an observed non-PD rating. No current label is a predictor.

## Frozen evaluation and fitting recipe

Before fitting, write and hash `config.json`, `cohort.json`, `input_manifest.json`,
`splits.json`, the expert-rating file and source files. Save exact source snapshots
and the environment. Refuse to overwrite an existing frozen evaluation.

Use five patient-disjoint outer folds at predefined seeds **71, 1729, 2026** and
three patient-disjoint inner folds (seed = outer seed + fold index + 100).
Use `StratifiedGroupKFold` and fail if either class is missing from any training or
validation split. Materialize and validate every split before fitting. Visits
from one patient never span train and test in a repeat. Repeats are correlated.

Compare enhancing-volume change versus previous scan and historical nadir,
regularized logistic regression and small gradient boosting. Separately evaluate
the previous-rating rule and optional previous-rating learned variant. Include a
**matched volumes-only ablation with exactly the same training visits and outer
and inner splits** as the previous-rating variant.

Fixed modest grid:

- Logistic: training-only StandardScaler, L2 logistic C = 0.1, 1, 10,
  max iterations 2000, random state 71; no class weighting.
- HistGradientBoostingClassifier: 100 iterations, 7 leaves, minimum 20 samples per
  leaf, learning rate .05, L2 = 1 or 10, no early stopping, random state 71.
- Volume score `(current_mL + 1)/(current_mL + reference_mL + 2)`; the 1 mL
  pseudocount is predefined. Also report the untuned binary heuristic: at least
  25% increase **and** at least 1 mL absolute increase. These are not RANO rules.
- Previous-rating rule raw score: 1 for previous PD, else 0; its untuned binary
  operating point is reported separately.

Select each learned model's candidate by pooled inner OOF raw Brier score, with
fixed grid order for ties. Each fold fits its own scaler/model. Fit Platt logistic
calibration (C=1) to the selected candidate's patient-level inner OOF logit scores,
clipped to [1e-6, 1-1e-6] before taking logits. Apply that calibrator to outer test
scores without refitting. Rules use the same calibration recipe. No test labels
are used in selection, scaling, calibration or cutoffs.

From calibrated inner OOF scores, choose the largest low cutoff retaining at
least 90% sensitivity and smallest high cutoff retaining at least 90% specificity;
order the cutoffs conservatively if they overlap. Choose a separate binary cutoff
maximizing training balanced accuracy, using the smallest candidate for ties.
Candidates are observed training probabilities, their next representable larger
values, zero, and 1.0000001. Research bands are lower below the low cutoff, higher
at/above the high cutoff, and defer otherwise. Training sensitivity/specificity
targets are not guaranteed in outer patients. Calibration and threshold derivation
reuse selected-candidate inner OOF labels; only outer results estimate achieved
performance. There are no formal conformal guarantees.

Run with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. The all-cohort refit is fixed
in advance as automated logistic regression, not chosen from outer performance.
Refit its grid/calibration/cutoffs using the same three-fold patient OOF recipe,
seed 71. It is a research export with **no independent accuracy estimate**.

## Reporting and execution

Report outer raw/calibrated probabilities, patient/visit and fold membership,
training-selected thresholds, binary decisions, research bands and quality
insufficiency. Save inner OOF probabilities, candidate selection scores and patient
membership locally. Report AUROC, average precision, Brier, fixed-bin calibration,
mean predicted/observed risk, calibration gap/ECE, binary sensitivity/specificity,
band coverage/deferral and band operating performance. Separate the subset whose
immediately previous observed rating is SD/PR/CR, with denominators and uncertainty.

Use 2000 percentile bootstrap resamples of whole patients, seed 90210, retaining
all their visits and all three repeat predictions together. Resample patients
present in each analysis subset. One-class samples are omitted only for undefined
metrics; valid counts are saved. Metrics weight visits equally within the pooled
repeats, so patients with more visits contribute more. The bootstrap does not
refit models or capture workflow-selection uncertainty. This is internal evidence.

See [RUNBOOK.md](RUNBOOK.md) for exact commands and [completed results](../results/assessment_v1/REPORT.md).
Source and result checksums, environment, logs and refit bundles live under
`runs/assessment_v1/`. The local completion record is
`runs/assessment_build/FINAL_REPORT.md`. No release is published or pushed.
