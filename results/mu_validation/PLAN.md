# External validation on MU-Glioma-Post: analysis plan

Written 2026-11-17, before any model was applied to MU data. `scripts/mu_freeze_model.py`
records this file's SHA-256 in `frozen.json`, and `scripts/mu_validation.py` refuses to run if
the file has changed since.

**Looked at before freezing:** the clinical table's columns and counts of patients, scans and
progression dates, to judge feasibility. No volume was linked to an outcome and no model was
applied to MU.

## Data

MU-Glioma-Post (University of Missouri; TCIA, CC BY 4.0; Yaseen et al., Sci Data 2025,
doi:10.7937/7K9K-3C83).
* **Segmentations:** nnU-Net masks refined by neuroradiologists, with labels NETC (1), SNFH (2), ET (3) and resection cavity (4).
* **Clinical table:** days from diagnosis to each MRI, first surgery, radiotherapy, first progression and death.

## Model

Frozen by `scripts/mu_freeze_model.py` before MU outcomes are read.
* **Primary model:** the fully automated two-compartment progression call.
  * Same features, monotone constraints and GBM settings as the volumes-only swap models in `scripts/progression_robustness.py`.
  * Trained on all 374 LUMIERE rated follow-ups with DeepBraTumIA atlas volumes.
  * No expert rating, clinical factors or registration.
* **Secondary model:** the same, trained on HD-GLIO native volumes.
* **Thresholds:** fixed from LUMIERE held-out scores on follow-ups whose previous scan was not rated PD, because MU's labels describe first progression. Cutoffs at 80% specificity and at 90% sensitivity.

## Mapping MU to the model's inputs

* **Compartments:** enhancing = ET; non-enhancing = NETC + SNFH; the resection cavity is ignored.
* **Volumes:** voxel count × voxel volume of the released masks.
* **Scan time:** weeks = days from diagnosis / 7. Scans before the first surgery are dropped. A scan enters only if both its day and its mask exist.
* **History:** all earlier MU scans of the patient.
* **Segment start** (feature `years_since_segment_start`): the first-surgery day. LUMIERE's segment starts at the post-operative scan, which MU does not include.

## Cohort and labels

* Primary diagnosis GBM, at least two scans.
* **PD scan:** the scan within 14 days of the documented first-progression day (the closest one; the earlier one on ties).
* **Non-PD scans:**
  * scans more than 14 days before that day;
  * all scans of patients without documented progression.
* **Excluded:**
  * scans after the PD window;
  * patients with progression but no recorded day.
* **Evaluable calls:** scans with at least one earlier MU scan. A PD scan that is the patient's first MU scan cannot be called.

## Endpoints

**Primary:** AUROC of the primary model, PD vs non-PD, with a 95% CI from 2000 patient resamples.
* The primary hypothesis is AUROC > 0.5, meaning the lower 95% bound is above 0.5.
* The benchmark is the same model's LUMIERE held-out AUROC on follow-ups whose previous scan was not PD, recorded in `frozen.json`. The result is called consistent with LUMIERE if the MU interval contains that value.

**Secondary:**
* AUROC of the secondary model.
* Sensitivity and specificity at the frozen thresholds.
* Brier score, and mean probability against the PD rate.
* Comparators:
  * enhancing change vs nadir (AUROC);
  * enhancing change vs previous scan (AUROC);
  * the volume rule (enhancing ≥ 40% and ≥ 1 mL above the nadir of earlier scans) as a binary call.

**Sensitivity analyses:**
* only patients whose first progression was documented by imaging (types 1 and 3);
* no re-operation (`Multiple surgeries` = 0);
* PD scans on the exact progression day only;
* all glioma types.

**Radiotherapy window:** specificity at the 80%-specificity threshold and AUROC, for scans up to 12 weeks after the end of radiotherapy against later scans. This is the RANO pseudoprogression window, which LUMIERE cannot date.

**Volume source check:** the TCIA volume spreadsheet does not label most rows by scan.
* Rows with a `-Post-treatment_k` suffix are assigned to timepoint k.
* Unsuffixed rows are assigned in timepoint order when a label's row count equals the patient's number of scans. A label with no rows is zero. Any other count makes the label unknown, and the patient is skipped.
* Spreadsheet volumes are compared with mask volumes, and the endpoints are repeated on them.
* If the mask package cannot be obtained, the spreadsheet run is reported as the main analysis, labelled as such.

## Survival: external test of the volume-rule lead (EXPERIMENTS 12c, 12f)

**Patients and time scale:**
* The GBM cohort above, excluding anyone whose recorded death precedes their last scan.
* Time origin is diagnosis. Patients enter at their first MU scan.
* Exit is at death (event) or the last documented day in the record (censored).

**Definitions,** each evaluated at MU scans that have an earlier scan:
* **documented progression:** the first such scan on or after (first-progression day − 14);
* **volume rule:** as above;
* **automated call:** primary-model probability ≥ 0.5, as in `progression_survival.py`;
* **secondary:** documented progression at its recorded day, which is not restricted to scans.

**Primary:** time-dependent C-index of the volume rule minus documented progression, with a 95% CI from 2000 patient resamples. Censoring-aware time-dependent Cox hazard ratios are also reported.
* **Replicates:** the CI is above 0.
* **Consistent:** positive, but the CI includes 0.
* **Does not replicate:** ≤ 0.
