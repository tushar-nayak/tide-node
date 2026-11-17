# External validation on MU-Glioma-Post

Pre-specified in PLAN.md; model and thresholds frozen on LUMIERE (frozen.json). 95% CIs from 2000 patient resamples. PD = the scan at the documented first progression; non-PD = earlier scans and all scans of patients without progression.

Cohort: 157 GBM patients, 113 with at least two scans; 145 evaluable calls from 76 patients, 48 PD (2 progressed patients without a recorded day excluded).

## 1. Discrimination

| score | AUROC [95% CI] | minus primary [95% CI] |
| :--- | :---: | :---: |
| model trained on DeepBraTumIA volumes (primary) | 0.563 [0.467, 0.658] | +0.000 [+0.000, +0.000] |
| model trained on HD-GLIO volumes | 0.586 [0.486, 0.685] | +0.023 [-0.035, +0.080] |
| enhancing change vs nadir | 0.559 [0.466, 0.653] | -0.004 [-0.095, +0.087] |
| enhancing change vs previous | 0.496 [0.408, 0.588] | -0.067 [-0.179, +0.041] |

LUMIERE benchmark (same model, held out, previous scan not PD): 0.725 [0.655, 0.793].

## 2. Operating points at the frozen LUMIERE thresholds

PD rate 0.33.

| model | specificity at the 80%-spec threshold | sensitivity there | sensitivity at the 90%-sens threshold | specificity there | Brier | mean P(PD) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| model trained on DeepBraTumIA volumes (primary) | 0.51 | 0.60 | 0.92 | 0.11 | 0.384 | 0.69 |
| model trained on HD-GLIO volumes | 0.45 | 0.69 | 0.96 | 0.11 | 0.354 | 0.67 |

Volume rule as a binary call: sensitivity 0.62, specificity 0.49.

## 3. Sensitivity analyses

| subset | calls (PD) | model trained on DeepBraTumIA volumes (primary) | model trained on HD-GLIO volumes | enhancing change vs nadir |
| :--- | ---: | :---: | :---: | :---: |
| imaging-documented progression only (types 1, 3) | 137 (44) | 0.573 [0.471, 0.674] | 0.598 [0.496, 0.701] | 0.578 [0.478, 0.673] |
| no re-operation | 98 (31) | 0.540 [0.428, 0.653] | 0.604 [0.485, 0.715] | 0.527 [0.407, 0.651] |
| PD scans on the exact progression day only | 139 (42) | 0.544 [0.443, 0.644] | 0.574 [0.470, 0.671] | 0.547 [0.451, 0.642] |
| all glioma types | 218 (61) | 0.580 [0.491, 0.664] | 0.629 [0.545, 0.705] | 0.571 [0.490, 0.647] |

## 4. Radiotherapy window (primary model)

| scans | calls (PD) | specificity at the 80%-spec threshold | sensitivity there | AUROC [95% CI] |
| :--- | ---: | ---: | ---: | :---: |
| within 12 weeks after radiotherapy | 29 (5) | 0.46 | 0.20 | 0.283 [0.093, 0.507] |
| later | 94 (42) | 0.48 | 0.67 | 0.571 [0.454, 0.681] |

22 calls have no radiotherapy dates.

## 5. Survival: does the volume rule beat documented progression? (time-dependent, censoring-aware)

99 patients, 57 deaths (14 excluded: death before last scan or no follow-up). Time origin diagnosis; entry at the first MU scan; censored at the last documented day.

| definition | progressed | median days entry -> call | HR [95% CI] | C-index [95% CI] | C minus documented (at scans) [95% CI] |
| :--- | ---: | ---: | :---: | :---: | :---: |
| documented progression (at scans) | 76 | 102 | 7.85 [3.52, 27.91] | 0.671 [0.599, 0.741] | +0.000 [+0.000, +0.000] |
| volume rule | 71 | 78 | 3.07 [1.62, 6.80] | 0.613 [0.529, 0.692] | -0.058 [-0.161, +0.039] |
| automated call (P >= 0.5) | 90 | 77 | 5.49 [2.17, 25.16] | 0.598 [0.544, 0.655] | -0.072 [-0.149, +0.007] |
| documented progression (recorded day) | 90 | 91 | 9.74 [4.16, 54.17] | 0.669 [0.607, 0.726] | -0.002 [-0.049, +0.053] |

## 6. Volume source check (TCIA spreadsheet vs masks)

Spreadsheet rows mapped to scans: {'explicit': 58, 'in_order': 82, 'skipped_ambiguous': 63, 'skipped_mixed': 0}. Of 355 scans in both sources, all four volumes agree exactly for 344.
Endpoints on spreadsheet volumes: 57 calls (18 PD); primary AUROC 0.563 [0.401, 0.713].
