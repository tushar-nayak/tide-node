# Automated progression call (RANO PD at the current scan)

- 374 expert-rated follow-up scans (PD / SD / PR / CR), 79 patients, PD rate 0.63.
- Patient-level 5-fold CV x 3 fold seeds; out-of-fold probabilities averaged over seeds; 95% CIs from 2000 patient resamples. Inputs: this and earlier scans only, plus the previous expert rating and age/sex/MGMT.

| score | AUROC [95% CI] | minus previous rating | minus volume model (V+R+S) | AUROC, previous scan not PD | minus enhancing-vs-nadir rule |
| :--- | :---: | :---: | :---: | :---: | :---: |
| previous_rating | 0.684 [0.622, 0.738] | +0.000 [+0.000, +0.000] | -0.137 [-0.210, -0.071] | 0.523 [0.453, 0.604] | -0.126 [-0.218, -0.017] |
| gbm_volumes | 0.790 [0.740, 0.841] | +0.107 [+0.035, +0.184] | -0.031 [-0.046, -0.015] | 0.739 [0.673, 0.801] | +0.090 [+0.007, +0.176] |
| gbm_volumes_rating_clinical | 0.821 [0.775, 0.864] | +0.137 [+0.071, +0.205] | +0.000 [+0.000, +0.000] | 0.731 [0.662, 0.798] | +0.083 [-0.003, +0.174] |
| released_monotone_gbm | 0.820 [0.774, 0.862] | +0.136 [+0.072, +0.203] | -0.001 [-0.010, +0.007] | 0.728 [0.660, 0.792] | +0.080 [-0.004, +0.170] |
| gbm_plus_masks | 0.833 [0.792, 0.870] | +0.149 [+0.086, +0.213] | +0.012 [-0.005, +0.029] | 0.754 [0.690, 0.812] | +0.105 [+0.026, +0.188] |
| logistic_plus_masks | 0.824 [0.777, 0.868] | +0.141 [+0.081, +0.200] | +0.003 [-0.040, +0.047] | 0.748 [0.682, 0.815] | +0.100 [+0.036, +0.170] |
| rule_enhancing_change_vs_nadir | 0.711 [0.649, 0.772] | +0.027 [-0.059, +0.116] | -0.110 [-0.180, -0.045] | 0.649 [0.573, 0.723] | +0.000 [+0.000, +0.000] |
| rule_new_enhancing_ml | 0.731 [0.671, 0.791] | +0.047 [-0.045, +0.134] | -0.090 [-0.156, -0.027] | 0.721 [0.645, 0.783] | +0.072 [+0.006, +0.133] |

Previous scan not PD: 238 scans, PD rate 0.50.

## Operating points (pooled out-of-fold scores)

| score | sensitivity at 80% specificity | sensitivity at 90% specificity | specificity at 90% sensitivity | Brier |
| :--- | ---: | ---: | ---: | ---: |
| previous_rating | 0.57 | 0.25 | 0.12 | 0.201 |
| gbm_volumes | 0.63 | 0.48 | 0.38 | 0.181 |
| gbm_volumes_rating_clinical | 0.67 | 0.58 | 0.44 | 0.168 |
| released_monotone_gbm | 0.69 | 0.58 | 0.44 | 0.168 |
| gbm_plus_masks | 0.74 | 0.65 | 0.43 | 0.163 |
| logistic_plus_masks | 0.71 | 0.53 | 0.51 | 0.163 |
| rule_enhancing_change_vs_nadir | 0.50 | 0.39 | 0.20 | - |
| rule_new_enhancing_ml | 0.55 | 0.45 | 0.00 | - |
| constant PD rate | - | - | - | 0.233 |

## PD scans found, by the expert's stated reason (80% specificity)

| reason | PD scans | with mask features | volumes + rating + clinical only |
| :--- | ---: | ---: | ---: |
| T2 progression | 18 | 0.39 | 0.33 |
| T2 progression + lesion | 92 | 0.83 | 0.77 |
| enhancing growth (no rationale text) | 2 | 0.50 | 0.50 |
| new / non-target lesion | 124 | 0.73 | 0.65 |

## Excluded input: the LessThan3Months flag

The rating file flags 26 scans (1 rated PD). Of 78 first follow-ups 9-26 weeks after surgery, only 16 are flagged. Flagged scans grew (median enhancing log-change vs nadir +0.48) while unflagged early scans did not (-0.03; 53% PD). The flag records the rater's decision not to call growth PD, so using it would leak the label.

