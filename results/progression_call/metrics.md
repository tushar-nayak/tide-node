# Automated progression call (RANO PD at the current scan)

- 374 expert-rated follow-up scans (PD / SD / PR / CR), 79 patients, PD rate 0.63.
- Patient-level 5-fold CV x 3 fold seeds; out-of-fold probabilities averaged over seeds; 95% CIs from 2000 patient resamples. Inputs: this and earlier scans only, plus the previous expert rating and age/sex/MGMT.

| score | AUROC [95% CI] | minus previous rating | minus released volume model | AUROC, previous scan not PD | minus enhancing-vs-nadir rule |
| :--- | :---: | :---: | :---: | :---: | :---: |
| previous_rating | 0.684 [0.622, 0.738] | +0.000 [+0.000, +0.000] | -0.136 [-0.206, -0.069] | 0.523 [0.453, 0.604] | -0.126 [-0.218, -0.017] |
| unconstrained_volumes | 0.821 [0.775, 0.864] | +0.137 [+0.071, +0.205] | +0.001 [-0.007, +0.010] | 0.731 [0.662, 0.798] | +0.083 [-0.003, +0.174] |
| released_volumes | 0.820 [0.774, 0.862] | +0.136 [+0.072, +0.203] | +0.000 [+0.000, +0.000] | 0.728 [0.660, 0.792] | +0.080 [-0.004, +0.170] |
| released_volumes_masks | 0.835 [0.796, 0.871] | +0.151 [+0.093, +0.212] | +0.015 [-0.002, +0.032] | 0.760 [0.701, 0.815] | +0.112 [+0.035, +0.192] |
| reference_registered_masks | 0.838 [0.797, 0.873] | +0.154 [+0.093, +0.217] | +0.018 [-0.000, +0.036] | 0.762 [0.701, 0.818] | +0.113 [+0.034, +0.197] |
| automated_volumes | 0.785 [0.734, 0.835] | +0.101 [+0.030, +0.176] | -0.035 [-0.050, -0.020] | 0.734 [0.672, 0.796] | +0.086 [+0.007, +0.170] |
| automated_volumes_masks | 0.797 [0.753, 0.841] | +0.113 [+0.045, +0.184] | -0.023 [-0.043, -0.001] | 0.756 [0.701, 0.812] | +0.108 [+0.035, +0.186] |
| rule_enhancing_change_vs_nadir | 0.711 [0.649, 0.772] | +0.027 [-0.059, +0.116] | -0.109 [-0.178, -0.044] | 0.649 [0.573, 0.723] | +0.000 [+0.000, +0.000] |
| rule_new_enhancing_ml | 0.713 [0.648, 0.775] | +0.030 [-0.061, +0.117] | -0.106 [-0.175, -0.040] | 0.699 [0.621, 0.761] | +0.050 [-0.020, +0.114] |

Previous scan not PD: 238 scans, PD rate 0.50.

## Operating points (pooled out-of-fold scores)

| score | sensitivity at 80% specificity | sensitivity at 90% specificity | specificity at 90% sensitivity | Brier |
| :--- | ---: | ---: | ---: | ---: |
| previous_rating | 0.57 | 0.25 | 0.12 | 0.201 |
| unconstrained_volumes | 0.67 | 0.58 | 0.44 | 0.168 |
| released_volumes | 0.69 | 0.58 | 0.44 | 0.168 |
| released_volumes_masks | 0.73 | 0.67 | 0.38 | 0.162 |
| reference_registered_masks | 0.73 | 0.67 | 0.43 | 0.161 |
| automated_volumes | 0.63 | 0.45 | 0.33 | 0.184 |
| automated_volumes_masks | 0.64 | 0.56 | 0.36 | 0.181 |
| rule_enhancing_change_vs_nadir | 0.50 | 0.39 | 0.20 | - |
| rule_new_enhancing_ml | 0.54 | 0.39 | 0.00 | - |
| constant PD rate | - | - | - | 0.233 |

## PD scans found, by the expert's stated reason (80% specificity)

| reason | PD scans | released, with atlas masks | released, volumes only |
| :--- | ---: | ---: | ---: |
| T2 progression | 18 | 0.39 | 0.28 |
| T2 progression + lesion | 92 | 0.82 | 0.77 |
| enhancing growth (no rationale text) | 2 | 0.50 | 0.50 |
| new / non-target lesion | 124 | 0.72 | 0.70 |

## Excluded input: the LessThan3Months flag

The rating file flags 26 scans (1 rated PD). Of 78 first follow-ups 9-26 weeks after surgery, only 16 are flagged. Flagged scans grew (median enhancing log-change vs nadir +0.48) while unflagged early scans did not (-0.03; 53% PD). The flag records the rater's decision not to call growth PD, so using it would leak the label.

