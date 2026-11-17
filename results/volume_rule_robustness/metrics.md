# Volume rule vs expert: robustness of the survival result

73 patients, all deaths observed; time-dependent Cox and C-index as in progression_survival.py; 95% CIs from 2000 patient resamples.

## 1. Rule (>= 40% and >= 1 mL above nadir) from three volume sources

| definition | progressed | median weeks start -> call | median weeks call -> death | HR [95% CI] | C-index [95% CI] | C minus expert [95% CI] |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: |
| expert | 67 | 27 | 48 | 2.14 [0.84, 9.23] | 0.539 [0.486, 0.593] | +0.000 [+0.000, +0.000] |
| volume rule, DeepBraTumIA (atlas) | 67 | 23 | 45 | 3.91 [1.78, 13.00] | 0.607 [0.548, 0.663] | +0.068 [+0.009, +0.130] |
| volume rule, DeepBraTumIA (native) | 67 | 23 | 45 | 3.91 [1.78, 13.00] | 0.607 [0.548, 0.663] | +0.068 [+0.009, +0.130] |
| volume rule, HD-GLIO (native) | 66 | 26 | 46 | 3.09 [1.39, 9.74] | 0.597 [0.538, 0.651] | +0.058 [+0.000, +0.114] |

Patient-level surrogacy: Spearman correlation of progression-free survival (progression or death) with overall survival.

| definition | median PFS (weeks) | Spearman PFS-OS [95% CI] | minus expert [95% CI] |
| :--- | ---: | :---: | :---: |
| expert | 27 | 0.210 [-0.031, 0.426] | +0.000 [+0.000, +0.000] |
| volume rule, DeepBraTumIA (atlas) | 25 | 0.309 [0.063, 0.519] | +0.099 [-0.123, +0.315] |
| volume rule, DeepBraTumIA (native) | 25 | 0.309 [0.063, 0.519] | +0.099 [-0.123, +0.315] |
| volume rule, HD-GLIO (native) | 27 | 0.241 [-0.005, 0.456] | +0.031 [-0.188, +0.241] |

DeepBraTumIA vs HD-GLIO rule: both called progression in 64 patients; same scan in 53, HD-GLIO earlier in 4, later in 7.

## 2. Threshold grid: C-index minus expert [95% CI]

### DeepBraTumIA (atlas)

| above nadir | >= 0 mL | >= 0.5 mL | >= 1 mL | >= 2 mL |
| :--- | :---: | :---: | :---: | :---: |
| 25% | +0.008 [-0.052, +0.065] | +0.065 [+0.006, +0.126] | +0.068 [+0.009, +0.130] | +0.069 [+0.006, +0.133] |
| 40% | +0.008 [-0.052, +0.066] | +0.065 [+0.006, +0.126] | +0.068 [+0.009, +0.130] | +0.069 [+0.006, +0.133] |
| 65% | +0.007 [-0.054, +0.066] | +0.063 [+0.001, +0.126] | +0.066 [+0.004, +0.130] | +0.069 [+0.007, +0.133] |
| 100% | -0.002 [-0.060, +0.050] | +0.050 [-0.004, +0.109] | +0.053 [-0.003, +0.113] | +0.056 [-0.002, +0.116] |

### HD-GLIO (native)

| above nadir | >= 0 mL | >= 0.5 mL | >= 1 mL | >= 2 mL |
| :--- | :---: | :---: | :---: | :---: |
| 25% | -0.015 [-0.068, +0.037] | +0.059 [+0.001, +0.115] | +0.058 [+0.000, +0.114] | +0.063 [+0.006, +0.119] |
| 40% | -0.015 [-0.068, +0.037] | +0.059 [+0.001, +0.115] | +0.058 [+0.000, +0.114] | +0.063 [+0.006, +0.119] |
| 65% | -0.012 [-0.066, +0.039] | +0.061 [+0.003, +0.118] | +0.058 [+0.000, +0.114] | +0.063 [+0.006, +0.119] |
| 100% | -0.024 [-0.073, +0.022] | +0.048 [-0.004, +0.098] | +0.045 [-0.006, +0.095] | +0.063 [+0.006, +0.119] |

## 3. Landmarks (progressed by the landmark; fixed covariate)

| definition | 26 weeks: progressed, HR [95% CI] | 52 weeks: progressed, HR [95% CI] |
| :--- | :---: | :---: |
| expert | 32/70, 1.06 [0.65, 1.76] | 44/53, 1.82 [0.70, 4.60] |
| volume rule, DeepBraTumIA (atlas) | 35/70, 1.10 [0.66, 1.80] | 36/53, 1.44 [0.76, 2.57] |
| volume rule, DeepBraTumIA (native) | 35/70, 1.10 [0.66, 1.80] | 36/53, 1.44 [0.76, 2.57] |
| volume rule, HD-GLIO (native) | 32/70, 1.02 [0.61, 1.66] | 34/53, 1.09 [0.54, 1.89] |
