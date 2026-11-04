# Approach screen: which questions can LUMIERE answer?

Patient-level 5-fold CV x 3 fold seeds; out-of-fold predictions averaged over seeds; 95% CIs from 2000 patient resamples.

## A. Long-horizon volume forecasting (1473 within-segment pairs, 79 patients)

MAE in log1p(mL), mean of necrosis / enhancing / edema. Skill = 1 - MAE / copy-last MAE.

| horizon | pairs (patients) | copy-last | patient slope | drift | Gompertz | GBM | GBM skill | GBM - drift [95% CI] | GBM - Gompertz [95% CI] |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| all horizons | 1473 (79) | 0.886 | 1.683 | 0.882 | 0.776 | 0.773 | +12.8% | -0.142 [-0.187, -0.097] | -0.028 [-0.062, +0.002] |
| 1-13 weeks | 249 (63) | 0.687 | 0.944 | 0.683 | 0.654 | 0.655 | +4.6% | -0.027 [-0.081, +0.027] | +0.007 [-0.034, +0.050] |
| 14-26 weeks | 335 (69) | 0.805 | 1.192 | 0.799 | 0.745 | 0.732 | +9.1% | -0.097 [-0.149, -0.048] | -0.028 [-0.062, +0.009] |
| 27-52 weeks | 423 (65) | 0.937 | 1.608 | 0.931 | 0.814 | 0.801 | +14.5% | -0.190 [-0.264, -0.117] | +0.002 [-0.038, +0.040] |
| >52 weeks | 466 (36) | 1.004 | 2.499 | 1.003 | 0.829 | 0.840 | +16.4% | -0.211 [-0.310, -0.104] | -0.032 [-0.108, +0.041] |

## B. Current-visit progression call (374 rated visits, 79 patients, PD rate 0.63)

| score | AUROC, all visits [95% CI] | minus previous rating | AUROC, previous visit not PD [95% CI] | minus enhancing-vs-nadir rule |
| :--- | :---: | :---: | :---: | :---: |
| previous_rating | 0.684 [0.623, 0.740] | +0.000 [+0.000, +0.000] | 0.523 [0.451, 0.604] | -0.126 [-0.219, -0.026] |
| enhancing_change_vs_nadir | 0.711 [0.644, 0.773] | +0.027 [-0.061, +0.119] | 0.649 [0.577, 0.720] | +0.000 [+0.000, +0.000] |
| total_change_vs_last | 0.713 [0.659, 0.766] | +0.029 [-0.052, +0.115] | 0.684 [0.616, 0.749] | +0.035 [-0.055, +0.130] |
| gbm_volumes_only | 0.790 [0.736, 0.841] | +0.107 [+0.033, +0.186] | 0.739 [0.672, 0.809] | +0.090 [+0.012, +0.173] |
| logistic_all | 0.808 [0.755, 0.854] | +0.125 [+0.065, +0.189] | 0.724 [0.658, 0.792] | +0.076 [+0.005, +0.151] |
| gbm_all | 0.821 [0.772, 0.864] | +0.137 [+0.070, +0.209] | 0.731 [0.662, 0.804] | +0.083 [-0.001, +0.171] |

Previous visit not PD: 238 visits, PD rate 0.50.

## C. Dynamic prognosis (73 patients, 454 visits, median remaining survival 52 weeks)

Excluded (death recorded before last scan): Patient-064, Patient-073, Patient-086.

| model | C-index, every visit [95% CI] | minus clinical + time | C-index, first post-op visit [95% CI] | minus clinical + time |
| :--- | :---: | :---: | :---: | :---: |
| clinical_and_time | 0.599 [0.536, 0.659] | +0.000 [+0.000, +0.000] | 0.573 [0.488, 0.653] | +0.000 [+0.000, +0.000] |
| plus_ratings | 0.634 [0.572, 0.691] | +0.035 [+0.010, +0.065] | 0.573 [0.482, 0.657] | +0.000 [-0.030, +0.031] |
| plus_ratings_and_volumes | 0.678 [0.628, 0.722] | +0.078 [+0.022, +0.131] | 0.617 [0.539, 0.692] | +0.044 [-0.000, +0.095] |
