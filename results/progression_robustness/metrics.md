# Progression call: robustness checks on LUMIERE

Held-out AUROC (patient-level 5-fold CV x 3 seeds, 2000 patient resamples). 374 rated follow-ups, 79 patients, PD rate 0.63.

## 1. How much does the model rest on the expert?

| inputs | AUROC, all scans | minus released volumes model | AUROC, previous scan not PD | minus released |
| :--- | :---: | :---: | :---: | :---: |
| volumes + rating + clinical (released) | 0.820 [0.775, 0.862] | +0.000 [+0.000, +0.000] | 0.728 [0.659, 0.796] | +0.000 [+0.000, +0.000] |
| volumes + clinical, no rating | 0.779 [0.728, 0.833] | -0.041 [-0.060, -0.022] | 0.725 [0.658, 0.793] | -0.003 [-0.019, +0.014] |
| volumes only | 0.785 [0.736, 0.834] | -0.035 [-0.050, -0.019] | 0.734 [0.666, 0.800] | +0.006 [-0.009, +0.022] |
| volumes + masks + rating + clinical (released) | 0.835 [0.796, 0.870] | +0.015 [-0.002, +0.032] | 0.760 [0.700, 0.820] | +0.032 [+0.003, +0.059] |
| volumes + masks + clinical, no rating | 0.798 [0.752, 0.843] | -0.022 [-0.045, +0.001] | 0.753 [0.694, 0.812] | +0.025 [-0.002, +0.055] |
| volumes + masks only | 0.797 [0.753, 0.840] | -0.023 [-0.044, -0.001] | 0.756 [0.698, 0.815] | +0.028 [-0.003, +0.059] |

Mask models against the released mask model, all scans: volumes + masks + clinical, no rating: -0.037 [-0.058, -0.016]; volumes + masks only: -0.038 [-0.054, -0.020]. Previous scan not PD: volumes + masks + clinical, no rating: -0.007 [-0.025, +0.012]; volumes + masks only: -0.004 [-0.017, +0.013].

## 2. Segmenter swap

374 scans, 79 patients, PD rate 0.63 (0 scans dropped because a native segmentation was missing). Two compartments (enhancing, non-enhancing) for every model. The released three-compartment model scores 0.820 [0.773, 0.861] on these rows.

Agreement of the features between sources (Pearson r):

| sources | enhancing volume | non-enhancing volume | enhancing change vs previous | enhancing change vs nadir | non-enhancing change vs nadir |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dbt_atlas vs dbt_native | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dbt_native vs hd_native | 0.92 | 0.90 | 0.88 | 0.85 | 0.83 |
| dbt_atlas vs hd_native | 0.92 | 0.90 | 0.88 | 0.85 | 0.83 |

### Train -> test, with previous rating and clinical factors

Thresholds are fixed on the training source's own held-out scores (80% specificity and 90% sensitivity there) and applied unchanged.

| train -> test | AUROC | minus DBT atlas -> DBT atlas | specificity at the 80%-spec threshold | sensitivity there | sensitivity at the 90%-sens threshold | Brier | mean P(PD) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| dbt_atlas -> dbt_atlas | 0.826 [0.779, 0.867] | +0.000 [+0.000, +0.000] | 0.80 | 0.67 | 0.90 | 0.167 | 0.65 |
| dbt_atlas -> dbt_native | 0.825 [0.778, 0.866] | -0.001 [-0.004, +0.001] | 0.79 | 0.67 | 0.90 | 0.167 | 0.65 |
| dbt_atlas -> hd_native | 0.829 [0.776, 0.873] | +0.003 [-0.028, +0.035] | 0.79 | 0.69 | 0.89 | 0.165 | 0.65 |
| dbt_native -> dbt_atlas | 0.829 [0.782, 0.870] | +0.003 [-0.002, +0.008] | 0.80 | 0.67 | 0.90 | 0.165 | 0.65 |
| dbt_native -> dbt_native | 0.829 [0.782, 0.870] | +0.003 [-0.002, +0.008] | 0.80 | 0.67 | 0.90 | 0.165 | 0.65 |
| hd_native -> dbt_atlas | 0.819 [0.769, 0.866] | -0.007 [-0.034, +0.021] | 0.77 | 0.71 | 0.90 | 0.169 | 0.67 |
| hd_native -> hd_native | 0.815 [0.766, 0.858] | -0.011 [-0.036, +0.016] | 0.80 | 0.68 | 0.90 | 0.170 | 0.66 |

### Train -> test, volumes only

Thresholds are fixed on the training source's own held-out scores (80% specificity and 90% sensitivity there) and applied unchanged.

| train -> test | AUROC | minus DBT atlas -> DBT atlas | specificity at the 80%-spec threshold | sensitivity there | sensitivity at the 90%-sens threshold | Brier | mean P(PD) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| dbt_atlas -> dbt_atlas | 0.785 [0.728, 0.835] | +0.000 [+0.000, +0.000] | 0.80 | 0.64 | 0.90 | 0.185 | 0.64 |
| dbt_atlas -> dbt_native | 0.785 [0.728, 0.835] | +0.000 [-0.002, +0.002] | 0.80 | 0.64 | 0.90 | 0.185 | 0.64 |
| dbt_atlas -> hd_native | 0.797 [0.736, 0.848] | +0.012 [-0.024, +0.047] | 0.80 | 0.66 | 0.89 | 0.180 | 0.64 |
| dbt_native -> dbt_atlas | 0.787 [0.732, 0.836] | +0.002 [-0.001, +0.005] | 0.80 | 0.64 | 0.90 | 0.184 | 0.64 |
| dbt_native -> dbt_native | 0.787 [0.731, 0.835] | +0.001 [-0.002, +0.005] | 0.80 | 0.64 | 0.90 | 0.184 | 0.64 |
| hd_native -> dbt_atlas | 0.787 [0.731, 0.842] | +0.002 [-0.027, +0.031] | 0.74 | 0.68 | 0.92 | 0.182 | 0.66 |
| hd_native -> hd_native | 0.790 [0.732, 0.841] | +0.004 [-0.025, +0.035] | 0.80 | 0.65 | 0.90 | 0.179 | 0.65 |
