# Directly paired patient effects

Difference = ODE minus comparator. Negative favors ODE for MAE/log-volume error; positive favors ODE for Dice. Equal patient weights; percentile patient bootstrap; repeated volume losses averaged within patient, not independent replicates.

Bootstrap draws: 20000; seed: 20261103. CIs are descriptive, unadjusted for multiple comparisons and conditional on saved fitted models.

| Comparison | Patients | Rows | Difference | 95% CI |
|---|---:|---:|---:|---:|
| volume: neural_ode - ridge MAE | 79 | 1197 | +0.007882 | [-0.013699, +0.029005] |
| volume: neural_ode - gbm MAE | 79 | 1197 | +0.004022 | [-0.012175, +0.019036] |
| volume: neural_ode - gompertz MAE | 79 | 1197 | -0.005003 | [-0.032286, +0.021828] |
| volume: neural_ode - persistence MAE | 79 | 1197 | -0.080480 | [-0.120836, -0.041999] |
| volume_quick: neural_ode - ridge MAE | 79 | 399 | -0.003061 | [-0.028366, +0.021361] |
| volume_quick: neural_ode - gbm MAE | 79 | 399 | -0.011770 | [-0.032093, +0.009082] |
| volume_quick: neural_ode - gompertz MAE | 79 | 399 | -0.016423 | [-0.041706, +0.009642] |
| volume_quick: neural_ode - persistence MAE | 79 | 399 | -0.092340 | [-0.134356, -0.051430] |
| atlas: ode - static dice_mean | 79 | 399 | -0.001851 | [-0.007763, +0.004065] |
| atlas: ode - static dice_wt | 79 | 399 | +0.000205 | [-0.004611, +0.004707] |
| atlas: ode - static dice_tc | 79 | 399 | -0.002670 | [-0.007286, +0.001627] |
| atlas: ode - static dice_et | 79 | 399 | -0.003088 | [-0.018259, +0.011898] |
| atlas: ode - static change_dice_mean | 79 | 399 | +0.008538 | [-0.001048, +0.018089] |
| atlas: ode - static change_dice_wt | 79 | 399 | +0.023888 | [+0.012775, +0.037524] |
| atlas: ode - static change_dice_tc | 79 | 399 | +0.008777 | [+0.000759, +0.017403] |
| atlas: ode - static change_dice_et | 79 | 399 | -0.007049 | [-0.029575, +0.014064] |
| atlas: ode - static logvol_err_mean | 79 | 399 | +0.020378 | [-0.004614, +0.048323] |
| atlas: ode - static logvol_err_wt | 79 | 399 | -0.016825 | [-0.033500, -0.000755] |
| atlas: ode - static logvol_err_tc | 79 | 399 | +0.008665 | [-0.009214, +0.027263] |
| atlas: ode - static logvol_err_et | 79 | 399 | +0.069295 | [-0.001820, +0.153085] |
| registered: ode - static dice_mean | 79 | 399 | -0.003117 | [-0.008014, +0.001281] |
| registered: ode - static dice_wt | 79 | 399 | -0.000992 | [-0.005605, +0.003548] |
| registered: ode - static dice_tc | 79 | 399 | -0.000793 | [-0.004620, +0.003630] |
| registered: ode - static dice_et | 79 | 399 | -0.007566 | [-0.018887, +0.001121] |
| registered: ode - static change_dice_mean | 79 | 399 | -0.004187 | [-0.017514, +0.008550] |
| registered: ode - static change_dice_wt | 79 | 399 | -0.001204 | [-0.020832, +0.018797] |
| registered: ode - static change_dice_tc | 79 | 399 | -0.008941 | [-0.022650, +0.003479] |
| registered: ode - static change_dice_et | 79 | 399 | -0.002417 | [-0.017434, +0.014224] |
| registered: ode - static logvol_err_mean | 79 | 399 | -0.021345 | [-0.042809, -0.004821] |
| registered: ode - static logvol_err_wt | 79 | 399 | -0.005983 | [-0.023759, +0.010917] |
| registered: ode - static logvol_err_tc | 79 | 399 | +0.004760 | [-0.010519, +0.020975] |
| registered: ode - static logvol_err_et | 79 | 399 | -0.062811 | [-0.115557, -0.021303] |
