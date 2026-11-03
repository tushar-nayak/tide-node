# Step 3: next-visit RANO progression (PD vs SD/PR/CR)

374 expert-rated transitions, 79 patients, PD rate 63%. Patient-level 5-fold CV x 3 fold seeds (step 1's folds); out-of-fold scores averaged over seeds; 95% patient-bootstrap CIs.

| score | AUROC [95% CI] | minus small-tumor rule [95% CI] | minus last-rating rule [95% CI] | Brier |
| :--- | :---: | :---: | :---: | ---: |
| small tumor grows (-last volume) | 0.565 [0.491, 0.642] | - | -0.118 [-0.207, -0.026] | nan |
| interval length | 0.428 [0.363, 0.500] | -0.137 [-0.219, -0.049] | -0.256 [-0.341, -0.151] | nan |
| last visit's RANO rating | 0.684 [0.621, 0.740] | +0.118 [+0.026, +0.207] | - | 0.201 |
| logistic, volumes + timing + clinical | 0.593 [0.530, 0.657] | +0.028 [-0.053, +0.114] | -0.090 [-0.158, -0.020] | 0.230 |
| logistic, + last rating | 0.686 [0.620, 0.747] | +0.121 [+0.034, +0.209] | +0.002 [-0.042, +0.053] | 0.212 |
| gradient boosting, all features | 0.723 [0.660, 0.781] | +0.158 [+0.083, +0.235] | +0.040 [-0.015, +0.092] | 0.204 |
| step 1 GBM predicted volume change | 0.641 [0.569, 0.714] | +0.076 [+0.029, +0.124] | -0.042 [-0.121, +0.041] | nan |
| step 1 Neural ODE predicted volume change | 0.633 [0.563, 0.708] | +0.068 [+0.031, +0.108] | -0.051 [-0.133, +0.037] | nan |

Brier of always predicting the PD rate: 0.233. Oracle (true next-scan volume change, not available at forecast time): AUROC 0.713.

PD rate by the last visit's rating: PD 87% (n=136), SD 53% (n=79), PR 53% (n=17), CR 32% (n=25), Post-Op 45% (n=98), unrated 79% (n=19).

Within the last visit's rating (does anything predict PD beyond "PD stays PD"?):

| subset | n | PD rate | small tumor grows (-last volume) | logistic, volumes + timing + clinical | logistic, + last rating | gradient boosting, all features | step 1 GBM predicted volume change | step 1 Neural ODE predicted volume change |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| last not PD (SD/PR/CR/Post-Op/unrated) | 238 | 50% | 0.576 [0.494, 0.652] | 0.513 [0.449, 0.589] | 0.556 [0.479, 0.637] | 0.599 [0.527, 0.675] | 0.589 [0.513, 0.668] | 0.605 [0.530, 0.683] |
| last PD | 136 | 87% | 0.576 [0.442, 0.718] | 0.513 [0.364, 0.633] | 0.508 [0.361, 0.637] | 0.653 [0.521, 0.768] | 0.655 [0.505, 0.784] | 0.557 [0.409, 0.699] |

Secondary, response (PR/CR, rate 12%) vs not: last visit's RANO rating 0.782 [0.673, 0.851]; logistic, + last rating 0.784 [0.697, 0.846]; gradient boosting, all features 0.731 [0.613, 0.809]; large tumor (+last volume) 0.473 [0.331, 0.606].
