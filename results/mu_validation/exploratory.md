# MU-Glioma-Post validation: exploratory follow-up

Not pre-specified; written after the frozen call scored AUROC 0.56 (metrics.md). 145 evaluable calls, 48 PD, 76 GBM patients.

## 1. Does enhancing growth come before documented progression?

| scans | n | volume rule positive | median enhancing change vs nadir (log1p mL) | median P(PD), frozen model |
| :--- | ---: | ---: | ---: | ---: |
| progression scans | 48 | 0.62 | 0.52 | 0.84 |
| earlier scans of patients who progressed | 42 | 0.60 | 0.42 | 0.71 |
| scans of patients who never progressed | 55 | 0.44 | 0.16 | 0.73 |

## 2. Can volumes predict MU's labels at all? (patient-level 5-fold CV x 3 seeds within MU)

| score | AUROC [95% CI] |
| :--- | :---: |
| frozen LUMIERE model | 0.563 [0.467, 0.664] |
| MU-trained, same features | 0.572 [0.471, 0.677] |
| MU-trained, + time since radiotherapy | 0.658 [0.543, 0.765] |
