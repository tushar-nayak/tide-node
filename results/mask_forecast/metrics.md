# Step 2: 3D next-visit tumor mask forecasting (LUMIERE, atlas space, 2 mm)

Patient-level 5-fold CV (same folds as step 1). Persistence = copy the last mask.
change-Dice: Dice between predicted and real change relative to the last mask (persistence scores 0 unless nothing changed).
Patient diff = mean over patients of (model - persistence); 95% bootstrap CI over patients.

| Variant | folds | transitions | metric | WT model / persist | TC model / persist | ET model / persist | patient diff [95% CI] | patients better |
| :--- | ---: | ---: | :--- | :---: | :---: | :---: | :---: | ---: |
| ode | 5 | 399 | dice | 0.521 / 0.506 | 0.355 / 0.350 | 0.231 / 0.234 | +0.010 [+0.000, +0.019] | 44/79 |
| ode | 5 | 399 | change_dice | 0.184 / 0.015 | 0.151 / 0.023 | 0.212 / 0.113 | +0.127 [+0.107, +0.148] | 63/79 |
| ode | 5 | 399 | logvol_err | 0.619 / 0.635 | 0.885 / 0.826 | 0.906 / 0.768 | +0.043 [-0.014, +0.092] | 23/79 |
| static | 5 | 399 | dice | 0.521 / 0.506 | 0.360 / 0.350 | 0.239 / 0.234 | +0.011 [+0.005, +0.017] | 49/79 |
| static | 5 | 399 | change_dice | 0.167 / 0.015 | 0.143 / 0.023 | 0.225 / 0.113 | +0.118 [+0.101, +0.135] | 63/79 |
| static | 5 | 399 | logvol_err | 0.624 / 0.635 | 0.868 / 0.826 | 0.845 / 0.768 | +0.023 [-0.031, +0.067] | 24/79 |

logvol_err: |log1p(predicted mL) - log1p(true mL)|, lower is better.
