# Step 2: 3D next-visit tumor mask forecasting (LUMIERE, 2 mm, frame: registered)

Patient-level 5-fold CV (same folds as step 1). Persistence = copy the last mask.
change-Dice: Dice between predicted and real change relative to the last mask (persistence scores 0 unless nothing changed).
Patient diff = mean over patients of (model - persistence); 95% bootstrap CI over patients.

| Variant | folds | transitions | metric | WT model / persist | TC model / persist | ET model / persist | patient diff [95% CI] | patients better |
| :--- | ---: | ---: | :--- | :---: | :---: | :---: | :---: | ---: |
| ode | 5 | 399 | dice | 0.573 / 0.570 | 0.408 / 0.406 | 0.266 / 0.278 | +0.003 [-0.003, +0.010] | 30/79 |
| ode | 5 | 399 | change_dice | 0.100 / 0.015 | 0.095 / 0.023 | 0.164 / 0.110 | +0.071 [+0.053, +0.092] | 47/79 |
| ode | 5 | 399 | logvol_err | 0.609 / 0.633 | 0.825 / 0.822 | 0.820 / 0.762 | -0.008 [-0.052, +0.030] | 19/79 |
| static | 5 | 399 | dice | 0.574 / 0.570 | 0.408 / 0.406 | 0.277 / 0.278 | +0.006 [+0.001, +0.012] | 32/79 |
| static | 5 | 399 | change_dice | 0.103 / 0.015 | 0.097 / 0.023 | 0.168 / 0.110 | +0.076 [+0.058, +0.093] | 48/79 |
| static | 5 | 399 | logvol_err | 0.607 / 0.633 | 0.826 / 0.822 | 0.861 / 0.762 | +0.013 [-0.030, +0.050] | 15/79 |

logvol_err: |log1p(predicted mL) - log1p(true mL)|, lower is better.
