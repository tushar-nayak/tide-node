# Original intensity-forecast model, re-tested on held-out patients

Patient-level 5-fold CV (5 folds done): 399 transitions, 79 patients.
Persistence = latest history scan registered to the target. Slices: 5 centre axial slices, 4 modalities.
rel. improvement = per-patient mean of (persistence - model) / persistence, then averaged over patients (the statistic behind the original +6.7%); positive = model better. 95% bootstrap CI over patients.

| metric | model | persistence | rel. improvement [95% CI] | patients better |
| :--- | ---: | ---: | :---: | ---: |
| mse | 0.005590 | 0.006326 | +8.7% [+5.7%, +11.7%] | 64/79 |
| focal_mse | 0.012787 | 0.013471 | +2.5% [-1.0%, +6.0%] | 50/79 |
| mae | 0.036163 | 0.038931 | +5.4% [+3.6%, +7.2%] | 64/79 |
| mse_FLAIR | 0.005993 | 0.006900 | +5.9% [+1.3%, +10.1%] | 66/79 |
| mse_T1 | 0.007163 | 0.008220 | -9.9% [-21.7%, +1.5%] | 55/79 |
| mse_T2 | 0.006414 | 0.006989 | +7.3% [+4.9%, +9.5%] | 69/79 |
| mse_CT1 | 0.002791 | 0.003193 | +1.7% [-3.6%, +6.9%] | 63/79 |
