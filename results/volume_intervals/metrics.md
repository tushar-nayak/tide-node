# Step 1b: calibrated ranges for the next-visit tumor volume forecast

- Cohort: 79 patients, 399 transitions; exclusions {'pre_op_scans': 39, 'missing_volumes': 0, 'missing_seg': 0, 'surgery_targets': 31}
- Outer: patient-level 5-fold CV, fold seeds [0, 1, 2] (same assignments as step 1). Every transition is scored once per fold seed on a model and calibration that never saw its patient.
- Inner: calibration refit on training patients only: 5-fold out-of-fold GBM residuals, fold seeds [0, 1, 2].
- Point model: released contract `tide-research-all-cohort-gbm-v1-existing-baseline-settings`. Check against step 1 held-out GBM forecasts: max abs difference 0.0 log1p(mL) over 1197 rows (0 unmatched).
- Units: log1p(mL). Total = log1p(sum of compartment mL). Interval score = width + (2/alpha) x miss distance (lower is better).
- Patient coverage = mean over patients of each patient's coverage; 95% CI from patient bootstrap.

## Held-out coverage, total tumor volume

| Range | Nominal | Coverage | Patient coverage [95% CI] | Below / above | Mean width | Interval score |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: |
| last scan, fixed width | 50% | 0.500 | 0.395 [0.325, 0.465] | 0.251 / 0.249 | 0.704 | 2.198 |
| last scan, fixed width | 80% | 0.802 | 0.757 [0.690, 0.819] | 0.096 / 0.102 | 2.067 | 3.561 |
| last scan, fixed width | 90% | 0.901 | 0.896 [0.851, 0.935] | 0.049 / 0.049 | 3.189 | 4.726 |
| last scan, fixed width | 95% | 0.957 | 0.957 [0.932, 0.978] | 0.022 / 0.022 | 4.661 | 5.572 |
| last scan, scaled width | 50% | 0.506 | 0.462 [0.396, 0.525] | 0.251 / 0.243 | 0.801 | 2.157 |
| last scan, scaled width | 80% | 0.805 | 0.776 [0.722, 0.828] | 0.095 / 0.099 | 2.064 | 3.370 |
| last scan, scaled width | 90% | 0.900 | 0.898 [0.858, 0.934] | 0.048 / 0.053 | 3.065 | 4.157 |
| last scan, scaled width | 95% | 0.955 | 0.956 [0.931, 0.979] | 0.026 / 0.019 | 3.890 | 4.780 |
| GBM forecast, fixed width | 50% | 0.512 | 0.399 [0.331, 0.467] | 0.242 / 0.246 | 0.752 | 1.878 |
| GBM forecast, fixed width | 80% | 0.799 | 0.751 [0.687, 0.811] | 0.099 / 0.101 | 1.735 | 3.069 |
| GBM forecast, fixed width | 90% | 0.895 | 0.891 [0.853, 0.925] | 0.059 / 0.046 | 2.570 | 4.099 |
| GBM forecast, fixed width | 95% | 0.948 | 0.952 [0.928, 0.972] | 0.024 / 0.028 | 3.823 | 5.052 |
| GBM forecast, scaled width (deployed) | 50% | 0.515 | 0.454 [0.385, 0.522] | 0.248 / 0.236 | 0.813 | 1.830 |
| GBM forecast, scaled width (deployed) | 80% | 0.800 | 0.791 [0.739, 0.839] | 0.099 / 0.100 | 1.668 | 2.916 |
| GBM forecast, scaled width (deployed) | 90% | 0.891 | 0.894 [0.859, 0.928] | 0.056 / 0.053 | 2.691 | 3.675 |
| GBM forecast, scaled width (deployed) | 95% | 0.949 | 0.950 [0.924, 0.972] | 0.026 / 0.025 | 3.366 | 4.243 |

## Deployed ranges, every quantity

| Quantity | Nominal | Coverage | Patient coverage [95% CI] | Mean width | Median range vs forecast | Median width (mL) |
| :--- | ---: | ---: | :---: | ---: | :---: | ---: |
| necrosis | 50% | 0.520 | 0.500 [0.439, 0.560] | 1.260 | x0.55 to x1.88 | 9.7 |
| necrosis | 80% | 0.817 | 0.807 [0.754, 0.855] | 2.277 | x0.35 to x3.26 | 22.0 |
| necrosis | 90% | 0.906 | 0.914 [0.884, 0.944] | 3.004 | x0.25 to x5.18 | 38.2 |
| necrosis | 95% | 0.949 | 0.945 [0.917, 0.969] | 3.757 | x0.18 to x8.17 | 63.4 |
| enhancing | 50% | 0.525 | 0.469 [0.408, 0.533] | 1.189 | x0.60 to x2.01 | 4.5 |
| enhancing | 80% | 0.818 | 0.781 [0.721, 0.839] | 2.460 | x0.43 to x5.46 | 16.6 |
| enhancing | 90% | 0.906 | 0.890 [0.843, 0.929] | 3.111 | x0.36 to x9.17 | 29.5 |
| enhancing | 95% | 0.948 | 0.940 [0.908, 0.966] | 3.660 | x0.35 to x14.36 | 48.7 |
| edema | 50% | 0.517 | 0.447 [0.376, 0.516] | 0.740 | x0.75 to x1.33 | 24.3 |
| edema | 80% | 0.801 | 0.784 [0.731, 0.832] | 1.738 | x0.51 to x1.96 | 62.9 |
| edema | 90% | 0.891 | 0.890 [0.854, 0.925] | 2.622 | x0.37 to x2.88 | 109.6 |
| edema | 95% | 0.942 | 0.944 [0.918, 0.967] | 3.366 | x0.28 to x3.97 | 158.0 |
| total | 50% | 0.515 | 0.454 [0.385, 0.522] | 0.813 | x0.76 to x1.49 | 38.3 |
| total | 80% | 0.800 | 0.791 [0.739, 0.839] | 1.668 | x0.53 to x2.13 | 85.4 |
| total | 90% | 0.891 | 0.894 [0.859, 0.928] | 2.691 | x0.33 to x3.11 | 148.3 |
| total | 95% | 0.949 | 0.950 [0.924, 0.972] | 3.366 | x0.24 to x3.90 | 196.4 |

## Is the deployed range sharper than the alternatives?

Patient-level mean interval score, deployed minus alternative (negative = deployed better), 95% CI over patients.

| Quantity | Nominal | vs last scan, fixed | vs last scan, scaled | vs GBM, fixed |
| :--- | ---: | :---: | :---: | :---: |
| necrosis | 80% | -0.750 [-1.131, -0.425] (-17%) | -0.870 [-1.084, -0.672] (-19%) | -0.071 [-0.208, +0.059] (-2%) |
| necrosis | 90% | -0.318 [-0.720, +0.174] (-7%) | -1.132 [-1.389, -0.887] (-20%) | +0.278 [+0.080, +0.500] (+7%) |
| enhancing | 80% | -0.250 [-0.517, +0.004] (-6%) | -0.197 [-0.354, -0.053] (-5%) | -0.079 [-0.285, +0.122] (-2%) |
| enhancing | 90% | -0.121 [-0.587, +0.343] (-2%) | -0.399 [-0.661, -0.166] (-7%) | +0.028 [-0.337, +0.439] (+1%) |
| edema | 80% | -0.629 [-1.064, -0.253] (-16%) | -0.638 [-0.989, -0.330] (-17%) | -0.053 [-0.254, +0.158] (-2%) |
| edema | 90% | -0.657 [-1.283, -0.046] (-14%) | -0.631 [-0.904, -0.356] (-13%) | +0.001 [-0.470, +0.437] (+0%) |
| total | 80% | -0.590 [-0.967, -0.250] (-16%) | -0.608 [-0.857, -0.376] (-16%) | -0.003 [-0.168, +0.156] (-0%) |
| total | 90% | -0.708 [-1.276, -0.202] (-15%) | -0.626 [-0.949, -0.345] (-14%) | -0.029 [-0.450, +0.326] (-1%) |

## Coverage of the 80% total-volume range by subgroup

| Subgroup | Transitions | last scan, fixed | last scan, scaled | GBM, fixed | GBM, scaled (deployed) | deployed mean width |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon 0-8 wk | 88 | 0.818 | 0.780 | 0.856 | 0.799 | 1.324 |
| horizon 9-16 wk | 229 | 0.811 | 0.831 | 0.814 | 0.806 | 1.664 |
| horizon 17-26 wk | 61 | 0.792 | 0.776 | 0.743 | 0.814 | 1.944 |
| horizon >26 wk | 21 | 0.667 | 0.714 | 0.571 | 0.698 | 2.338 |
| single-scan history | 107 | 0.670 | 0.788 | 0.642 | 0.785 | 2.623 |
| two or more history scans | 292 | 0.850 | 0.812 | 0.857 | 0.806 | 1.317 |
| last total volume, lowest third | 133 | 0.737 | 0.789 | 0.799 | 0.805 | 1.802 |
| last total volume, middle third | 133 | 0.877 | 0.855 | 0.807 | 0.822 | 1.586 |
| last total volume, highest third | 133 | 0.792 | 0.772 | 0.792 | 0.774 | 1.614 |
| target rating PD | 236 | 0.857 | 0.833 | 0.829 | 0.823 | 1.583 |
| target rating SD | 92 | 0.667 | 0.746 | 0.688 | 0.768 | 2.032 |
| target rating PR/CR | 46 | 0.804 | 0.804 | 0.826 | 0.797 | 1.542 |
| target rating unrated | 25 | 0.773 | 0.760 | 0.880 | 0.707 | 1.354 |

## Direction calls: ranges that exclude 'no change'

Calls per fold seed are out of 399 transitions. 'Correct' = the volume then moved in the called direction; base rate = how often it moved that way among all transitions.

| Range | Quantity | Nominal | Growth calls | per seed | correct (base rate) | Shrink calls | per seed | correct (base rate) |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| GBM forecast, fixed width | total | 50% | 16.9% | 67 | 0.81 (0.55) | 15.5% | 62 | 0.76 (0.44) |
| GBM forecast, fixed width | total | 80% | 5.3% | 21 | 0.77 (0.55) | 2.4% | 10 | 0.90 (0.44) |
| GBM forecast, fixed width | enhancing | 50% | 8.8% | 35 | 0.57 (0.52) | 0.7% | 3 | 0.75 (0.38) |
| GBM forecast, fixed width | enhancing | 80% | 0.1% | 0 | 0.00 (0.52) | 0.0% | 0 | - (0.38) |
| GBM forecast, scaled width (deployed) | total | 50% | 18.5% | 74 | 0.81 (0.55) | 12.3% | 49 | 0.82 (0.44) |
| GBM forecast, scaled width (deployed) | total | 80% | 5.6% | 22 | 0.84 (0.55) | 1.2% | 5 | 0.93 (0.44) |
| GBM forecast, scaled width (deployed) | enhancing | 50% | 10.9% | 43 | 0.57 (0.52) | 0.7% | 3 | 0.62 (0.38) |
| GBM forecast, scaled width (deployed) | enhancing | 80% | 0.2% | 1 | 0.50 (0.52) | 0.0% | 0 | - (0.38) |

## Calibration stability

Deployed multipliers (lower, upper) of the scaled residual, total volume, across the 15 outer training splits:

- 80%: lower -1.54 to -1.18, upper 1.50 to 1.77
- 90%: lower -2.59 to -2.22, upper 2.22 to 2.58

Observed horizons: 1-155 weeks (95th percentile 27, 99th 48). Ranges for horizons beyond the 99th percentile rest on very few transitions.

