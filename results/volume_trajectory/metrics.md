# Step 1: tumor volume trajectory forecasting (LUMIERE)

- Patients: 79, forecast transitions: 399
- Exclusions: {'pre_op_scans': 39, 'missing_volumes': 0, 'missing_seg': 0, 'surgery_targets': 31}
- Protocol: 5-fold patient-level CV x 3 repeats; neural ODE = 5-seed ensemble
- Error unit: absolute error in log1p(volume, mL), averaged over necrosis / enhancing / edema
- Patient-level diff: mean over patients of (method MAE - persistence MAE); negative = better than persistence; 95% bootstrap CI over patients

| Method | MAE (mean) | necrosis | enhancing | edema | skill vs persistence | patient diff [95% CI] | patients better | AUROC (PD) | direction acc. |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| persistence | 0.7448 | 0.8456 | 0.7745 | 0.6144 | +0.0% | +0.0000 [+0.0000, +0.0000] | 0/79 | 0.500 | nan |
| loglinear | 0.9933 | 1.0994 | 0.9716 | 0.9089 | -33.4% | +0.2194 [+0.1641, +0.2768] | 4/79 | 0.541 | 0.368 |
| gompertz | 0.7192 | 0.7818 | 0.8008 | 0.5750 | +3.4% | -0.0755 [-0.1136, -0.0382] | 62/79 | 0.584 | 0.642 |
| ridge | 0.7207 | 0.7734 | 0.8121 | 0.5767 | +3.2% | -0.0884 [-0.1361, -0.0403] | 50/79 | 0.653 | 0.672 |
| gbm | 0.6931 | 0.7507 | 0.7880 | 0.5407 | +6.9% | -0.0845 [-0.1186, -0.0509] | 58/79 | 0.635 | 0.687 |
| neural_ode | 0.7081 | 0.7664 | 0.8019 | 0.5560 | +4.9% | -0.0805 [-0.1222, -0.0440] | 53/79 | 0.628 | 0.653 |

Reference: AUROC for RANO PD using the *true* next-scan volume change = 0.713 (upper bound for any volume-change score; PD rate 0.63, n=1122 rated targets). Persistence predicts zero change, so its AUROC is 0.5 by construction.

Rows are pooled over repeats (each transition appears once per repeat).
