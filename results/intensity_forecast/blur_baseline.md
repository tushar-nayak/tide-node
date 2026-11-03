# Control: Gaussian-blurred persistence vs the network (held-out patients)

Blur sigma (pixels, in-plane) tuned on each fold's training patients: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}.
rel. improvement over plain persistence, same statistic as metrics.md; 95% bootstrap CI over patients.

| metric | network | blurred persistence | network minus blur, patient-level [95% CI] |
| :--- | :---: | :---: | :---: |
| mse | +8.7% | +3.7% | +5.0% [+2.1%, +8.0%] |
| focal_mse | +2.5% | +4.1% | -1.6% [-5.1%, +2.0%] |
| mae | +5.4% | -1.2% | +6.7% [+4.8%, +8.5%] |
| mse_FLAIR | +5.9% | +4.9% | +0.9% [-3.6%, +4.9%] |
| mse_T1 | -9.9% | +3.1% | -13.0% [-24.4%, -1.9%] |
| mse_T2 | +7.3% | +3.3% | +4.0% [+1.6%, +6.2%] |
| mse_CT1 | +1.7% | +0.9% | +0.9% [-3.8%, +5.5%] |
