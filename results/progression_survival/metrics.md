# Does the automated progression call predict survival like the expert's?

- 73 patients with consistent survival data, all 73 deaths observed; time origin = first scan of the first surgery-free segment. 22 patients had a re-operation, counted as progression for every definition if none was called earlier.
- Automated calls use held-out probabilities from models with no expert rating and no clinical factors.
- Thresholds: 0.5 (pre-specified) and 'rate-matched', where the automated call flags as many patients as the expert ({"automated (+masks)": {"p>=0.5": 0.5, "rate-matched": 0.365}, "automated (volumes)": {"p>=0.5": 0.5, "rate-matched": 0.485}}).
- 95% CIs from 2000 patient resamples.

## Progression as a time-varying covariate (all follow-up)

A time-varying HR rewards late calls (a call just before death looks maximally 'prognostic'), so read it with the median time from the call to death; the landmark analyses below avoid this bias.

| definition | patients progressed (via re-operation) | median weeks start -> call | median weeks call -> death | HR for death after progression [95% CI] | time-dependent C-index [95% CI] | C minus expert [95% CI] |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: |
| expert | 67 (5) | 27 | 48 | 2.14 [0.80, 10.04] | 0.539 [0.484, 0.590] | +0.000 [+0.000, +0.000] |
| volume rule | 67 (8) | 23 | 45 | 3.91 [1.90, 13.10] | 0.607 [0.549, 0.662] | +0.068 [+0.008, +0.132] |
| automated (+masks) p>=0.5 | 66 (5) | 25 | 50 | 1.68 [0.77, 4.92] | 0.530 [0.481, 0.583] | -0.009 [-0.066, +0.050] |
| automated (+masks) rate-matched | 67 (2) | 23 | 51 | 1.54 [0.60, 6.69] | 0.522 [0.469, 0.580] | -0.016 [-0.066, +0.034] |
| automated (volumes) p>=0.5 | 67 (5) | 25 | 49 | 2.03 [0.85, 8.60] | 0.544 [0.492, 0.599] | +0.006 [-0.038, +0.056] |
| automated (volumes) rate-matched | 67 (5) | 24 | 50 | 1.94 [0.79, 8.34] | 0.541 [0.486, 0.596] | +0.002 [-0.041, +0.051] |

Joint Cox (expert + automated (+masks) p>=0.5): expert HR 1.91 [0.55, 9.44], automated HR 1.30 [0.50, 4.13].

## Landmark 26 weeks after the segment start (70 patients alive)

| definition | progressed by landmark | median remaining weeks, progressed | not progressed | landmark HR [95% CI] |
| :--- | ---: | ---: | ---: | :---: |
| expert | 32 | 52 | 51 | 1.06 [0.64, 1.73] |
| volume rule | 35 | 52 | 51 | 1.10 [0.66, 1.79] |
| automated (+masks) p>=0.5 | 35 | 43 | 59 | 1.13 [0.68, 1.85] |
| automated (+masks) rate-matched | 39 | 43 | 59 | 1.11 [0.64, 1.76] |
| automated (volumes) p>=0.5 | 35 | 43 | 59 | 1.13 [0.68, 1.86] |
| automated (volumes) rate-matched | 37 | 43 | 59 | 1.18 [0.71, 1.92] |

Expert vs automated (+masks, 0.5) status at the landmark: 79% agreement, kappa 0.57.

## Landmark 52 weeks after the segment start (53 patients alive)

| definition | progressed by landmark | median remaining weeks, progressed | not progressed | landmark HR [95% CI] |
| :--- | ---: | ---: | ---: | :---: |
| expert | 44 | 35 | 67 | 1.82 [0.67, 4.38] |
| volume rule | 36 | 32 | 52 | 1.44 [0.77, 2.53] |
| automated (+masks) p>=0.5 | 42 | 32 | 55 | 1.41 [0.68, 2.72] |
| automated (+masks) rate-matched | 46 | 35 | 55 | 1.61 [0.53, 4.41] |
| automated (volumes) p>=0.5 | 43 | 36 | 47 | 1.28 [0.52, 2.70] |
| automated (volumes) rate-matched | 43 | 36 | 47 | 1.28 [0.52, 2.70] |

Expert vs automated (+masks, 0.5) status at the landmark: 81% agreement, kappa 0.39.

Timing, 65 patients progressed under both: automated call earlier in 16, same scan in 34, later in 15 (median difference +0 weeks).

