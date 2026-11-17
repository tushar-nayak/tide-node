# Where the automated progression call disagrees with the expert

Held-out probabilities from patient-level CV; 374 rated follow-ups, 79 patients, 236 rated PD. Model: released volumes + masks; operating point: the 'likely PD' threshold (0.696, 80% specificity). The expert's rationale, the LessThan3Months flag and the next scan are used only to explain errors.

Hits 172, misses 64, false alarms 28, true negatives 110. 24 PD scans fell in the lowest band ('PD less likely').

## A. PD scans: found or missed, by the expert's reason

| expert's reason for PD | PD scans | misses | found (masks model) | found (volumes model) | in lowest band | median new enhancing (mL) | median enhancing change vs nadir (log) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2/FLAIR progression only | 18 | 11 | 39% | 28% | 22% | 0.00 | -0.07 |
| measurable lesion growth | 83 | 6 | 93% | 89% | 0% | 3.14 | +1.84 |
| new lesion | 80 | 24 | 70% | 64% | 12% | 0.30 | +0.48 |
| non-measurable / non-target lesion | 54 | 23 | 57% | 61% | 19% | 0.03 | +0.24 |
| other or no text | 1 | 0 | 100% | 100% | 0% | 21.07 | +2.55 |

Previous scan already PD: 22% of misses, 60% of hits.

## B. Non-PD scans: false alarms, by the expert's reason

| expert's reason for not calling PD | scans | false-alarm rate | false alarms | mean P(PD) |
| :--- | ---: | ---: | ---: | ---: |
| growth attributed to treatment (<3 months / irradiated field) | 34 | 29% | 10 | 0.46 |
| no rationale | 89 | 10% | 9 | 0.41 |
| other rationale | 15 | 60% | 9 | 0.62 |

By rating: SD 92 scans, 16% false alarms; PR 20 scans, 45% false alarms; CR 26 scans, 15% false alarms. 8 false alarms carry the LessThan3Months flag.

By previous rating: previous scan PD: 18 non-PD scans, 33% false alarms (6); previous scan not PD: 120 non-PD scans, 18% false alarms (22).

Of the 6 false alarms after a PD rating, 5 were rated PR; the enhancing volume fell since the PD scan in 5, and 1 had no enhancing tumor on either scan (-95% to -25%). The fully automated volumes + masks model, which does not see the rating, also calls 5 of the 5 PR scans and 0 of the others at its own 80%-specificity threshold.

## C. Was the model early? The next scan

| non-PD scans | scans | with a rated next scan | next scan PD or re-operation | share | median weeks to it |
| :--- | ---: | ---: | ---: | ---: | ---: |
| false alarms | 28 | 25 | 13 | 52% | 9 |
| true negatives | 110 | 99 | 49 | 49% | 13 |
| growth attributed to treatment, called | 10 | 9 | 6 | 67% | 12 |
| growth attributed to treatment, not called | 24 | 21 | 18 | 86% | 12 |

## D. Segmentation quality

Disagreement = |difference in enhancing log-change since the previous scan| between DeepBraTumIA and HD-GLIO on the native grid (374 scans with both).

| scans | n | PD rate | AUROC (masks model) [95% CI] | AUROC (volumes model) | error rate |
| :--- | ---: | ---: | :---: | :---: | ---: |
| segmenters agree (lowest third) | 125 | 0.60 | 0.848 [0.795, 0.901] | 0.832 | 21% |
| middle third | 124 | 0.66 | 0.826 [0.730, 0.916] | 0.803 | 25% |
| segmenters disagree (highest third) | 125 | 0.63 | 0.831 [0.755, 0.901] | 0.833 | 28% |

Median disagreement: 0.24 on errors, 0.14 on correct calls.

## E. Time since surgery

| weeks since surgery | scans | PD rate | AUROC [95% CI] | sensitivity | false-alarm rate |
| :--- | ---: | ---: | :---: | ---: | ---: |
| 0-13 weeks | 46 | 0.46 | 0.762 [0.594, 0.895] | 62% | 20% |
| 14-26 weeks | 93 | 0.48 | 0.795 [0.686, 0.889] | 58% | 17% |
| >26 weeks | 235 | 0.72 | 0.840 [0.785, 0.887] | 78% | 23% |

## F. Concentration

92 errors in 55 of 79 patients; the 10 patients with most errors hold 37% of them. Every error is listed in errors.csv.

CIs: 2000 patient resamples.
