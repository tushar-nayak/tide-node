# Completed internal assessment benchmark

All LUMIERE results are exploratory internal estimates of expert current-visit PD ratings. No independent patients, anatomical certification, clinical readiness, or formal conformal guarantee.

Intervals resample whole patients 2000 times, retaining all their visits and three repeat predictions. They describe patient sampling, not uncertainty from reselecting the pipeline or changing cohorts. Patients with many visits contribute more to visit-level metrics. Research bands defer between training-selected cutoffs. Quality-admitted coverage is **0%** because anatomy is unverified.

| Variant / model | Patients / visits / PD | AUROC (95% CI) | AP | Brier | Sensitivity | Specificity | Band coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| previous_rating / boosted | 62 / 252 / 173 | 0.839 (0.784–0.889) | 0.926 | 0.152 | 0.755 | 0.747 | 0.688 |
| previous_rating / enhancing_nadir | 62 / 252 / 173 | 0.733 (0.660–0.809) | 0.855 | 0.186 | 0.775 | 0.654 | 0.452 |
| previous_rating / enhancing_previous | 62 / 252 / 173 | 0.748 (0.691–0.810) | 0.873 | 0.185 | 0.586 | 0.840 | 0.603 |
| previous_rating / logistic | 62 / 252 / 173 | 0.808 (0.740–0.870) | 0.903 | 0.166 | 0.730 | 0.806 | 0.622 |
| previous_rating / previous_rating_rule | 62 / 252 / 173 | 0.692 (0.601–0.768) | 0.803 | 0.179 | 0.665 | 0.785 | 0.000 |
| volumes / boosted | 79 / 371 / 233 | 0.811 (0.758–0.859) | 0.873 | 0.169 | 0.718 | 0.766 | 0.595 |
| volumes / enhancing_nadir | 79 / 371 / 233 | 0.705 (0.642–0.766) | 0.803 | 0.206 | 0.714 | 0.609 | 0.421 |
| volumes / enhancing_previous | 79 / 371 / 233 | 0.679 (0.619–0.737) | 0.774 | 0.214 | 0.569 | 0.744 | 0.406 |
| volumes / logistic | 79 / 371 / 233 | 0.772 (0.715–0.827) | 0.844 | 0.184 | 0.621 | 0.790 | 0.523 |
| volumes_matched_previous / boosted | 62 / 252 / 173 | 0.781 (0.712–0.850) | 0.877 | 0.170 | 0.709 | 0.700 | 0.552 |
| volumes_matched_previous / enhancing_nadir | 62 / 252 / 173 | 0.733 (0.660–0.809) | 0.855 | 0.186 | 0.775 | 0.654 | 0.452 |
| volumes_matched_previous / enhancing_previous | 62 / 252 / 173 | 0.748 (0.691–0.810) | 0.873 | 0.185 | 0.586 | 0.840 | 0.603 |
| volumes_matched_previous / logistic | 62 / 252 / 173 | 0.757 (0.667–0.838) | 0.859 | 0.185 | 0.576 | 0.861 | 0.496 |

## New-progression subset

Previous visit had observed SD/PR/CR; current visits include both PD and non-PD. Denominators below count unique visits; repeat predictions stay together in patient bootstrap.

| Variant / model | Patients / visits / PD | AUROC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
| --- | --- | --- | --- | --- |
| previous_rating / boosted | 49 / 120 / 58 | 0.696 (0.585–0.797) | 0.489 (0.398–0.576) | 0.769 (0.627–0.881) |
| previous_rating / enhancing_nadir | 49 / 120 / 58 | 0.665 (0.569–0.762) | 0.707 (0.593–0.810) | 0.672 (0.477–0.830) |
| previous_rating / enhancing_previous | 49 / 120 / 58 | 0.700 (0.601–0.800) | 0.575 (0.463–0.685) | 0.796 (0.680–0.889) |
| previous_rating / logistic | 49 / 120 / 58 | 0.648 (0.528–0.775) | 0.339 (0.244–0.441) | 0.898 (0.797–0.974) |
| previous_rating / previous_rating_rule | 49 / 120 / 58 | 0.438 (0.340–0.545) | 0.000 (0.000–0.000) | 1.000 (1.000–1.000) |
| volumes / boosted | 49 / 120 / 58 | 0.748 (0.654–0.838) | 0.701 (0.600–0.802) | 0.694 (0.529–0.857) |
| volumes / enhancing_nadir | 49 / 120 / 58 | 0.671 (0.577–0.765) | 0.661 (0.552–0.762) | 0.683 (0.483–0.843) |
| volumes / enhancing_previous | 49 / 120 / 58 | 0.703 (0.601–0.803) | 0.540 (0.433–0.660) | 0.849 (0.733–0.934) |
| volumes / logistic | 49 / 120 / 58 | 0.691 (0.587–0.784) | 0.586 (0.470–0.694) | 0.753 (0.603–0.872) |
| volumes_matched_previous / boosted | 49 / 120 / 58 | 0.731 (0.634–0.824) | 0.649 (0.542–0.753) | 0.688 (0.544–0.829) |
| volumes_matched_previous / enhancing_nadir | 49 / 120 / 58 | 0.665 (0.569–0.762) | 0.707 (0.593–0.810) | 0.672 (0.477–0.830) |
| volumes_matched_previous / enhancing_previous | 49 / 120 / 58 | 0.700 (0.601–0.800) | 0.575 (0.463–0.685) | 0.796 (0.680–0.889) |
| volumes_matched_previous / logistic | 49 / 120 / 58 | 0.680 (0.552–0.800) | 0.437 (0.321–0.557) | 0.860 (0.737–0.952) |

Machine-readable metrics contain intervals for every measure, calibration gaps/ECE, band operating performance, fixed heuristic performance, per-seed results and valid bootstrap counts. Platt calibration and cutoffs use selected-candidate inner OOF scores within each outer training set. The all-cohort logistic refit has no independent accuracy claim. Rule heuristics are not RANO criteria.

Provenance: `freeze.json`, `config.json`, `input_manifest.json`, `cohort.json`, `splits.json`; local training audits and environment under `runs/assessment_v1/`.
