# Tumor Interval Dynamics Estimation via Neural ODE Sequences
## (previously glioblastoma-evolution)

Longitudinal glioblastoma MRI analysis on the LUMIERE dataset (79 patients, 399
scan-to-scan transitions, 374 expert-rated follow-up scans). Every experiment is scored on
patients the model never saw, against the trivial rule for that question, with
patient-level bootstrap confidence intervals.

**What works: an automated progression call.** The released model reads the automated tumor
volumes of the current and earlier scans, the previous expert rating, and age, sex and MGMT.
If the atlas segmentations of the last two scans are given, it also reads the new tumor
outside the prior tumor. It estimates whether an expert would rate the current scan as
progressive disease (PD) under RANO.
* On held-out patients, AUROC is **0.835** [0.80, 0.87] with segmentations and 0.82 [0.77,
  0.86] from volumes alone. Carrying the previous rating forward scores 0.68 (difference
  +0.15 [+0.09, +0.21]).
* For new progression (previous scan not PD), it scores 0.76 with segmentations and 0.73
  without. An enhancing-volume rule scores 0.65.
* About half of all scans fall in its "likely PD" band, and 85-86% of those were rated PD.
* Trained on one automatic segmenter and tested on LUMIERE's other one (HD-GLIO), it keeps
  its accuracy: AUROC 0.829 vs 0.826.
* Without the previous expert rating it loses 0.04 overall but nothing for new progression.
* It finds 93% of PD scans with measurable growth. Its misses are mostly new lesions,
  non-measurable lesions and T2/FLAIR-only progression. A third of its false alarms are
  growth that the expert attributed to treatment.
* It runs on CPU in a few seconds (see Run the models below).
* **It did not transfer to a second centre.** A pre-specified test on MU-Glioma-Post scored
  AUROC 0.56 [0.47, 0.66] against clinically documented progression, against 0.73 on
  comparable LUMIERE scans. A model trained within MU did no better (0.57). Volumes don't
  track MU's documented progression, so LUMIERE alone cannot establish a transferable
  call (EXPERIMENTS.md 12h).

**Survival check.** A fully automated version of the call (no expert rating) was tested
against overall survival. It is as prognostic as the expert's call: time-dependent C-index
0.53-0.54 vs 0.54, with a difference near 0. In about half of patients it flags the same
first-progression scan as the expert, and earlier or later about equally often in the
rest. Neither is a strong survival marker in LUMIERE. A simple volumetric rule (enhancing
tumor ≥ 40% above nadir) did better than the expert: C-index 0.61 vs 0.54 (CI excludes 0).
* It holds with the second segmenter: 0.60.
* It needs at least about 0.5 mL of absolute growth.
* Landmark analyses do not confirm it.
* It did not replicate on MU-Glioma-Post. There, documented progression was the stronger
  marker: C-index 0.67 vs 0.61 for the rule.

**What does not work: forecasting the next visit.** Once the evaluation is leakage-free,
the scans are aligned and trivial controls are included, carrying the last scan forward is
close to the best next-visit forecast. This holds for intensity, the 3D tumor mask and the
next RANO rating. No Neural ODE model beats the best simple baseline for its target. The
volume forecast gains a modest 5-7%, now with calibrated ranges. Full details, numbers and
commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The draft paper
[MANUSCRIPT.md](MANUSCRIPT.md) covers both parts: forecasting, and assessment with the
survival check.

## Findings

| Question | Result | EXPERIMENTS.md § |
| :--- | :--- | :---: |
| Does the original "+6.7% over persistence" hold on unseen patients? | The number holds: whole-slice MSE is +8.7% [+5.7, +11.7]. The tumor-forecasting reading does not: in the tumor region it is +2.5% [-1.0, +6.0], and a 1-pixel blur of the last scan does at least as well (+4.1%). | 8 |
| Can tumor volume be forecast? | Learned models cut volume error 5-7% below persistence (CIs exclude 0). A population Gompertz fit alone gets 3.4%, and the Neural ODE is tied with ridge regression and gradient boosting. | 7 |
| Can the 3D tumor mask be forecast? | In LUMIERE's atlas space, consecutive scans are misaligned by a median 2.5 mm. That misalignment was 40-45% of the change the first models appeared to predict. After registration, Dice is 0.573 vs 0.570 for copy-last-mask (+0.003 [-0.003, +0.010]). | 2b, 7 |
| How reliable are the mask labels? | Two automatic segmenters on the *same* scan agree only to whole-tumor Dice 0.65. Visits rated stable at most 12 weeks apart overlap at 0.58. Much of the visit-to-visit difference may be labelling and alignment rather than tumor change. | 9a |
| Where does the mask model help? | Nowhere specific. It enlarges the tumor in about 80% of cases (truth: 53%), including tumors that later shrank. Its sense of direction is "small tumors grow, big ones shrink." | 9b |
| Can the next RANO rating be forecast? | The last visit's rating alone gives AUROC 0.684. No learned model beats it significantly (best +0.04 [-0.015, +0.092]). For patients not currently progressing, every score is ≤ 0.61. | 10 |
| Can the volume forecast give a trustworthy range? | Yes. On unseen patients, the 80/90/95% ranges held 80/89/95% of next-visit total volumes. Centred on the model, they score 16% better (interval score, CI excludes 0) than equally calibrated ranges around the last scan. They are wide: a typical 80% range spans ×0.53 to ×2.1 of the forecast. | 11 |
| Which other questions can this data answer? | **Is the current scan PD?** Yes: AUROC 0.82 vs 0.68 (CI excludes 0). Remaining survival from each visit: modestly, C-index 0.68 vs 0.60 for clinical factors and time (CI excludes 0). Volume 6-12+ months ahead: beats copy-last by 14-16%, but a 2-parameter population Gompertz model does about as well (CIs include 0). | 12 |
| How good is the progression call? | From volumes + previous rating + clinical: AUROC 0.820 [0.774, 0.862], 69% of PD scans found at 80% specificity. With the new-tumor features from the two atlas segmentations (8 mm margin, no registration): 0.835 [0.796, 0.871], 73% found; new progression 0.760 vs 0.728. A post-radiotherapy flag in the rating file would add more, but raters set it selectively, so it leaks the label and is excluded. | 12b |
| Does the automated call predict survival like the expert's? | Equally well, and equally weakly. Time-dependent C-index is 0.530-0.544 (automated) vs 0.539 (expert), differences near 0 with CIs including 0. The first-progression scan matches in 34 of 65 patients, with the rest split evenly earlier and later. A simple volume rule is more prognostic than the expert: 0.607, +0.068 [+0.008, +0.132]. | 12c |
| Does the call depend on one segmenter, or on the expert? | No. Trained on DeepBraTumIA and tested on HD-GLIO volumes of the same held-out scans, AUROC is 0.829 vs 0.826 (+0.003 [-0.028, +0.035]), and the fixed thresholds keep their specificity (0.79) and sensitivity (0.69). The two segmenters' growth measurements correlate only 0.83-0.88. Without the previous rating, AUROC drops 0.04 overall but not for new progression (0.734 vs 0.728). | 12d |
| Where does it disagree with the expert? | Of 64 misses, 24 are new lesions (70% of new-lesion PD found), 23 non-measurable lesions (57% found) and 11 T2/FLAIR-only progression (39% found); measurable growth is found 93% of the time. Of 28 false alarms, a third are growth the expert attributed to treatment (< 3 months or irradiated field; 24 of 30 such scans were PD or re-operated at the next visit), and a third are partial responses. Segmenter disagreement matters little. | 12e |
| Is the volume rule's survival advantage robust? | Partly. With HD-GLIO volumes it is +0.058 [0.000, +0.114] (DeepBraTumIA +0.068), and any relative threshold from 25% to 65% gives the same. Without a minimum absolute growth (≥ 0.5 mL) it disappears. The PFS–OS correlation points the same way (0.31 vs 0.21, CI includes 0), and landmark analyses show no advantage. | 12f |
| Would more patients help the call? | Not the fully automated one: its AUROC is flat from 32 training patients on (0.797 at 32, 47 and 63). The models with the previous rating still gain about 0.007 per 16 patients. A second dataset is mainly for validation. A temporal split is impossible (no calendar dates in LUMIERE). | 12g |
| Does the call transfer to another centre? | No. In a pre-specified test on MU-Glioma-Post (76 patients with glioblastoma, 145 calls, clinically documented first progression), AUROC is 0.563 [0.467, 0.658] against 0.725 on LUMIERE, with specificity 0.51 at the frozen 80%-specificity threshold. A model trained within MU scores 0.572; growth was as common before documented progression as at it. The volume rule's survival advantage does not replicate: −0.058 [−0.161, +0.039] against documented progression. | 12h |

Figures: `results/figures/`, listed at the end of EXPERIMENTS.md.

## Run the models

Both released models take a small JSON file and run on CPU in seconds. The input is
DeepBraTumIA volumes in mm³ for each visit since the last surgery. See
[PREDICTION.md](PREDICTION.md) for the formats and every check.

**Progression call.** The last visit in `history` is the scan to assess, with no rating.
The visit before it needs its expert rating (`PD`, `SD`, `PR`, `CR`, `Post-Op` or
`unknown`). For new-lesion detection, add `"segmentation": "<path to the atlas-space
seg_mask.nii.gz>"` to the last two visits. The tool then uses its `volumes_masks` model:

```json
{"version": 1, "patient_id": "example",
 "history": [
   {"week": 0,  "rating": "Post-Op", "volumes_mm3": {"Necrotic_NonEnhancing": 3000, "Enhancing_Core": 2500, "Edema_Compartment": 20000}},
   {"week": 12, "rating": "SD",      "volumes_mm3": {"Necrotic_NonEnhancing": 2800, "Enhancing_Core": 2000, "Edema_Compartment": 15000}},
   {"week": 24,                      "volumes_mm3": {"Necrotic_NonEnhancing": 4200, "Enhancing_Core": 5200, "Edema_Compartment": 26000}}],
 "clinical": {"Age at surgery (years)": "61", "Sex": "male", "MGMT qualitative": "not methylated"}}
```

```bash
.venv/bin/python scripts/predict.py --history scan.json \
  --bundle runs/packaging/release-20261103/progression-gbm --trust-progression-bundle --output-dir runs/my-call
```

```text
Scan at week 24: probability of RANO progression 0.89 -> likely PD (on held-out patients, 85% of scans in this band were rated PD)
  Model: volumes (volumes, previous rating and clinical; add both segmentations for new-lesion detection)
  'likely PD' at >= 0.70 (held-out specificity 80%, sensitivity 69%); 'PD less likely' below 0.42 (sensitivity 90%, specificity 44%); held-out AUROC 0.82
  Enhancing tumor 5.2 mL: +160% vs previous (2.0), +160% vs nadir (2.0); total 35.4 mL (+79% vs previous). Previous rating: SD
```

With segmentations, the output adds a line such as `Outside the prior tumor (+8 mm): new
enhancing 0.08 mL in 0 new component(s); new FLAIR abnormality 6.7 mL`. On held-out
patients the three bands were:

| band | volumes model | volumes_masks model |
| :--- | :--- | :--- |
| "likely PD" | 51% of scans, 85% rated PD | 53%, 86% PD |
| "uncertain" | 26%, 49% PD | 26%, 41% PD |
| "PD less likely" | 23%, 28% PD | 20%, 32% PD |

Subtle T2/FLAIR progression is mostly missed (39% found). A stable scan usually lands in
"uncertain" rather than "PD less likely". EXPERIMENTS.md section 12e analyses every error.

**Volume forecast with ranges.** This gives next-visit compartment volumes:

```bash
.venv/bin/python scripts/predict.py --history volume-history.json --horizon-weeks 12 \
  --bundle runs/packaging/release-20261103/volume-all --trust-volume-bundle \
  --intervals runs/packaging/release-20261103/volume-intervals.json --output-dir runs/my-forecast
```

```text
Forecast for week 22 (12 weeks after the last scan), mL:
  ...
  Total                  last    36.5  forecast    40.1  ranges 50% 30.6-56.9  80% 22.1-80.2  90% 13.7-120.6  95% 9.9-150.5
```

* `runs/packaging/` is not in git. Build it with `scripts/build_research_release.sh <new dir>`.
* The 3D mask forecast (`--bundle .../atlas-ode-cv-ensemble.json` with an imaging history) also
  runs, but in held-out testing it was no better than copying the last mask (section 7).
* Inputs must already be LUMIERE-style: atlas-space images and DeepBraTumIA labels. No
  segmentation or registration of raw MRI is included.
* Research use only; not clinically validated.

### What changed from the earlier claim

Earlier versions of this README reported a +6.7% mean MSE improvement over persistence
across 81 patient-level runs (`results/lumiere_full_v1_metrics.*`, kept for reference).
Every row of that table was scored on the patient's own training pairs
(`metric_split=all`). Re-tested on held-out patients, the number survives, but the gain is
smoothing and global intensity correction, not tumor forecasting (section 8).

Three data issues found along the way are documented in EXPERIMENTS.md:
* **Duplicate scans per week** were resolved by filesystem order. Fixed in
  `lumiere_weeks.py` (section 1).
* **Atlas-space misalignment** between visits. Fixed by `register_step2_pairs.py`
  (section 2b).
* **LUMIERE's HD-GLIO native-space masks are mis-oriented** for many scans. Worked around by
  image registration (section 9a).

## Repository map

| Step | Script | Output |
| :--- | :--- | :--- |
| cohort rules (no pre-op scans, no forecasting across surgery) | `scripts/lumiere_cohort.py`, `scripts/lumiere_weeks.py` | - |
| original intensity model, held-out retest + blur control | `scripts/intensity_forecast_cv.py` (model: `scripts/run_neural_ode_pipeline.py`) | `results/intensity_forecast/` |
| step 1: volume trajectory | `scripts/volume_trajectory_experiment.py` | `results/volume_trajectory/` |
| step 2: 3D mask forecast (atlas or registered frame) | `scripts/build_atlas_cache.py`, `scripts/register_step2_pairs.py`, `scripts/mask_forecast_experiment.py` | `results/mask_forecast{,_registered}/` |
| alignment check | `scripts/alignment_check.py` | `results/alignment/` |
| inter-segmenter agreement + stratified step 2 | `scripts/step2_followup_analysis.py` | `results/step2_followups/` |
| step 3: next-visit RANO rating | `scripts/rano_forecast_experiment.py` | `results/rano_forecast/` |
| volume forecast ranges (evaluation + calibration) | `scripts/volume_intervals.py` | `results/volume_intervals/`, `runs/packaging/*/volume-intervals.json` |
| approach screen: which questions the data can answer | `scripts/approach_screen.py` | `results/approach_screen/` |
| progression call: evaluation, then released model | `scripts/progression_call_experiment.py`, `scripts/progression_model.py` | `results/progression_call/`, `runs/packaging/*/progression-gbm` |
| progression call vs survival (expert vs automated) | `scripts/progression_survival.py` | `results/progression_survival/` |
| progression call: segmenter swap, dependence on the expert | `scripts/progression_robustness.py` | `results/progression_robustness/` |
| progression call: where it disagrees with the expert | `scripts/progression_errors.py` | `results/progression_errors/` |
| volume rule vs survival: segmenter, thresholds, surrogacy | `scripts/volume_rule_robustness.py` | `results/volume_rule_robustness/` |
| progression call: learning curve | `scripts/progression_learning_curve.py` | `results/progression_learning_curve/` |
| external test on MU-Glioma-Post (plan, freeze, run, exploratory) | `scripts/mu_freeze_model.py`, `scripts/mu_validation.py`, `scripts/mu_validation_explore.py` | `results/mu_validation/`, `runs/mu_validation/` |
| CLI (packaged models) | `scripts/predict.py`, `scripts/prediction.py`, `scripts/package_models.py` | user-chosen output directory |
| figures | `scripts/plot_step_results.py`, `scripts/visualize_mask_forecasts.py` | `results/figures/` |

## Reproducing

Data: see [DATA_SETUP.md](DATA_SETUP.md). LUMIERE is not redistributed here.

Long jobs run in tmux with logs in `runs/logs/` and a `*.DONE` sentinel. All are resumable,
and registrations are cached and never repeated. The exact commands are in
EXPERIMENTS.md section 6. In short:

```bash
.venv/bin/python scripts/volume_trajectory_experiment.py           # step 1, CPU
.venv/bin/python scripts/build_atlas_cache.py                      # step 2 input cache, once (CPU)
.venv/bin/python scripts/register_step2_pairs.py                   # step 2 registration, once (~1 h CPU)
tmux new -d -s gbm_followups 'bash scripts/run_followup_experiments.sh'   # intensity retest + step 2 registered (GPU, ~6 h)
.venv/bin/python scripts/step2_followup_analysis.py ceiling        # ~1 h CPU
.venv/bin/python scripts/step2_followup_analysis.py report
.venv/bin/python scripts/rano_forecast_experiment.py               # step 3, ~1 min CPU
.venv/bin/python scripts/volume_intervals.py evaluate              # volume ranges, held-out check, ~25 s CPU
.venv/bin/python scripts/approach_screen.py                        # alternative tasks screen, ~6 min CPU
.venv/bin/python scripts/progression_call_experiment.py            # progression call, ~1 min CPU (needs the step 2 caches)
tmux new -d -s gbm_survival '.venv/bin/python scripts/progression_survival.py > runs/logs/progression_survival.log 2>&1'   # ~3 min
tmux new -d -s prog_robust '.venv/bin/python scripts/progression_robustness.py > runs/logs/progression_robustness.log 2>&1'   # a few min
.venv/bin/python scripts/progression_errors.py                     # error analysis, < 1 min (after progression_robustness.py)
tmux new -d -s vol_rule '.venv/bin/python scripts/volume_rule_robustness.py > runs/logs/volume_rule_robustness.log 2>&1'   # ~2 min (after progression_robustness.py)
tmux new -d -s prog_lc '.venv/bin/python scripts/progression_learning_curve.py > runs/logs/progression_learning_curve.log 2>&1'   # ~6 min
# external test: the freeze already exists and refuses to overwrite; mu_validation.py reads results/mu_validation/mask_volumes.csv
tmux new -d -s mu_val '.venv/bin/python scripts/mu_validation.py > runs/logs/mu_validation.log 2>&1'   # ~1 min
.venv/bin/python scripts/mu_validation_explore.py                  # exploratory follow-up, < 1 min
.venv/bin/python scripts/plot_step_results.py
```

Tests:

```bash
python3 -m py_compile scripts/*.py
python3 -m unittest discover -s tests
```

## Legacy pipeline

`scripts/run_neural_ode_pipeline.py` is the original 2D Attention U-Net + Neural ODE
intensity forecaster (per-patient runs, `results/lumiere_full_v1_metrics.*`). It still
runs, and the smoke test below exercises it on synthetic data, but its per-patient
evaluation scores training pairs. Use `scripts/intensity_forecast_cv.py` for held-out
numbers.

```bash
python3 scripts/generate_dummy_lumiere.py --output-dir test_data --num-weeks 3
python3 scripts/run_neural_ode_pipeline.py --lumiere --data-dir test_data --patients Patient-999 \
  --epochs 1 --batch-size 1 --model-size tiny --run-name smoke_default_holdout_3wk
```

## Status and limitations

* One dataset (LUMIERE, single centre), 79 patients. There is no external validation.
* All tumor labels are automated (DeepBraTumIA). Their reliability is the main limit on any
  mask-level claim (section 9a).
* Results are complete and frozen for the analyses above. `PUBLICATION_PROTOCOL.md`
  describes the original intensity-forecasting protocol and is superseded by
  EXPERIMENTS.md for what is reported here.
