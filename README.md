# Tumor Interval Dynamics Estimation via Neural ODE Sequences
## (previously glioblastoma-evolution)

Longitudinal glioblastoma MRI analysis on the LUMIERE dataset (79 patients, 399
scan-to-scan transitions, 374 expert-rated follow-up scans). Every experiment is scored on
patients the model never saw, against the trivial rule for that question, with
patient-level bootstrap confidence intervals.

**What works: an automated progression call.** The released model reads the automated tumor
volumes of the current and earlier scans, the previous expert rating, and age, sex and MGMT.
It estimates whether an expert would rate the current scan as progressive disease (PD) under
RANO.
* On held-out patients, AUROC is **0.82** [0.77, 0.86]. Carrying the previous rating forward
  scores 0.68 (difference +0.14 [+0.07, +0.20]).
* For new progression (previous scan not PD), it scores 0.73. An enhancing-volume rule scores
  0.65.
* About half of all scans fall in its "likely PD" band, and 85% of those were rated PD.

It runs on CPU in a few seconds (see Run the models below).

**What does not work: forecasting the next visit.** Once the evaluation is leakage-free,
the scans are aligned and trivial controls are included, carrying the last scan forward is
close to the best next-visit forecast. This holds for intensity, the 3D tumor mask and the
next RANO rating. No Neural ODE model beats the best simple baseline for its target. The
volume forecast gains a modest 5-7%, now with calibrated ranges. Full details, numbers and
commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The draft paper
[MANUSCRIPT.md](MANUSCRIPT.md) covers the forecasting results only.

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
| How good is the progression call? | Released model (volumes + previous rating + clinical): AUROC 0.820 [0.774, 0.862]; 69% of PD scans found at 80% specificity. Adding registered-mask features (new lesions) gives 0.833 (+0.012, CI includes 0). A post-radiotherapy flag in the rating file would add more, but raters set it selectively, so it leaks the label and is excluded. | 12 |

Figures: `results/figures/`, listed at the end of EXPERIMENTS.md.

## Run the models

Both released models take a small JSON file and run on CPU in seconds. The input is
DeepBraTumIA volumes in mm³ for each visit since the last surgery. See
[PREDICTION.md](PREDICTION.md) for the formats and every check.

**Progression call.** The last visit in `history` is the scan to assess, with no rating.
The visit before it needs its expert rating (`PD`, `SD`, `PR`, `CR`, `Post-Op` or
`unknown`):

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
  'likely PD' at >= 0.70 (held-out specificity 80%, sensitivity 69%); 'PD less likely' below 0.42 (sensitivity 90%, specificity 44%); held-out AUROC 0.82
  Enhancing tumor 5.2 mL: +160% vs previous (2.0), +160% vs nadir (2.0); total 35.4 mL (+79% vs previous). Previous rating: SD
```

The three bands, on held-out patients:
* "likely PD": 51% of scans, 85% of them rated PD.
* "uncertain": 26% of scans, 49% rated PD.
* "PD less likely": 23% of scans, 28% rated PD.

Volumes alone cannot see new non-enhancing lesions or subtle T2 progression, so a stable
scan usually lands in "uncertain" rather than "PD less likely".

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
.venv/bin/python scripts/progression_call_experiment.py            # progression call, ~1-2 min CPU (needs the step 2 caches)
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
