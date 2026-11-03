# Reframed experiments (November 2026)

The original experiment (predict raw future MRI intensities at the volume's centre slices,
one model per patient) did not produce valid evidence: every row of
`results/lumiere_full_v1_metrics.csv` is `metric_split=all`, i.e. scored on the training
pairs. This document describes the bug fix and the two replacement experiments.

## 1. Bug fix: duplicate scans per week

LUMIERE stores several scans under one week number for 46 of 91 patients
(`week-000-1` = pre-op, `week-000-2` = post-op, confirmed by the expert RANO table; also
re-resections such as `week-040-1/-2`). Both loaders parsed only the number, so the scan
kept for that week depended on filesystem `glob` order.

* `scripts/lumiere_weeks.py` now resolves each week deterministically (highest suffix =
  latest scan of the week; falls back to an earlier one only if the latest is incomplete).
  Both `run_neural_ode_pipeline.py` and `preprocess_lumiere_registration.py` use it.
* Registration cache entries now carry `registration_source.json` naming the exact source
  scans. Old entries without a marker are trusted only if neither week had duplicates.
  227 of 2,278 cached registrations (38 patients) failed that check and were regenerated;
  nothing else is ever re-registered.

## 2. Registration: what it does and which experiments need it

**What it is.** `scripts/preprocess_lumiere_registration.py` aligns every earlier scan of a
patient to each later "target" scan, so that history and target line up voxel by voxel.
For every target week it registers *all* prior weeks to it (a superset of any history
window), which is why it only ever has to be done once.

**How.** SimpleITK, 3D affine transform, Mattes mutual information (50 bins, 20% random
sampling), regular-step gradient descent (200 iterations), 3-level pyramid (shrink 4/2/1,
smoothing 2/1/0 mm), geometry-centred initialisation. The fixed image is the target's
contrast-enhanced T1 (CT1); each moving modality is min-max normalised and registered to it,
then resampled with linear interpolation.

**Output.** `data/lumiere_registered/<Patient>/target-<www>/week-<mmm>/.../*_skull_strip_registered.nii.gz`
(~50 GB, 2,278 (target, moving) pairs for 81 patients), plus `registration_manifest.json`
listing every job and a `registration_source.json` marker per pair naming the exact source
scans (see section 1). The setup script `scripts/setup_lumiere_data.sh` runs it in 8 parallel
batches; a fresh full run takes ~1-1.5 days of CPU (GPU does not help).

**Never re-run unnecessarily.** A pair is skipped if all four outputs exist *and* its marker
matches the scans the resolver currently selects. Only a changed source scan (new LUMIERE
release, or the duplicate-week fix) triggers re-registration of that pair alone. To move to
another machine, copy `data/lumiere_registered/` instead of re-running (see `DATA_SETUP.md`).

**Which experiments use it.**
* The original slice pipeline (`run_neural_ode_pipeline.py --registered-data-dir ...`): yes.
* Step 1 (volumes): no; it only reads the per-scan volume JSONs.
* Step 2 (3D masks), original run: no; it assumed LUMIERE's per-scan atlas space already
  aligns the scans of one patient. Section 2b shows that assumption is wrong.
* Step 2, registered frame: uses its own one-off registration (`register_step2_pairs.py`,
  section 2b), because this cache holds no masks and no transforms.
* Section 8 (original intensity model on held-out patients): yes.

**Known limitation.** Each modality is registered independently to the target CT1 instead of
estimating one transform per scan and applying it to all four modalities; the modalities of
one historical scan can therefore end up very slightly misaligned with each other.

## 2b. Alignment check: atlas space is not voxel-aligned between visits

`scripts/alignment_check.py` (CPU, ~7 min) compares, for all 399 cohort transitions, the last
scan as LUMIERE ships it in atlas space vs the same scan from the affine cache above, both
against the target scan. It scores only healthy tissue: the target brain minus both tumor masks
dilated by 5 mm. Output: `results/alignment/`, figure `results/figures/alignment_atlas_vs_affine.png`.

| metric (healthy tissue) | atlas (original step 2 input) | after affine registration | patient diff [95% CI] |
| :--- | ---: | ---: | :---: |
| T1 correlation, median | 0.566 | 0.848 | +0.245 [+0.220, +0.271] |
| FLAIR correlation, median | 0.498 | 0.792 | +0.262 [+0.233, +0.288] |
| T2 correlation, median | 0.466 | 0.830 | +0.344 [+0.316, +0.372] |
| brain-centroid shift, median | 2.5 mm | 0.6 mm | -2.8 mm [-3.2, -2.5] |

Registration improved T1 correlation in 100% of transitions. Atlas alignment is worst for
post-operative and early scans: Patient-001's week-000 is tilted ~1 cm against week-044.
Worse atlas alignment goes with lower persistence Dice in step 2 (Spearman +0.46). So part of
the "real change" the original step 2 scored was misregistration that no model can forecast.
(The brain-mask Dice row in `results/alignment/summary.md` understates the gain: the old
cache's linear resampling smears the brain edge.)

**Fix: `scripts/register_step2_pairs.py`** (CPU, run once, ~1 h, skips finished pairs). For
every transition it registers the target scan, and the second-to-last scan, into the frame of
the **last** scan. That is the scan the forecast is made from, so the inputs carry no target
information. Each pair gets one transform: rigid, then affine, on T1, with both tumors masked
out, pyramid 4 mm -> 2 mm. That transform is applied to all four modalities and to the
segmentation (nearest neighbour). If a result overlaps the brain worse than the untouched
atlas alignment, it falls back to rigid-only, then to identity; `qc.method` records which.
Outputs: `data/lumiere_step2_aligned/<Patient>/<moving>__in__<fixed>.{npy,h5,json}` (2 mm
cache in the `build_atlas_cache.py` format, the transform, and QC), 691 pairs. Step 2 then
runs with `--frame registered` (outputs in `runs/mask_forecast_registered/` and
`results/mask_forecast_registered/`).

Result over the 691 pairs: 534 rigid+affine, 132 rigid, 24 needed the unmasked retry, and 1
kept the atlas alignment (it was already well aligned). Healthy-tissue T1 correlation went
from a median 0.57 to 0.85 (10th percentile 0.33 -> 0.62). Brain-mask Dice, resampled
nearest-neighbour, went from 0.961 to 0.981 (10th percentile 0.921 -> 0.965), and no pair got
worse by more than 0.02.

## 3. Cohort rules (shared by both steps) — `scripts/lumiere_cohort.py`

* One scan per week (above). Scans rated `Pre-Op` are dropped.
* A `Post-Op` scan (including re-resections) starts a new segment; history never crosses
  a segment, and an interval that ends in surgery is never a forecast target.
* Result: 79 patients, 399 forecast transitions (39 pre-op scans and 31 surgery intervals excluded).
* Evaluation is always **patient-level** cross-validation: a test patient is never seen in
  training. Clinical tables (RANO ratings, demographics) are in `data/lumiere/clinical/`
  (downloaded from the LUMIERE Figshare collection; access date in `ACCESS_DATE.txt`).

## 4. Step 1 — tumor volume trajectory forecasting

`scripts/volume_trajectory_experiment.py` (CPU, ~1 h). Predicts log1p(mL) of necrosis,
enhancing tumor and edema at the next visit from all earlier visits of the segment, the
interval length, and age / sex / MGMT. 5-fold patient CV × 3 fold seeds.

Methods: persistence, log-linear extrapolation, population Gompertz, ridge, gradient
boosting, and an ODE-RNN Neural ODE (5-seed ensemble, residual decoder that starts at
persistence, early stopping on held-out training patients).

Outputs: `results/volume_trajectory/{metrics.md,metrics.json,predictions.csv}`.

## 5. Step 2 — 3D next-visit tumor mask forecasting

* `scripts/build_atlas_cache.py` (run once, ~20 min, 6.2 GB): LUMIERE's DeepBraTumIA
  `atlas/` outputs are already in one common 1 mm atlas space, so scans are voxel-aligned
  without further registration. The cache stores every cohort scan at 2 mm: 4 z-scored
  modalities + whole-tumor / tumor-core / enhancing fractions. Existing files are skipped.
* `scripts/mask_forecast_experiment.py`: 3D U-Net whose bottleneck is evolved by a Neural
  ODE over the forecast interval; outputs a displacement field that warps the last mask plus
  residual logits. Both heads start at zero, so the untrained model *is* persistence.
  Ablation `static`: same network with dt injected by an embedding instead of an ODE.
  Same patient folds as step 1. Metrics: Dice (WT/TC/ET), change-Dice (does the predicted
  change match the real change?), volume error; all against persistence with patient-level
  bootstrap CIs.

Outputs: `results/mask_forecast/{metrics.md,metrics.json}`, per-fold predictions and
checkpoints in `runs/mask_forecast/<variant>/fold<k>/`.

## 6. Running long jobs

Follow-up queue (section 8, then step 2 in the registered frame):
`tmux new -d -s gbm_followups 'bash scripts/run_followup_experiments.sh'`.
Noise ceiling (section 9):
`tmux new -d -s gbm_step2_followups '.venv/bin/python -u scripts/step2_followup_analysis.py ceiling > runs/logs/step2_ceiling.log 2>&1 && touch runs/logs/step2_ceiling.DONE'`,
then `.venv/bin/python scripts/step2_followup_analysis.py report`.

Every long job runs in tmux with logs under `runs/logs/` (gitignored) and a `*.DONE`
sentinel when finished. All of them are resumable.

```bash
tmux new -d -s gbm_step1 '.venv/bin/python -u scripts/volume_trajectory_experiment.py > runs/logs/step1_volume_trajectory.log 2>&1'
tmux new -d -s gbm_step2 'bash scripts/run_mask_forecast_cv.sh'   # all folds x variants, skips finished folds
.venv/bin/python scripts/mask_forecast_experiment.py summarize    # aggregate whatever folds are done
.venv/bin/python scripts/plot_step_results.py                     # results/figures/*.png
tmux ls; tail -f runs/logs/mask_forecast_ode_fold0.log             # check progress
```

Environment: `.venv/` (Python 3.14, torch 2.13 + CUDA 12.9, requirements.txt + scikit-learn).

## 7. Results

### Step 1 (final: 5-fold patient CV x 3 repeats, 79 patients, 399 transitions)

Full table: `results/volume_trajectory/metrics.md`; figure: `results/figures/step1_volume_vs_persistence.png`.

| Method | MAE log1p(mL) | vs persistence | patient-level diff [95% CI] | AUROC for next-scan RANO PD |
| :--- | ---: | ---: | :---: | ---: |
| persistence | 0.745 | - | - | 0.500 |
| log-linear extrapolation | 0.993 | -33.4% | +0.219 [+0.164, +0.277] | 0.541 |
| Gompertz (population) | 0.719 | +3.4% | -0.076 [-0.114, -0.038] | 0.584 |
| ridge | 0.721 | +3.2% | -0.088 [-0.136, -0.040] | 0.653 |
| gradient boosting | 0.693 | +6.9% | -0.085 [-0.119, -0.051] | 0.635 |
| Neural ODE (ODE-RNN) | 0.708 | +4.9% | -0.081 [-0.122, -0.044] | 0.628 |

Reference: using the *true* next-scan volume change gives AUROC 0.713 for RANO PD, the
ceiling for any volume-change score.

Reading:
* Every learned model beats persistence, and the patient-level CIs exclude zero. Naive
  per-patient exponential extrapolation is much worse than persistence: volumes between
  visits are dominated by treatment effects, not smooth growth.
* The Neural ODE is **not** better than simple tabular models: it is statistically tied
  with ridge / gradient boosting / Gompertz (overlapping CIs), and gradient boosting has the
  lowest raw MAE in all three repeats. The defensible claim is "learned dynamics beat
  persistence by ~5-7%", not "Neural ODEs beat other learners".
* Predicted volume change carries real progression signal (AUROC ~0.63-0.65 against an
  oracle ceiling of 0.71), without seeing the target scan.

### Step 2 (final: 5-fold patient CV, 79 patients, 399 transitions)

Full table: `results/mask_forecast/metrics.md`; figure: `results/figures/step2_mask_dice.png`.

| Variant | WT Dice (model / persistence) | TC | ET | patient Dice diff [95% CI] | change-Dice diff [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Neural ODE | 0.521 / 0.506 | 0.355 / 0.350 | 0.231 / 0.234 | +0.010 [+0.000, +0.019] | +0.127 [+0.107, +0.148] |
| no-ODE (dt embedding) | 0.521 / 0.506 | 0.360 / 0.350 | 0.239 / 0.234 | +0.011 [+0.005, +0.017] | +0.118 [+0.101, +0.135] |

Reading:
* Overlap gain over persistence is real but tiny (~+0.01 Dice). The models do predict *some*
  of the real change (change-Dice +0.12, 63/79 patients), but typically only 5-15% of the
  changed voxels; large expansions, shrinkage after treatment and new distant lesions are
  missed. Volume error is not improved.
* The Neural ODE bottleneck adds nothing over a plain dt embedding (tied everywhere).
* In fold 0 neither variant beat persistence on validation patients, so its early-stopped
  model is persistence; folds 1-4 improved validation Dice by 0.01-0.02.
* Caveat: some "real change" may be misalignment between atlas-space scans (e.g. Patient-031
  week-184 vs week-190 in `step2_forecast_ode_fold3_final.png` shows visibly different
  anatomy at the same slice). This inflates the change no model can predict.
  Tested directly below.

### Step 2 in the registered frame (final: 5-fold patient CV, 79 patients, 399 transitions)

Same folds, models, training and metrics as above. The only change is
`--frame registered`: the target mask and history are resampled into the last scan's frame
with the section 2b transforms, so persistence is no longer penalised by misalignment.
Full table: `results/mask_forecast_registered/metrics.md`.

| Variant | WT Dice (model / persistence) | TC | ET | patient Dice diff [95% CI] | change-Dice diff [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Neural ODE | 0.573 / 0.570 | 0.408 / 0.406 | 0.266 / 0.278 | +0.003 [-0.003, +0.010] | +0.071 [+0.053, +0.092] |
| no-ODE (dt embedding) | 0.574 / 0.570 | 0.408 / 0.406 | 0.277 / 0.278 | +0.006 [+0.001, +0.012] | +0.076 [+0.058, +0.093] |

Reading:
* Alignment alone does more than either model. Persistence WT Dice rises from 0.506 to 0.570
  (TC 0.350 to 0.406, ET 0.234 to 0.278). About 0.06 Dice of the atlas-frame "change" was
  misalignment.
* The model's margin over persistence shrinks with it. Dice gain falls from +0.010 to
  +0.003 (ODE, CI includes 0) and +0.006 (no-ODE). Change-Dice gain falls from about +0.12 to
  +0.07. Roughly 40-45% of the change the atlas-frame models "caught" was misalignment, not
  tumor change. Enhancing-tumor Dice is now slightly *below* persistence for the ODE variant.
* Volume error is still not improved (logvol_err diff -0.008 [-0.052, +0.030], ODE).
* Early stopping kept persistence (epoch -1) in folds 0 and 4 for the ODE, and in fold 0
  for no-ODE. Every fold's last-epoch weights score below the best on validation patients,
  so longer training would not help.
* The Neural ODE still adds nothing over a dt embedding.
* Bottom line for step 2: once scans are properly aligned, neither model forecasts
  next-visit tumor masks meaningfully better than copying the last mask.

## 8. The original claim re-tested on held-out patients — `scripts/intensity_forecast_cv.py`

The original +6.7% was the per-patient mean of (persistence MSE - model MSE) / persistence
MSE, scored on each patient's own training pairs. This re-test keeps the original model
(`run_neural_ode_pipeline.py`'s attention U-Net + Neural ODE, zero-initialised warp/residual
heads, original loss), inputs (5 centre axial slices of all 4 modalities, history registered
to the target via `data/lumiere_registered/`) and metric. It changes only the evaluation:
* same cohort as steps 1/2 (no pre-op scans, no surgery targets, last <= 4 scans of the segment);
* patient-level 5-fold CV, same folds as steps 1/2, so a test patient is never trained on;
* checkpoint chosen on held-out training patients; 40 epochs.

`build-cache` extracts the slice stacks once (`data/lumiere_slices_registered/`, 2 GB); each
fold takes ~25 min on the GPU. Output: `results/intensity_forecast/metrics.{md,json}`.

### Result (final: 5 folds, 79 held-out patients, 399 transitions)

| metric | model | persistence | rel. improvement [95% CI] | patients better |
| :--- | ---: | ---: | :---: | ---: |
| whole-slice MSE (original metric) | 0.00559 | 0.00633 | +8.7% [+5.7%, +11.7%] | 64/79 |
| tumor-region (focal) MSE | 0.01279 | 0.01347 | +2.5% [-1.0%, +6.0%] | 50/79 |
| whole-slice MAE | 0.0362 | 0.0389 | +5.4% [+3.6%, +7.2%] | 64/79 |

**Control: blurred persistence** (`intensity_forecast_cv.py blur-baseline`, CPU, a few minutes).
This is persistence smoothed in-plane with a Gaussian; the width is tuned on the training
patients of each fold, and sigma = 1 px was chosen in every fold. Under uncertainty, an
MSE-trained forecaster can gain just by blurring. Output: `results/intensity_forecast/blur_baseline.md`.

| metric | network | blurred persistence | network minus blur [95% CI] |
| :--- | :---: | :---: | :---: |
| whole-slice MSE | +8.7% | +3.7% | +5.0% [+2.1%, +8.0%] |
| tumor-region MSE | +2.5% | +4.1% | -1.6% [-5.1%, +2.0%] |
| whole-slice MAE | +5.4% | -1.2% | +6.7% [+4.8%, +8.5%] |

Reading:
* The original number is roughly reproduced on unseen patients: +8.7% vs the claimed +6.7%.
  It is therefore not an artefact of training-set scoring.
* It is not evidence of tumor forecasting. Inside the tumor region the network is no better
  than persistence (CI includes 0), and a 1-pixel blur of the last scan does at least as well
  there. About 40% of the whole-slice gain is plain smoothing. The rest is outside the tumor:
  likely correction of global intensity and normalisation differences between scans, which
  the MAE gain over blur suggests. Per-modality, T1 even gets worse (-9.9%, per-patient mean).
* The defensible statement is: "the intensity model reduces whole-slice error vs persistence
  on held-out patients, but shows no measurable skill where the tumor is."

## 9. Step 2 follow-ups: noise ceiling and where the model helps — `scripts/step2_followup_analysis.py`

No training. `ceiling` (CPU, tmux `gbm_step2_followups`, about 1 h on 11 cores) maps HD-GLIO's mask of
every step-2 target scan into the atlas; `report` (a few seconds) writes
`results/step2_followups/{summary.md,summary.json,ceiling.csv}` and
`results/figures/step2_strata_registered.png`.

### 9a. Noise ceiling

LUMIERE has no test-retest scans: its same-week duplicates are pre-op/post-op pairs with
surgery in between. Two proxies instead:

**(a) Same scan, two segmenters.** Every step-2 target scan's DeepBraTumIA atlas mask (the
step-2 label) is compared with LUMIERE's HD-GLIO-AUTO mask of the *same* scan, scored on the
step-2 grid (2 mm, > 0.5).

LUMIERE's HD-GLIO *native-space* files are the dataset authors' own back-transform, and for
many scans they are mis-oriented: the mask lands outside the brain or on the wrong side.
They are not used. Instead, HD-GLIO's `registered/` mask, which shares a grid with
HD-GLIO's own CT1, is mapped by registering that CT1 to DeepBraTumIA's atlas CT1 (rigid
from 4 starting orientations, best Mattes MI, then affine). Tumor masks are never used, so
this cannot inflate agreement. QC: brain-mask Dice median 0.978, and 3/399 scans fall below
0.9; excluding them changes nothing.

| comparison (399 target scans, 79 patients) | median Dice [IQR] | patient mean [95% CI] |
| :--- | :---: | :---: |
| resampling floor: DeepBraTumIA's own native mask pushed to atlas | 0.995 [0.992, 0.998] | 0.985 |
| DeepBraTumIA vs HD-GLIO, whole tumor | 0.723 [0.577, 0.819] | 0.648 [0.612, 0.680] |
| same, DeepBraTumIA without necrosis/cavity (HD-GLIO has no such label) | 0.764 [0.651, 0.832] | 0.689 [0.655, 0.721] |
| DeepBraTumIA vs HD-GLIO, enhancing (329 scans where either finds ET) | 0.632 [0.135, 0.800] | 0.509 [0.463, 0.557] |

**(b) Visits where nothing clinically changed.** Copy-last-mask in the registered frame:

| subset | transitions | WT median [IQR] | WT mean | ET mean |
| :--- | ---: | :---: | ---: | ---: |
| expert-rated SD, interval <= 12 wk | 32 | 0.601 [0.394, 0.821] | 0.581 | 0.384 |
| expert-rated SD, any interval | 92 | 0.524 [0.299, 0.783] | 0.509 | 0.314 |
| all step-2 transitions | 399 | 0.616 [0.394, 0.789] | 0.570 | 0.278 |

Reading:
* Two automated segmenters looking at the *same image* agree only to WT Dice 0.65 (patient
  mean). Copying the last visit's mask already reaches 0.57. Across visits, the target is
  almost as uncertain as within one scan.
* Visits a neuroradiologist rated as stable, at most 12 weeks apart, still only reach a mean
  WT Dice of 0.58. That is no better than copy-last-mask over all transitions. Rescan,
  segmentation and registration noise, plus change too small to cross a RANO threshold,
  costs about 0.4 Dice at a visit with no clinically meaningful change.
* Implication: for whole tumor, the attainable headroom over copy-last-mask is small.
  Roughly 0.08 Dice separates it from same-scan segmenter agreement, and a perfect forecast
  can't be scored much above the labels' own reliability. The step-2 models' +0.003 to
  +0.006 uses a small fraction of that, and section 9b shows the fraction they do use is a
  growth prior, not tumor-specific forecasting. Enhancing tumor has more nominal headroom
  (0.28 vs 0.51) but is the noisiest label.
* Same-segmenter test-retest would be higher than DeepBraTumIA-vs-HD-GLIO agreement, which
  also contains definitional differences. So (a) is a pessimistic estimate of label
  reliability, and (b) is the more direct one.

### 9b. Step 2 split by how the tumor changed (registered frame)

Strata are defined from the registered frame and applied to all four runs. The full
tables, including the atlas frame, are in `results/step2_followups/summary.md`.

| stratum | transitions | copy-last-mask WT Dice | Dice diff, ODE [95% CI] | Dice diff, no-ODE [95% CI] | forecast / true log-volume change (ODE) |
| :--- | ---: | ---: | :---: | :---: | :---: |
| RANO PD (progression) | 236 | 0.591 | +0.007 [+0.001, +0.015] | +0.010 [+0.003, +0.018] | +0.08 / +0.26 |
| RANO SD (stable) | 92 | 0.509 | -0.003 [-0.016, +0.008] | +0.001 [-0.009, +0.011] | +0.04 / -0.62 |
| RANO PR (partial response) | 20 | 0.625 | -0.009 [-0.022, +0.002] | -0.010 [-0.025, +0.001] | +0.05 / -0.23 |
| RANO CR (complete response) | 26 | 0.568 | -0.034 [-0.085, -0.001] | -0.030 [-0.076, -0.001] | +0.18 / -0.19 |
| volume shrank > 20% | 109 | 0.424 | +0.000 [-0.008, +0.007] | +0.001 [-0.008, +0.009] | +0.01 / -1.11 |
| volume within -20% / +25% | 144 | 0.767 | -0.014 [-0.033, +0.002] | -0.015 [-0.034, -0.000] | +0.07 / -0.00 |
| volume grew > 25% | 146 | 0.485 | +0.009 [-0.001, +0.019] | +0.019 [+0.010, +0.029] | +0.12 / +0.81 |
| low overlap with last mask (< 0.47) | 133 | 0.265 | +0.003 [-0.005, +0.011] | +0.008 [+0.001, +0.016] | +0.09 / -0.15 |
| high overlap with last mask (>= 0.74) | 133 | 0.839 | -0.014 [-0.028, -0.003] | -0.012 [-0.023, -0.002] | +0.07 / +0.03 |

Dice diff = model minus copy-last-mask, averaged over WT/TC/ET, as in the main table.
Log-volume change = log((V+1)/(V_last+1)), whole tumor.

Reading:
* The model does not know which way a tumor is going. Its forecast grows the tumor in 78-82%
  of the transitions where it moves at all (truth: 53%). This includes tumors that later
  shrank by more than 20% (forecast +0.01 vs truth -1.11) and complete responses (+0.18
  vs -0.19). The skill is a population prior, "tumors tend to grow a bit," applied
  everywhere.
* That prior is the whole effect. It helps a little where the tumor grew (+0.009 / +0.019)
  and under progression, and costs Dice where nothing changed (high overlap -0.014,
  stable volume -0.014) and under complete response (-0.03).
* The model's apparent sense of direction (Spearman +0.38 to +0.47 between forecast and true
  volume change) comes from "small tumors grow, big ones shrink." The last volume alone
  gives +0.47 to +0.50, and once it is removed the forecast's correlation with the truth is
  +0.05 to +0.11.
* change-Dice is above copy-last-mask in every stratum (+0.03 to +0.10). Persistence scores
  ~0 on change-Dice by construction, so any change at all earns some. It is not a sign of
  skill on its own.
* The atlas frame shows the same pattern with larger numbers: grew > 25% +0.043, stable
  -0.014. Its forecasts grow the tumor in 91-96% of transitions.

## 10. Step 3 — forecasting the next visit's RANO rating — `scripts/rano_forecast_experiment.py`

Question: from what is known at the last scan, will the neuroradiologist rate the next scan
as progressive disease (PD)? The reference is the expert rating, not an automated mask, so
step 2's label-noise ceiling does not apply.

* Same cohort and folds as step 1: 5-fold patient CV × 3 fold seeds. Each transition's
  out-of-fold score is averaged over the seeds.
* Data: 374 expert-rated transitions from 79 patients, 63% PD.
* Features use only information available at the last scan:
  * volume history, including height above the segment's nadir (RANO measures progression
    against the nadir), and interval;
  * the last visit's expert rating;
  * age, sex and MGMT.
  * Survival is never used.
* Runtime: about 40 s on CPU. Output: `results/rano_forecast/` and
  `results/figures/step3_rano_auroc.png`.

| score | AUROC [95% CI] | minus last-rating rule [95% CI] |
| :--- | :---: | :---: |
| trivial: small tumor grows (-last volume) | 0.565 [0.491, 0.642] | -0.118 [-0.207, -0.026] |
| trivial: interval length | 0.428 [0.363, 0.500] | -0.256 [-0.341, -0.151] |
| **trivial: last visit's rating** | **0.684 [0.621, 0.740]** | - |
| logistic, volumes + timing + clinical | 0.593 [0.530, 0.657] | -0.090 [-0.158, -0.020] |
| logistic, all features incl. last rating | 0.686 [0.620, 0.747] | +0.002 [-0.042, +0.053] |
| gradient boosting, all features | 0.723 [0.660, 0.781] | +0.040 [-0.015, +0.092] |
| step 1 GBM, predicted volume change | 0.641 [0.569, 0.714] | -0.042 [-0.121, +0.041] |
| step 1 Neural ODE, predicted volume change | 0.633 [0.563, 0.708] | -0.051 [-0.133, +0.037] |

Brier score:

| predictor | Brier |
| :--- | ---: |
| always predict the PD rate | 0.233 |
| last rating | 0.201 |
| gradient boosting | 0.204 |
| logistic, all features | 0.212 |

PD rate by the last visit's rating: PD 87% (n=136), SD 53%, PR 53%, CR 32%, Post-Op 45%.

Within the last visit's rating, among the 238 transitions where the last scan was *not* PD
(50% then progress):

| score | AUROC [95% CI] |
| :--- | :---: |
| small-tumor rule | 0.576 [0.494, 0.652] |
| gradient boosting | 0.599 [0.527, 0.675] |
| step 1 Neural ODE | 0.605 [0.530, 0.683] |
| logistic, all features | 0.556 [0.479, 0.637] |

Reading:
* The strongest predictor of next-visit progression is the current rating: once a patient
  is PD, the next visit is PD 87% of the time. It is the RANO analogue of copy-last-mask.
* No learned model beats it significantly. Gradient boosting is +0.04 AUROC (CI includes 0)
  and no better calibrated (Brier 0.204 vs 0.201). Step 1's volume forecasts, Neural ODE
  included, score below the last-rating rule.
* The clinically useful case is predicting *new* progression in a patient who is currently
  not progressing. There, every score is weak (AUROC ≤ 0.61, lower CI bounds ≈ 0.48-0.53)
  and none is distinguishable from the small-tumor rule.
* Secondary: response (PR/CR, 12%) is predicted mostly by the last rating (AUROC 0.78),
  and learned models add nothing.
* Taken together with sections 7-9: on LUMIERE, carrying the last observation forward is
  close to the best available next-visit forecast. This holds for intensity in the tumor
  region, for the 3D mask, and for the RANO rating. The one significant gain is step 1's
  volume error (5-7% below persistence), and much of that is mean reversion: a population
  Gompertz fit alone gets 3.4%. The Neural ODE never beats simple tabular models.

### Figures (`results/figures/`)

* `step1_volume_vs_persistence.png`: error of each method relative to persistence, 95% CI.
* `step1_patient_trajectories.png`: real total tumor volume over time for six held-out
  patients, with each next-visit forecast drawn from the last real scan.
* `step2_forecast_<variant>_fold<k>_<weights>.png`: the visual forecast. Per held-out
  transition: last scan with its tumor outline, the real next scan, the predicted outline
  over the real one, and a change map (change predicted correctly / missed / predicted but
  not real). Regenerate with `scripts/visualize_mask_forecasts.py --variant ode --fold k --weights final`.
  `best` = early-stopped weights used for the metrics (identical to persistence when
  training never beat it on validation patients); `final` = last-epoch weights (saved from
  fold 1 onward), which show what the network actually learned.
* `step2_forecast_registered_ode_fold<k>_final.png`: the same visual forecast in the
  registered frame (`visualize_mask_forecasts.py --frame registered ...`). The last scan,
  real next scan and outlines all share the last scan's frame.
* `step2_mask_dice.png`: Dice of model vs persistence per region. Top row is the atlas frame,
  bottom row the registered frame (regenerated by `scripts/plot_step_results.py`).
* `step2_strata_registered.png`: registered-frame step 2 gain over copy-last-mask (Dice and
  change-Dice, 95% CI) by RANO rating, volume change, overlap with the last mask, and interval
  (section 9b).
* `step3_rano_auroc.png`: AUROC for next-visit RANO progression, trivial rules vs learned
  models vs step 1 volume forecasts (section 10).
* `alignment_atlas_vs_affine.png`: per-transition healthy-tissue alignment of the last scan
  to the target, atlas space vs affine registration (section 2b).
