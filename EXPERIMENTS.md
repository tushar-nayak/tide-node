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
* Step 2 (3D masks): no. LUMIERE's DeepBraTumIA `atlas/` outputs are already resampled into
  one shared 1 mm atlas space per scan, so scans of a patient are aligned without this
  step, and the segmentation masks live in the same space. (The affine cache above
  registers intensity volumes only, not masks, so it could not feed a mask model directly.)

**Known limitation.** Each modality is registered independently to the target CT1 instead of
estimating one transform per scan and applying it to all four modalities; the modalities of
one historical scan can therefore end up very slightly misaligned with each other.

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
* `step2_mask_dice.png`: Dice of model vs persistence per region (regenerated by
  `scripts/plot_step_results.py` once all folds finish).
