# glioblastoma-evolution

History-conditioned Neural ODE experiments for longitudinal glioblastoma MRI forecasting on LUMIERE-style multi-modal scans.

The repository currently contains the runnable training pipeline, registration preprocessing, synthetic smoke-data generation, metric aggregation, and derived result tables from the local `lumiere_full_v1_*` run summaries.

## Model & Evaluation Update (in progress)

Two changes are landing on top of the `lumiere_full_v1` architecture used for the results below. Both are implemented and unit-tested, but not yet re-run across the full cohort, so the "Current Evidence" numbers below still describe the *previous* architecture:

- **Residual/delta prediction.** The Attention U-Net + Neural ODE model previously reconstructed the entire future slice stack from a latent code. It now predicts a delta added to the persistence baseline (latest known scan), with the final decoder layer zero-initialized so the model starts out exactly at persistence and only has to learn what changes. Whole-slice MSE is dominated by unchanged background/normal brain, which is why the persistence baseline was so hard to beat under the old reconstruct-from-scratch formulation.
- **Ground-truth tumor segmentation Dice.** LUMIERE ships real DeepBraTumIA tumor masks (`DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz`, atlas-registered and voxel-aligned with the intensity volumes) alongside every scan. Evaluation now reports whole-tumor Dice against these real masks (`wt_dice`), plus a `focal_mse`/`focal_mae` pair restricted to the FLAIR tumor-hyperintensity region, rather than relying only on whole-slice intensity MSE. The persistence baseline additionally reports real tumor-core and enhancing-tumor Dice (`tc_dice`, `et_dice`), since both sides of that comparison are real masks, unlike the model's thresholded-intensity proxy prediction.

Small real-LUMIERE spot checks (single patients, a handful of longitudinal weeks each) confirm the mechanism works as intended and that the real-mask Dice surfaces gaps whole-slice MSE was hiding. A full cohort re-run with these changes, following [PUBLICATION_PROTOCOL.md](PUBLICATION_PROTOCOL.md), is still needed before the numbers below are superseded.

Those spot checks also caught a real, pre-existing calibration bug: `relative_flair_volume_diff` and the new `wt_dice` proxy both thresholded predicted FLAIR at a fixed absolute `> 0.5` on the per-volume min-max-normalized intensity. Checked against the real ground-truth masks, that threshold turned out to essentially never fire on real LUMIERE volumes (confirmed on multiple patients: even the *persistence baseline's own* thresholded volume came out to ~0 pixels, while its real-mask Dice was clearly nonzero). Fixed by reusing the adaptive, peak-relative threshold (`0.45 × that volume's own max`) already used elsewhere in the codebase for the training loss (`flair_focus_mask`), via a new `adaptive_flair_mask` helper.

Note: because the single-patient spot checks train from scratch on 1-2 samples per patient (no cohort pretraining), they are not a fair test of the method itself and should not be read as "the new model underperforms" — they only validate that the code paths run correctly end-to-end on real data.

A third change has since landed on top of the first two:

- **Deformation-field prediction.** The model now has a second decoder head that predicts a dense 2D displacement field per target slice (shared across all 4 modalities of that slice, since they are co-registered scans of the same anatomy), applied to the persistence baseline via a differentiable warp (`grid_sample`) before the existing appearance-delta head adds any residual on top. Both heads are zero-initialized, so the model still starts out exactly at the persistence baseline (identity warp, zero delta) -- this is a strict superset of the residual-prediction change above, not a replacement, and cannot regress below it. A small smoothness penalty on the flow field discourages physically implausible warps. Rationale: most of a scan doesn't change between visits, and a warp explicitly separates "existing tissue moved" (mass effect) from "genuinely new signal appeared" (the residual), rather than asking one plain reconstruction head to implicitly learn both. Unit-tested (identity-at-init, directionally-correct warping, smoothness penalty behavior) and confirmed working end-to-end on a real training run (loss decreases, no NaNs, flow-smoothness term contributes as intended).

A full 81-patient cohort validation (needed to know whether any of these three changes actually beats the checked-in `+6.7%` checkpoint below) has not been run yet. Registration preprocessing alone is estimated at roughly 1-1.5 days of wall-clock (CPU-bound, ~9,112 SimpleITK registration calls across the cohort; GPU doesn't help this stage). A ready-to-run script for the full validation is at `scripts/run_full_cohort_validation.sh` (see its header comment for usage) -- it preprocesses registration in 8 load-balanced parallel batches, then runs cohort-pretrain + per-patient fine-tuning matching `PUBLICATION_PROTOCOL.md`, then aggregates results to a new `results/lumiere_residual_v1_metrics.*` prefix (existing `lumiere_full_v1_metrics.*` files are left untouched for side-by-side comparison).

## Current Evidence

The checked-in aggregate table is generated by `scripts/summarize_run_metrics.py` from local `run_summary.json` files:

- `results/lumiere_full_v1_metrics.csv`
- `results/lumiere_full_v1_metrics.json`
- `results/lumiere_full_v1_metrics.md`

Current aggregate from 81 patient-level runs:

- Mean model MSE: `0.00828323`
- Mean persistence-baseline MSE: `0.00876673`
- Mean relative improvement: `+6.7%`
- Positive-improvement patients: `65`
- Nonpositive-improvement patients: `16`

These tables use the best available metric split per run, preferring holdout metrics when present and otherwise falling back to the older all-sample summaries. The `metric_split` column makes that choice explicit for each row.

## Pipeline

Main entry points:

- `scripts/run_neural_ode_pipeline.py`: trains and evaluates the history-conditioned Attention U-Net Neural ODE forecaster.
- `scripts/preprocess_lumiere_registration.py`: pre-registers historical LUMIERE volumes to each target week.
- `scripts/generate_dummy_lumiere.py`: creates small synthetic LUMIERE-style data for smoke tests.
- `scripts/summarize_run_metrics.py`: aggregates run summaries into CSV, JSON, and Markdown tables.

## Quick Smoke Test

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_dummy_lumiere.py --output-dir test_data --num-weeks 3
python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir test_data \
  --patients Patient-999 \
  --epochs 1 \
  --batch-size 1 \
  --model-size tiny \
  --run-name smoke_default_holdout_3wk
```

## Full LUMIERE Workflow

```bash
python3 scripts/preprocess_lumiere_registration.py \
  --data-dir path/to/LUMIERE/Imaging \
  --output-dir data/lumiere_registered

python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir path/to/LUMIERE/Imaging \
  --registered-data-dir data/lumiere_registered \
  --separate-patient-runs \
  --epochs 40 \
  --run-name lumiere_full_v1

python3 scripts/summarize_run_metrics.py \
  --runs-dir runs \
  --pattern 'lumiere_full_v1_Patient-*/run_summary.json' \
  --output-dir results \
  --prefix lumiere_full_v1_metrics
```

## Validation

```bash
python3 -m py_compile scripts/*.py
python3 -m unittest discover -s tests
```

GitHub Actions runs the same Python validation workflow on pushes and pull requests.

## Publication Status

This is not yet a finished clinical manuscript package. The code now has reproducible aggregation, tests, CI, and tracked result tables, but a publication submission still needs a locked final run using [PUBLICATION_PROTOCOL.md](PUBLICATION_PROTOCOL.md), independent holdout reporting for all cohort runs, data-access instructions, ethics/data-use language, and manuscript figures generated from the final frozen outputs.
