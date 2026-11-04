# Standalone 3D mask forecasting

`scripts/predict.py` forecasts WT, TC and ET from historical MRI **and existing historical segmentations**. It does not segment an unlabelled MRI. It uses the static or ODE `MaskForecastNet` from `scripts/mask_forecast_experiment.py`; the older 2D intensity/attention pipeline is not supported. Research use only; this workflow has not been clinically validated.

Run from this checkout with the dependencies in `requirements.txt`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/tide-prediction-mpl \
  /home/tusharn/ppr/glioblastoma-evolution/.venv/bin/python scripts/predict.py \
  --history /path/history.json --horizon-weeks 12 \
  --bundle /path/bundle.json --output-dir /path/new-forecast --threads 4
```

The output directory must not exist. The command runs exclusively on CPU, without training, cache writes or checkpoint updates. No GPU option is provided. Any separate GPU experiment must acquire `flock /tmp/tide-release-gpu.lock` before starting the GPU process.

## History manifest, version 1

Paths are absolute or relative to the manifest. Example for one historical visit:

```json
{
  "version": 1,
  "patient_id": "example",
  "frame": "atlas",
  "history": [
    {
      "week": 0,
      "modalities": {
        "FLAIR": "week000/flair_skull_strip.nii.gz",
        "T1": "week000/t1_skull_strip.nii.gz",
        "T2": "week000/t2_skull_strip.nii.gz",
        "CT1": "week000/ct1_skull_strip.nii.gz"
      },
      "segmentation": "week000/seg_mask.nii.gz"
    }
  ]
}
```

Visits must be strictly increasing in weeks, within one longitudinal segment (do not bridge surgery/resection boundaries). The horizon is a finite positive number of weeks **after the final historical visit**. No future image, mask, clinical outcome, target registration or target-derived crop is accepted. Unknown fields, including target/future fields, are rejected. The caller is responsible for supplying only information available at forecast time; file content cannot prove its own provenance.

All images and masks must already be in the LUMIERE atlas grid: 182×218×182, orthogonal 1 mm voxels, matching finite affines. Input axes are retained, without canonical reorientation. Explicit spatial units must be mm; unknown units are interpreted as mm under this contract. Arbitrary native-space MRI is not supported. Historical segmentations must use **0=background, 1=enhancing tumor, 2=necrosis/nonenhancing core, 3=edema**, not a different BraTS label convention. WT={1,2,3}, TC={1,2}, ET={1}. Nonfinite values, incompatible grids and invalid labels fail validation.

Each visit may optionally include `"cache": "path/to/scan.npy"`. This is an authoritative, precomputed 7×91×109×91 array in training channel order (four normalized images, then three region fractions). Original MRI and segmentation paths remain required for input/coordinate validation. Cache provenance and agreement with those source files are the caller's responsibility. Caches are only read. Do not supply caches built in a future scan's frame.

For a `registered` bundle, the second-to-last visit must supply its existing aligned cache and `"aligned_to_week": <latest historical week>`. It must have been prepared by the history-pair portion of `register_step2_pairs.py`, using only that visit and the latest historical visit. The last visit stays in its own atlas frame; its optional cache must be unregistered. A single-visit history needs no alignment. The CLI deliberately does not run the cohort registration command, which also prepares future targets for evaluation. Automatic registration from raw history for registered bundles is not implemented.

## Model bundle contract, version 1

A bundle is a JSON descriptor plus a checkpoint (paths relative to the descriptor or absolute):

```json
{
  "version": 1,
  "variant": "static",
  "base": 16,
  "frame": "atlas",
  "preprocessing": "lumiere-atlas-pool2-float16-v1",
  "checkpoint": "/path/to/runs/mask_forecast/static/fold0/best.pt"
}
```

- `variant`: `static` or `ode`; `base` must match training (normally 16).
- Checkpoint: plain PyTorch `state_dict`, as saved in existing `best.pt`/`final.pt`. Loading uses `weights_only=True`, CPU mapping and strict key/shape matching. Training resume dictionaries (`last_state.pt`) and 2D model weights are unsupported.
- `frame`: `atlas` for atlas-trained weights, `registered` for last-scan-frame-trained weights. State dictionaries do not encode frame or preprocessing; the bundle author must identify them correctly. Existing weights need no conversion or modification.
- Architecture: 12 input channels; ODE uses existing fixed two-step RK4; persistence logit scale 8. Time gaps and horizon use weeks/52. Only the latest two visits affect the network, even when more history is validated.
- Preprocessing reproduces `build_atlas_cache.py`: per-modality positive-voxel z-score, standard deviation +1e-6, clip [-5,5], zero background; 2×2×2 mean pooling; region occupancy pooling; float16 round trip. No target statistics or target geometry is used.
- Inputs: latest seven channels, previous three masks (zero if absent), presence flag and previous gap; pad high sides to 96×112×96 with zeros. Output threshold is strictly logit >0, as in held-out evaluation. Region nesting is not repaired or forced.

## Outputs and coordinates

- `wt.nii.gz`, `tc.nii.gz`, `et.nii.gz`: independent uint8 binary masks, unpadded 91×109×91. No mutually exclusive combined label image is implied.
- `volumes.json`: predicted and last-observed region volumes in mL, calculated from the absolute affine determinant /1000.
- `metadata.json`: source paths, history/used weeks, forecast week, horizon, frame, source and output affines, output shape, units, checkpoint SHA-256, bundle configuration, device, Torch version and threshold. Source paths may contain patient identifiers; handle this output accordingly.
- `overlay.png`: three region overlays on the latest pooled FLAIR slice selected from the union of historical/predicted WT. Axes show array voxel indices, not anatomical orientation labels. This is a QC illustration, not a full-volume review.

Pooling centers output voxel `(i,j,k)` at source voxel `(2i+.5,2j+.5,2k+.5)`. Consequently `A_output = A_source @ [[2,0,0,.5],[0,2,0,.5],[0,0,2,.5],[0,0,0,1]]`. Both NIfTI sform and qform carry this affine, with mm units. Merely doubling voxel spacing without the half-voxel offset would place the pooled mask incorrectly. The output is in the latest scan's atlas frame, not its original native acquisition frame. No inverse transforms are inferred.

Export removes padding; the legacy evaluation scores the padded field, so its volumes can include predictions outside the real grid. CPU float32 inference can also differ near zero logits from historical CUDA mixed-precision evaluation. The tests compare CPU inputs and model behavior, not promise bitwise reproduction of GPU results. Horizons outside training experience and new acquisition distributions have not been validated. There are no uncertainty estimates, automatic image-quality decisions or treatment recommendations.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/tide-prediction-mpl \
  /home/tusharn/ppr/glioblastoma-evolution/.venv/bin/python \
  -m unittest discover -s tests -p 'test_prediction.py' -v
```

Tests cover missing segmentation, future-field rejection, chronology/horizon validation, geometry and labels, registered-history requirements, exact CPU input equivalence with `MaskForecastDataset`, operation without a target file, checkpoint loading and forward passes for both architectures, single-visit flags, NIfTI affine/units/shape, physical volumes, PNG creation and overwrite protection.

Local validation on 2026-11-03 also read the existing atlas static/ODE fold-0 `best.pt` checkpoints and held-out Patient-005 week-000-2 history, with a 15-week horizon. Raw preprocessing matched the stored input cache exactly. Both variants reproduced saved held-out WT/TC/ET volumes (146.968 / 81.888 / 0.720 mL) without loading any future scan or mask. These particular checkpoints produce persistence on this case; the check does not establish performance gains or general equivalence across trained checkpoints. The direct CLI also exported all artifacts successfully on CPU.

## Integrated release interface

`--bundle` also accepts a checksum-verified packaging directory, or a JSON
`mask_cv_ensemble` descriptor with `version: 1`, five relative `members`, a `label`,
and `aggregation: "mean_logits"`. The logits are averaged before thresholding;
frames must agree. These are **cross-validation ensembles**, not all-cohort mask
refits. Their component CV scores do not estimate ensemble generalization.

Volume bundles use the same CLI with `--trust-volume-bundle` (local trusted pickle).
The volume history JSON has `version: 1`, `patient_id`, and increasing `history`
entries containing `week` and `volumes_mm3`. Volume keys are exactly
`Necrotic_NonEnhancing`, `Enhancing_Core`, `Edema_Compartment`. Optional `clinical`
contains string values for `Age at surgery (years)`, `Sex`, `MGMT qualitative`.
No MRI, target, cohort files, or future segmentations are loaded by volume inference.
Supply only visits within the latest surgery-separated segment. Outputs are
non-overlapping compartment volumes in mL; these differ from mask WT/TC/ET.

Example: `.venv/bin/python scripts/predict.py --history volume-history.json
--horizon-weeks 12 --bundle runs/packaging/release-20261103/volume-all
--trust-volume-bundle --output-dir runs/new-volume-forecast`.

Mask inference requires all four historical modalities and existing labels on the
atlas grid. Registered inference requires history-only prepared alignment. A cache
override is a trusted prepared input; its declared relation to MRI is not independently
certified by the CLI. No new registration or native-space segmentation is performed.

## Forecast ranges for volume bundles

Add `--intervals` with the calibration file built for that bundle:

```bash
.venv/bin/python scripts/predict.py --history volume-history.json --horizon-weeks 12 \
  --bundle runs/packaging/release-20261103/volume-all --trust-volume-bundle \
  --intervals runs/packaging/release-20261103/volume-intervals.json \
  --output-dir runs/new-volume-forecast
```

For a made-up two-visit history (36.5 mL total at week 10), the terminal shows:

```text
Forecast for week 22 (12 weeks after the last scan), mL:
  Necrotic_NonEnhancing  last     3.5  forecast     3.9  ranges 50% 1.8-7.8  80% 0.8-13.2  90% 0.3-22.0  95% 0.0-34.3
  Enhancing_Core         last     9.0  forecast     7.2  ranges 50% 4.0-15.3  80% 2.3-41.3  90% 1.3-67.8  95% 0.5-108.1
  Edema_Compartment      last    24.0  forecast    29.0  ranges 50% 21.6-37.3  80% 14.8-54.4  90% 10.9-82.5  95% 8.2-113.9
  Total                  last    36.5  forecast    40.1  ranges 50% 30.6-56.9  80% 22.1-80.2  90% 13.7-120.6  95% 9.9-150.5
  Held-out coverage of the total-volume ranges (averaged over patients): 50% range -> 45%, 80% range -> 79%, 90% range -> 89%, 95% range -> 95%
```

`volumes.json` is unchanged. A new `intervals.json` holds:

- `volumes`: for each compartment and `Total`, the last observed volume, the forecast, and
  the 50/80/90/95% `ranges` in mL.
- `held_out_patient_coverage`: what each range achieved on held-out LUMIERE patients
  (EXPERIMENTS.md section 11).
- `horizon_within_calibration` and `warning`. The warning is set beyond 26 weeks: only 5%
  of LUMIERE intervals were that long, and the 80% range held 70% of next volumes there. It
  is also set beyond the longest interval seen (155 weeks).
- The calibration file path and SHA-256. Both are also recorded in `metadata.json`.

How the ranges are made. Residuals come from patient-grouped out-of-fold forecasts made
with the released GBM settings. A small scale model widens the range for longer horizons,
smaller tumors and single-scan histories. Conformal quantiles then set each level. The
calibration file records the SHA-256 of the bundle's `estimators.pkl`, and the CLI refuses
a calibration built for a different bundle. It also refuses one whose scale features or
floor differ from the code. Build a new file with
`scripts/volume_intervals.py calibrate --bundle <volume-all> --output <new file>`; it
refuses to overwrite, and `scripts/build_research_release.sh` runs it.

What the ranges mean:
- Coverage is marginal. Across LUMIERE-like visits, an 80% range contains the next volume
  about 80% of the time. It is not a guarantee for an individual patient.
- The ranges are wide. A typical 80% total-volume range runs from ×0.53 to ×2.1 of the
  forecast.
- Each compartment and the total are calibrated separately, so the compartment ranges need
  not add up to the total range.
- They assume DeepBraTumIA compartment volumes, no surgery in the interval and LUMIERE-like
  scans.
- Mask forecasts have no ranges.

## Progression call (is the current scan RANO PD?)

A `progression_gbm` bundle estimates the probability that an expert would rate the current
scan as progressive disease under RANO. It holds two models:
- `volumes` reads the automated compartment volumes of the current scan and of the earlier
  scans since the last surgery, the previous scan's expert rating, and optionally age, sex
  and MGMT. Held-out AUROC is 0.82.
- `volumes_masks` also reads the atlas-space segmentations of the previous and current
  scan. It measures new enhancing tumor, new enhancing components and new FLAIR abnormality
  outside the prior tumor, with an 8 mm margin and no registration. Held-out AUROC is 0.835.
  The CLI uses it whenever both segmentations are given.

Held-out evaluation is in EXPERIMENTS.md section 12b. Run it with:

```bash
.venv/bin/python scripts/predict.py --history scan.json \
  --bundle runs/packaging/release-20261103/progression-gbm --trust-progression-bundle --output-dir runs/my-call
```

`--horizon-weeks` and `--intervals` do not apply and are rejected.

History JSON (`version: 1`): `patient_id`, `history` and optional `clinical`, which takes
the same string fields as the volume forecast.
- `history` lists visits oldest first, each with `week` and `volumes_mm3`. The keys are
  `Necrotic_NonEnhancing`, `Enhancing_Core` and `Edema_Compartment`, in mm³.
- The last visit is the scan to assess and must not carry a rating.
- The visit before it must carry `rating`: one of `PD`, `SD`, `PR`, `CR`, `Post-Op` or
  `unknown`.
- Only the first visit may be `Post-Op`. A later `Post-Op` means a new surgery, so start
  the history there.
- Optionally, `segmentation` gives a path to a LUMIERE atlas-space label map, absolute or
  relative to the JSON file. It must be 182×218×182 at 1 mm, with labels 0, 1=enhancing,
  2=necrosis and 3=edema. Give it for both of the last two visits or for neither. Both
  files must share one atlas grid. They are pooled to 2 mm exactly as the training caches
  were.
- Weeks must strictly increase and volumes must be nonnegative. Unknown fields are
  rejected.

`progression.json` contains:
- `probability_pd` and `call`, one of three bands:
  - "likely PD": at or above the 80%-specificity threshold;
  - "PD less likely": below the 90%-sensitivity threshold;
  - "uncertain": in between.
- `held_out_pd_rate_in_this_band`: how often held-out scans in that band were rated PD.
  For the volumes model it is 85%, 49% and 28% for the three bands; for volumes_masks it is
  86%, 41% and 32%.
- The thresholds with their held-out sensitivity and specificity, and the held-out AUROC.
- `model`: `volumes` or `volumes_masks`.
- With segmentations, `new_tumor_outside_prior`: new enhancing mL, new enhancing
  components, new FLAIR abnormality mL, and the fraction of enhancing tumor inside the
  prior core.
- Plain enhancing and total volume changes against the previous scan and the nadir.
- The input checksum and bundle path.

The bundle (`progression_model.py fit-all`) holds a scikit-learn pickle (load only trusted
bundles), a manifest and `SHA256SUMS.json`. Loading checks the checksums, the scikit-learn
version, the feature list, and the hashes of `progression_model.py`,
`volume_trajectory_experiment.py`, `lumiere_cohort.py`, `lumiere_weeks.py` and
`build_atlas_cache.py`. Editing any of those files requires refitting the bundle.

Limits:
- Subtle T2/FLAIR progression is mostly missed (39% found with masks). Stable scans usually
  land in "uncertain".
- The model was fitted to one centre's expert readings and DeepBraTumIA volumes.
- It is not clinically validated.
