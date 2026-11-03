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
