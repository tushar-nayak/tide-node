# Moving to a New Machine: Dataset Download + Registration

This walks through getting the LUMIERE dataset onto a fresh machine and pre-registering
it, so that machine can run this project's training pipeline without repeating the
~1-1.5 day registration step every time.

## What you need

- ~120GB free disk: ~32GB for the downloaded zip, ~32GB extracted, ~50GB for the
  registered cache (registration produces full 3D volumes per patient/target/moving-week
  triple, not small files).
- Python 3.11+ with `requirements.txt` installed (`python3 -m pip install -r requirements.txt`).
- A multi-core CPU. Registration is CPU-bound (SimpleITK affine registration); a GPU helps
  the later training step but does nothing for this stage.

## Step 1: Download the dataset

This is the **LUMIERE dataset**: Suter, Y., Knecht, U., Valenzuela, W., Notter, M., Hewer,
E., Schucht, P., Wiest, R. and Reyes, M. (2022). *The LUMIERE dataset: Longitudinal
Glioblastoma MRI with expert RANO evaluation.* Scientific Data, 9(1), 768.
(DOI: `10.1038/s41597-022-01881-7`). Confirmed against what's locally on disk: the paper
reports 91 GBM patients / 638 study dates, which matches this project's local copy exactly.

- **Source**: Figshare collection --
  https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905
- **License**: non-commercial use (per the dataset's own companion code repo,
  https://github.com/ysuter/gbm-data-longitudinal). Record your access date when you
  download it -- `PUBLICATION_PROTOCOL.md` requires this for any publication claim.
- **Account/DUA requirement**: not confirmed from the sources checked -- verify on the
  Figshare page itself when you go to download.
- The Imaging item within that collection is what was locally renamed
  `Imagingv202211.zip` (that filename itself isn't an official convention from the
  dataset's own documentation -- it looks like a local naming choice, presumably noting a
  November 2022 access/version date. Whatever the collection's actual imaging item is
  named when you download it, that's the file to extract in Step 2.

Whatever it's named, the archive is expected to unzip into a
`Patient-XXX/week-YYY/...` directory structure (see "Expected layout" below) -- if the
downloaded archive has a different top-level layout, `scripts/setup_lumiere_data.sh` will
try to locate an `Imaging/` directory inside it automatically, but double check the result.

### Expected layout

Each patient/week directory should contain, at minimum:

```
Patient-XXX/week-YYY/DeepBraTumIA-segmentation/atlas/skull_strip/
  flair_skull_strip.nii.gz
  t1_skull_strip.nii.gz
  t2_skull_strip.nii.gz
  ct1_skull_strip.nii.gz
Patient-XXX/week-YYY/DeepBraTumIA-segmentation/atlas/segmentation/
  seg_mask.nii.gz                  (real tumor segmentation; used for the wt_dice/tc_dice/et_dice metrics)
  measured_volumes_in_mm3.json
```

A patient/week is only usable if all four skull-stripped modality files are present;
weeks missing any of them are silently skipped by the dataset loader. The segmentation
mask is optional (present for ~94% of weeks in the version of the dataset already on this
machine) -- weeks without it just get `None` for the mask-based metrics.

## Step 2 + 3: Extract and register

Run this on the new machine (inside tmux/screen or with `nohup`, since registration alone
takes ~1-1.5 days on a 12-core machine):

```bash
tmux new-session -s lumiere_setup 'bash scripts/setup_lumiere_data.sh path/to/Imagingv202211.zip'
```

If you've already extracted the archive yourself (or copied over an existing
`data/lumiere/extracted/Imaging` directory from another machine, which skips the ~32GB
download+extract entirely), omit the zip argument:

```bash
tmux new-session -s lumiere_setup 'bash scripts/setup_lumiere_data.sh'
```

This script:
1. Extracts the zip (skipped if `data/lumiere/extracted/Imaging` already exists).
2. Runs `scripts/preprocess_lumiere_registration.py` across the full cohort in 8
   load-balanced parallel batches, writing to `data/lumiere_registered/`.
3. Consolidates the registration manifest and verifies the cache is non-empty.

It is safe to interrupt and re-run: extraction is skipped if already done, and
registration skips any (patient, target week, moving week) triple already on disk.

**Only copying the cache instead?** If you already have a `data/lumiere_registered/`
directory from another machine with this same dataset version, you can skip this whole
script and just copy that ~50GB directory over directly -- registration output doesn't
depend on which machine produced it, only on the source volumes.

## Step 4: Verify

```bash
python3 -c "
import json
m = json.load(open('data/lumiere_registered/registration_manifest.json'))
print('patients:', len(m['patients']))
print('jobs:', m['generated_job_count'] + m['reused_job_count'])
"
```

On the 81-patient cohort this project currently tracks, expect `81` patients and `2278`
jobs. If the new machine's LUMIERE release has a different patient/week composition,
these numbers will differ -- that's fine, just sanity-check they're nonzero and roughly
proportional.

## Step 5: Use it

Point any training run at the registered cache instead of the raw imaging directory:

```bash
python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir data/lumiere/extracted/Imaging \
  --registered-data-dir data/lumiere_registered \
  --separate-patient-runs \
  --holdout-last-pair \
  --epochs 40 \
  --model-size standard \
  --device cuda \
  --run-name my_run
```

or use `scripts/run_full_cohort_validation.sh` directly, which assumes this setup has
already been done (it has its own copy of the registration step for a from-scratch run on
*this* machine; on a new machine, run `setup_lumiere_data.sh` first, or the two would
duplicate the same ~1.5-day registration work).

## Why this only needs to happen once per machine

Registration operates on full 3D volumes and, for every target week, registers every
prior week to it -- the complete superset of pairs any history-conditioning strategy
(prefix vs sliding history, any context size, any model config) could ever need. Slice
extraction happens later, at training time, and is fully decoupled from this step. So once
this finishes, any future training run that passes `--registered-data-dir
data/lumiere_registered` reads straight from disk and never re-registers anything,
regardless of what else changes about the experiment.

The one exception: if LUMIERE ever releases more patients or more weeks for existing
patients, those new pairs would need registering -- but that's incremental work (the
script skips everything already cached), not a redo of the whole cohort.
