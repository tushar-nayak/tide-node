"""History-only inference for the step-2 3D mask models (no cohort/target loading)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from build_atlas_cache import CACHE_SHAPE, MODS, pool2
from mask_forecast_experiment import MaskForecastNet, REGIONS, pad

PREPROCESSING = "lumiere-atlas-pool2-float16-v1"


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def keys(obj, required, optional=()):
    if not isinstance(obj, dict) or set(obj) - set(required) - set(optional) or set(required) - set(obj):
        raise ValueError(f"Expected fields {sorted(required)}; optional {sorted(optional)}")


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def resolve(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("File paths must be nonempty strings")
    path = (root / value).resolve()
    if not path.is_file():
        raise ValueError(f"Missing input file: {path}")
    return path


def load_bundle(path):
    path = Path(path).resolve()
    if path.is_dir():
        from package_models import load_mask
        model, manifest = load_mask(path)
        return model, dict(frame=manifest['preprocessing']['frame'], manifest=manifest,
                           bundle_directory=str(path))
    config = read_json(path)
    if config.get('kind') == 'mask_cv_ensemble':
        keys(config, ('version', 'kind', 'members', 'label', 'aggregation'))
        if config['version'] != 1 or config['aggregation'] != 'mean_logits' or len(config['members']) != 5:
            raise ValueError('Expected five CV members with mean_logits aggregation')
        if not isinstance(config['members'], list) or any(not isinstance(m, str) for m in config['members']):
            raise ValueError('Members must be bundle directory paths')
        member_paths = [(path.parent / member).resolve() for member in config['members']]
        if len(set(member_paths)) != 5 or not all(m.is_dir() for m in member_paths):
            raise ValueError('Expected five distinct packaged fold directories')
        loaded = [load_bundle(member) for member in member_paths]
        manifests = [meta['manifest'] for _, meta in loaded]
        if ({m['provenance'].get('fold') for m in manifests} != set(range(5)) or
            any(m['provenance'].get('all_cohort') is not False for m in manifests) or
            len({json.dumps(m['architecture']['constructor'], sort_keys=True) for m in manifests}) != 1):
            raise ValueError('Expected compatible CV folds 0 through 4')
        frames = {meta['frame'] for _, meta in loaded}
        if len(frames) != 1:
            raise ValueError('Ensemble coordinate frames differ')
        return MaskEnsemble([model for model, _ in loaded]), dict(config, frame=frames.pop(),
                    member_metadata=[meta for _, meta in loaded])
    keys(config, ('version', 'variant', 'base', 'frame', 'preprocessing', 'checkpoint'))
    if config['version'] != 1 or config['variant'] not in ('static', 'ode') or config['frame'] not in ('atlas', 'registered'):
        raise ValueError("Unsupported bundle version, variant or frame")
    base = config['base']
    if type(base) is not int or (base not in (1, 2, 4) and (base < 8 or base % 8)):
        raise ValueError("base must be a positive architecture-compatible integer")
    if config['preprocessing'] != PREPROCESSING:
        raise ValueError("Unsupported preprocessing contract")
    checkpoint = resolve(path.parent, config['checkpoint'])
    model = MaskForecastNet(12, base, config['variant'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    config = dict(config, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return model, config


def load_volume(path, reference=None):
    img = nib.load(path)
    affine = img.affine
    if img.shape != (182, 218, 182) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Expected finite 182x218x182 atlas geometry")
    if not np.allclose(affine[:3, :3].T @ affine[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError("Expected orthogonal 1 mm atlas voxels")
    if img.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
        raise ValueError("Atlas coordinates must use millimetres")
    if reference is not None and not np.allclose(affine, reference, atol=1e-4):
        raise ValueError("History images/masks must share the declared atlas grid")
    data = np.asarray(img.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite image values: {path}")
    return data, affine


def prepare_history(path, horizon, frame):
    horizon = number(horizon, 'horizon_weeks', positive=True)
    path = Path(path).resolve()
    manifest = read_json(path)
    keys(manifest, ('version', 'patient_id', 'frame', 'history'))
    if manifest['version'] != 1 or manifest['frame'] != frame:
        raise ValueError("Manifest version/frame does not match bundle")
    if not isinstance(manifest['patient_id'], str) or not manifest['patient_id']:
        raise ValueError("patient_id must be a nonempty string")
    history = manifest['history']
    if not isinstance(history, list) or not history:
        raise ValueError("At least one historical scan with an existing segmentation is required")
    weeks, arrays, sources = [], [], []
    reference = None
    for entry in history:
        if isinstance(entry, dict) and 'segmentation' not in entry:
            raise ValueError('Each historical visit requires an existing segmentation; MRI-only inference is unsupported')
        keys(entry, ('week', 'modalities', 'segmentation'), ('cache', 'aligned_to_week'))
        week = number(entry['week'], 'week')
        if weeks and week <= weeks[-1]:
            raise ValueError("History weeks must be strictly increasing")
        weeks.append(week)
        keys(entry['modalities'], MODS)
        paths = {m: resolve(path.parent, entry['modalities'][m]) for m in MODS}
        paths['SEG'] = resolve(path.parent, entry['segmentation'])
        chans = []
        for mod in MODS:
            v, affine = load_volume(paths[mod], reference)
            if reference is None:
                reference = affine
            m = v > 0
            if not m.any():
                raise ValueError(f"{mod} has no positive brain voxels")
            chans.append(pool2(np.where(m, np.clip((v - v[m].mean()) / (v[m].std() + 1e-6), -5, 5), 0.0)))
        seg, _ = load_volume(paths['SEG'], reference)
        if not np.isin(seg, (0, 1, 2, 3)).all():
            raise ValueError("Segmentation must use LUMIERE labels 0,1=ET,2=necrosis,3=edema")
        for labels in ((1, 2, 3), (1, 2), (1,)):
            chans.append(pool2(np.isin(seg, labels).astype(np.float32)))
        arr = np.stack(chans).astype(np.float16).astype(np.float32)
        source = {k: str(v) for k, v in paths.items()}
        if 'cache' in entry:
            cache = resolve(path.parent, entry['cache'])
            arr = np.asarray(np.load(cache, allow_pickle=False), dtype=np.float32)
            if arr.shape != CACHE_SHAPE or not np.isfinite(arr).all() or (arr[4:] < 0).any() or (arr[4:] > 1).any():
                raise ValueError("Cache must be finite 7x91x109x91 with region fractions in [0,1]")
            source['cache'] = str(cache)
        arrays.append(arr)
        arrays = arrays[-2:]
        sources.append(source)
    for i, entry in enumerate(history):
        if frame == 'registered' and i == len(history) - 2:
            if 'cache' not in entry or entry.get('aligned_to_week') != weeks[-1]:
                raise ValueError("Registered bundle requires previous cache aligned using history only to latest week")
        elif 'aligned_to_week' in entry:
            raise ValueError("aligned_to_week is only valid for the previous registered scan")
    last = arrays[-1]
    prev = arrays[-2][4:7] if len(arrays) > 1 else np.zeros_like(last[4:7])
    flag, gap = (1., (weeks[-1] - weeks[-2]) / 52.) if len(arrays) > 1 else (0., 0.)
    const = np.ones((2,) + last.shape[1:], dtype=np.float32) * np.array([flag, gap], dtype=np.float32)[:, None, None, None]
    batch = {'x': torch.from_numpy(pad(np.concatenate([last, prev, const])))[None],
             'last_mask': torch.from_numpy(pad(last[4:7]))[None],
             'dt': torch.tensor([horizon / 52.], dtype=torch.float32)}
    # A pooled voxel is centered on the center of its eight source voxels.
    pooling = np.diag([2., 2., 2., 1.])
    pooling[:3, 3] = .5
    metadata = dict(patient_id=manifest['patient_id'], history_weeks=weeks,
                    used_history_weeks=weeks[-2:], horizon_weeks=horizon, forecast_week=weeks[-1] + horizon,
                    frame=frame, affine=(reference @ pooling).tolist(), source_affine=reference.tolist(),
                    sources=sources, shape=list(CACHE_SHAPE[1:]), spatial_units='mm')
    return batch, last, metadata


@torch.inference_mode()
def infer(model, batch):
    logits, _ = model(batch['x'], batch['last_mask'], batch['dt'])
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced nonfinite logits")
    return (logits[0, :, :91, :109, :91] > 0).cpu().numpy()


def export_prediction(output, masks, last, metadata):
    # Exclusive creation avoids overwriting prior predictions or input data.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    affine = np.asarray(metadata['affine'])
    voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.
    volumes = {}
    for i, region in enumerate(REGIONS):
        img = nib.Nifti1Image(masks[i].astype(np.uint8), affine)
        img.header.set_xyzt_units('mm')
        img.set_sform(affine, code=2)
        img.set_qform(affine, code=2)
        nib.save(img, output / f'{region}.nii.gz')
        volumes[region] = dict(predicted_ml=float(masks[i].sum() * voxel_ml),
                               last_ml=float((last[4+i] > .5).sum() * voxel_ml))
    (output / 'volumes.json').write_text(json.dumps(volumes, indent=2) + '\n')
    metadata.update(voxel_ml=voxel_ml, threshold='logit > 0', padding_removed=True,
                    region_order=list(REGIONS), region_nesting_enforced=False,
                    device='cpu', torch_version=str(torch.__version__))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = int(np.argmax(np.logical_or(masks[0], last[4] > .5).sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(last[0, :, :, z].T, cmap='gray', origin='lower')
        ax.imshow(np.ma.masked_where(~masks[i, :, :, z].T, masks[i, :, :, z].T),
                  cmap='autumn', vmin=0, vmax=1, alpha=.5, origin='lower')
        ax.set_title(f'{REGIONS[i].upper()} forecast; voxel k={z}')
        ax.set_xlabel('voxel i'); ax.set_ylabel('voxel j')
    fig.tight_layout()
    fig.savefig(output / 'overlay.png', dpi=120)
    plt.close(fig)


class MaskEnsemble(torch.nn.Module):
    """Sequential CPU evaluation; CV ensembles have no independent test estimate."""
    def __init__(self, models):
        super().__init__()
        self.members = torch.nn.ModuleList(models)
        self.eval()

    def forward(self, x, last_mask, dt):
        total = None
        for model in self.members:
            logits, _ = model(x, last_mask, dt)
            total = logits if total is None else total + logits
        return total / len(self.members), None


def predict_volume_history(history_path, horizon, bundle_path, output, trusted=False):
    from types import SimpleNamespace
    from package_models import load_volume as load_volume_bundle, sha256
    from volume_trajectory_experiment import history_features, static_features, y_of
    horizon = number(horizon, 'horizon_weeks', positive=True)
    doc = read_json(history_path)
    keys(doc, ('version', 'patient_id', 'history'), ('clinical',))
    if doc['version'] != 1 or not isinstance(doc['patient_id'], str) or not doc['patient_id']:
        raise ValueError('Invalid volume history version/patient_id')
    if not isinstance(doc['history'], list) or not doc['history']:
        raise ValueError('Nonempty history required')
    scans = []
    # Use the exact experiment compartment names, including necrotic/non-enhancing.
    from lumiere_cohort import COMPARTMENTS
    for entry in doc['history']:
        keys(entry, ('week', 'volumes_mm3'))
        week = number(entry['week'], 'week')
        keys(entry['volumes_mm3'], COMPARTMENTS)
        volumes = {k: number(v, k) for k, v in entry['volumes_mm3'].items()}
        if min(volumes.values()) < 0 or (scans and week <= scans[-1].week):
            raise ValueError('Nonnegative volumes and increasing weeks required')
        scans.append(SimpleNamespace(week=week, volumes_mm3=volumes))
    clinical = doc.get('clinical', {})
    keys(clinical, (), ('Age at surgery (years)', 'Sex', 'MGMT qualitative'))
    if any(not isinstance(v, str) for v in clinical.values()):
        raise ValueError('Clinical values must be strings')
    t = SimpleNamespace(history=scans, last=scans[-1], dt_weeks=horizon)
    x = history_features(t, static_features(doc['patient_id'], {doc['patient_id']: clinical}))[None]
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite clinical features')
    models, manifest = load_volume_bundle(bundle_path, trusted=trusted)
    pred = np.expm1(np.maximum(y_of(scans[-1]) + np.array([m.predict(x)[0] for m in models]), 0))
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite volume forecast')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'volumes.json').write_text(json.dumps(dict(zip(COMPARTMENTS, pred.tolist())), indent=2) + '\n')
    (output / 'metadata.json').write_text(json.dumps(dict(patient_id=doc['patient_id'],
        horizon_weeks=horizon, forecast_week=scans[-1].week+horizon, output_unit='mL',
        history_sha256=sha256(history_path), bundle=manifest, clinical_validation=False), indent=2) + '\n')
