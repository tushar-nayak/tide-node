"""Generate retrospective case-review sheets and structural quality flags.

Expert labels select error cases only. No predictions are refitted or modified.
Header checks do not establish registration accuracy or segmentation correctness.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from .report import measure

MODALITIES = ('flair', 't1', 't2', 'ct1')


def files(root, patient, visit):
    atlas = root / patient / visit / 'DeepBraTumIA-segmentation' / 'atlas'
    paths = {}
    for mod in MODALITIES:
        candidates = sorted((atlas / 'skull_strip').glob(f'{mod}_skull_strip.nii*'))
        paths[mod] = candidates[0] if candidates else None
    paths['mask'] = atlas / 'segmentation' / 'seg_mask.nii.gz'
    return paths


def geometry(paths):
    flags, images = [], {}
    for name, path in paths.items():
        if path is None or not path.is_file():
            flags.append(f'missing_{name}')
            continue
        try:
            im = nib.load(path)
            if len(im.shape) != 3 or not all(n > 0 for n in im.shape):
                flags.append(f'invalid_shape_{name}')
            if not np.isfinite(im.affine).all() or abs(np.linalg.det(im.affine[:3, :3])) < 1e-8:
                flags.append(f'invalid_affine_{name}')
            if not np.isfinite(im.header.get_zooms()).all() or min(im.header.get_zooms()) <= 0:
                flags.append(f'invalid_spacing_{name}')
            images[name] = im
        except (OSError, ValueError, nib.filebasedimages.ImageFileError):
            flags.append(f'unreadable_header_{name}')
    ref = images.get('flair')
    if ref is not None:
        for name, im in images.items():
            if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-3, rtol=0):
                flags.append(f'within_scan_grid_mismatch_{name}')
    return sorted(flags), images


def select_cases(rows, per_group):
    groups = {'false_negative': [], 'false_positive': [], 'true_positive': [], 'true_negative': []}
    for row in rows:
        truth, pred = int(row['target']), int(row['binary_prediction'])
        key = ('true_' if truth == pred else 'false_') + ('positive' if pred else 'negative')
        groups[key].append(row)
    selected = []
    for group, candidates in groups.items():
        # Deterministic patient-diverse sampling, not visually cherry-picked images.
        candidates.sort(key=lambda r: hashlib.sha256(f"{r['patient_id']}/{r['visit']}".encode()).hexdigest())
        used = set()
        for row in candidates:
            if row['patient_id'] in used:
                continue
            selected.append((group, row))
            used.add(row['patient_id'])
            if len(used) >= per_group:
                break
    return groups, selected


def sheet(root, record, prediction, group, destination):
    visits = [record['payload']['history'][-2], record['payload']['history'][-1]]
    patient = record['patient_id']
    # Resolve exact source directory for the previous numerical week via the saved cohort;
    # shared deterministic acquisition resolver is used only for that historical visit.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from lumiere_weeks import resolve_patient_weeks
    resolved = resolve_patient_weeks(root / patient)
    sources = [resolved[int(visits[0]['week'])].source_name, record['visit']]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    flags = []
    for col, source in enumerate(sources):
        f, ims = geometry(files(root, patient, source))
        flags.extend([f'{source}:{x}' for x in f])
        mask_im = ims.get('mask')
        mask = np.asarray(mask_im.dataobj) if mask_im is not None else None
        z = int(np.argmax((mask > 0).sum(axis=(0, 1)))) if mask is not None and mask.ndim == 3 and (mask > 0).any() else None
        if mask is not None:
            if not np.isfinite(mask).all():
                flags.append(f'{source}:nonfinite_mask')
            if not set(np.unique(mask)).issubset({0, 1, 2, 3}):
                flags.append(f'{source}:unexpected_mask_labels')
            if not (mask > 0).any():
                flags.append(f'{source}:empty_tumor_mask_review')
        for row, mod in enumerate(('flair', 'ct1')):
            ax = axes[row, col]
            im = ims.get(mod)
            if im is None or len(im.shape) != 3:
                ax.text(.1, .5, f'Missing/invalid {mod}')
            else:
                volume = np.asarray(im.dataobj, dtype=np.float32)
                if not np.isfinite(volume).all():
                    flags.append(f'{source}:nonfinite_{mod}')
                zz = min(z if z is not None else im.shape[2] // 2, im.shape[2] - 1)
                sl = np.nan_to_num(volume[:, :, zz]).T
                lo, hi = np.percentile(sl, [1, 99])
                ax.imshow(sl, cmap='gray', origin='lower', vmin=lo, vmax=max(hi, lo+1e-6))
                if mask is not None and mask.shape == im.shape and np.allclose(mask_im.affine, im.affine, atol=1e-3, rtol=0):
                    for label, color in [(1, '#ff5555'), (2, '#ffcc55'), (3, '#55ccff')]:
                        binary = (mask[:, :, zz] == label).T
                        if binary.any() and not binary.all():
                            ax.contour(binary, levels=[.5], colors=[color], linewidths=.7)
                note = ' | empty/missing mask: central slice' if z is None else ''
                ax.set_title(f'{source} | {mod.upper()} | axial index {zz}{note}', fontsize=9)
            ax.axis('off')
    fig.suptitle(f"{patient} | {group.replace('_',' ')}\nExpert PD={prediction['target']}; research p={float(prediction['probability']):.3f}; seed={prediction['seed']}\nEach visit in its own grid; maximal tumor area, or central slice if no mask", fontsize=10)
    fig.text(.5, .025, 'Red: enhancing; gold: necrotic/non-enhancing; blue: edema. Anatomical acceptance: pending human review.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.90))
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return sorted(set(flags))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/assessment_v1'))
    parser.add_argument('--data', type=Path, default=Path('data/lumiere/extracted/Imaging'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--per-group', type=int, default=3)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error('--per-group must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    cohort_path = args.results / 'cohort.json'
    pred_path = args.results / 'outer_predictions.csv'
    cohort = json.loads(cohort_path.read_text())
    rows = [r for r in csv.DictReader(pred_path.open()) if r['variant']=='volumes' and r['method']=='boosted' and int(r['seed'])==args.seed]
    lookup = {(r['patient_id'], r['visit']): r for r in cohort}
    if not rows or len(rows) != len({(r['patient_id'], r['visit']) for r in rows}):
        raise ValueError('Empty or duplicate predictions for selected seed')
    quality, cache = [], {}
    for rec in cohort:
        measure(rec['payload'])
        key = rec['patient_id'], rec['visit']
        if key not in cache:
            cache[key] = geometry(files(args.data, *key))[0]
        m = measure(rec['payload'])['measurements']
        # These are review-priority heuristics, not validated failure thresholds.
        flags = list(cache[key])
        if m['Total']['current_ml'] == 0:
            flags.append('empty_current_tumor_review')
        previous, current = m['Total']['previous_ml'], m['Total']['current_ml']
        if previous > 0 and (current > 4*previous or current < previous/4):
            flags.append('large_volume_change_review')
        quality.append({'patient_id': key[0], 'visit': key[1], 'flags': flags,
                        'header_checks': 'flagged' if cache[key] else 'passed',
                        'voxel_checks': 'not_performed', 'anatomical_review': 'pending', 'quality_admitted': False})
    groups, selected = select_cases(rows, args.per_group)
    sheets, review_rows = [], []
    for group, row in selected:
        key = row['patient_id'], row['visit']
        name = f'{group}_{key[0]}_{key[1]}.png'
        try:
            flags = sheet(args.data, lookup[key], row, group, args.output / name)
            error = None
        except (OSError, ValueError, KeyError, IndexError) as exc:
            flags, error = ['sheet_generation_failed'], str(exc)
        sheets.append({'patient_id': key[0], 'visit': key[1], 'group': group, 'image': name if error is None else None, 'flags': flags, 'error': error})
        review_rows.append({'patient_id': key[0], 'visit': key[1], 'image': name, 'alignment': '', 'segmentation': '', 'surgery_history': '', 'reviewer': '', 'reviewed_at': '', 'decision': 'pending', 'notes': ''})
    summary = {'selection': 'single preselected seed 71 by default; volumes/boosted; hash-ordered patient-diverse cases per error group; retrospective labels used only to select review cases',
               'seed': args.seed, 'group_counts': {k:len(v) for k,v in groups.items()},
               'quality': quality, 'sheets': sheets, 'anatomical_acceptance': 'pending_human_review',
               'quality_admitted_coverage': 0.0, 'input_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cohort_path,pred_path)}}
    (args.output/'quality.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (args.output/'human_review.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(review_rows[0]))
        writer.writeheader(); writer.writerows(review_rows)
    lines=['# Retrospective assessment review', '', f'Header checks: {len(quality)} current scans. Review sheets: {len(sheets)}.',
           '', 'No automated flag is a diagnosis or anatomical approval. Human review is pending; quality-admitted coverage remains 0%.',
           'Only selected previous/current FLAIR/CT1 volumes and masks undergo voxel-value checks. Other modalities and unselected scans have header-only checks.',
           '', '## Cases', '']
    for item in sheets:
        lines.append(f"- {item['group']}: {item['patient_id']} / {item['visit']} — {item['error'] or '['+item['image']+']('+item['image']+')'}")
    lines += ['', 'Fill human_review.csv with alignment/segmentation/surgery findings, reviewer and date. Do not tune on these outcome-selected cases and call it independent validation.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Completed {len(quality)} header checks and {len(sheets)} review sheets in {args.output}',flush=True)


if __name__ == '__main__':
    main()
