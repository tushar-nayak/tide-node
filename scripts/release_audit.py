"""Read-only artifact audit. No training. See RELEASE_AUDIT.md for interpretation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import build_cohort, patient_folds, COMPARTMENT_SHORT


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_rows(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in out, f'duplicate case: {key}')
        out[key] = row
    require(bool(out), 'empty prediction table')
    return out


def finite(row, names):
    values = np.array([float(row[n]) for n in names])
    require(np.isfinite(values).all(), f'nonfinite metric: {row.get("patient_id")} {names}')
    return values


def paired_effect(left, right, keys, columns, n_boot=20000, seed=20261103):
    """Average paired metric differences within patient, resample patients jointly.

    Repeated predictions are averaged as losses, not ensemble predictions. Complete
    case/repeat coverage is checked by callers; no intersection or NaN dropping.
    """
    a, b = index_rows(left, keys), index_rows(right, keys)
    require(a.keys() == b.keys(), f'unpaired cases: left-only={len(a.keys()-b.keys())}, right-only={len(b.keys()-a.keys())}')
    values = defaultdict(list)
    for k in sorted(a):
        values[a[k]['patient_id']].append(float((finite(a[k], columns) - finite(b[k], columns)).mean()))
    patients = sorted(values)
    d = np.array([np.mean(values[p]) for p in patients])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {'patients': len(d), 'paired_rows': len(a), 'difference': float(d.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'positive_patients': int((d > 0).sum()), 'zero_patients': int((d == 0).sum()),
            'negative_patients': int((d < 0).sum())}


def validate_folds(rows, patients, n_folds, repeats, expected):
    require({int(r['repeat']) for r in rows} == set(range(repeats)), 'missing/unexpected repeats')
    for repeat in range(repeats):
        rr = [r for r in rows if int(r['repeat']) == repeat]
        require({int(r['fold']) for r in rr} == set(range(n_folds)), 'missing/unexpected folds')
        assignment = {p: f for f, ps in enumerate(patient_folds(patients, n_folds, repeat)) for p in ps}
        idx = index_rows(rr, ['patient_id', 'last_source', 'target_source'])
        require(set(idx) == set(expected), f'cohort coverage mismatch repeat {repeat}: missing={len(set(expected)-set(idx))}, extra={len(set(idx)-set(expected))}')
        for r in rr:
            require(assignment.get(r['patient_id']) == int(r['fold']), f'patient fold mismatch: {r["patient_id"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/release_audit'))
    p.add_argument('--bootstrap', type=int, default=20000)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--smoke', action='store_true', help='CPU inference from four saved fold-1 best checkpoints')
    args = p.parse_args()
    require(args.bootstrap > 0, 'bootstrap must be positive')
    source = args.source.resolve()
    require(not args.output.resolve().is_relative_to(source), 'output must be outside read-only source')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'source': str(source), 'seed': args.seed, 'bootstrap': args.bootstrap,
              'effects': {}, 'checks': {}, 'artifacts': {}, 'checkpoints': [], 'smoke': []}
    def record(path):
        report['artifacts'][str(path.relative_to(source))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    cohort = build_cohort(source/'data/lumiere/extracted/Imaging', source/'data/lumiere/clinical/LUMIERE-ExpertRating-v202211.csv', require_seg_mask=True)
    expected = {(t.patient_id, t.last.source_name, t.target.source_name): t for t in cohort.transitions}
    require(bool(expected), 'source cohort empty')
    report['cohort'] = {'patients': len(cohort.patients), 'transitions': len(expected), 'exclusions': cohort.excluded}
    for path in sorted((source/'scripts').glob('*.py')):
        record(path)
    record(source/'data/lumiere/clinical/LUMIERE-ExpertRating-v202211.csv')
    # Final volume results live in results/, while the quick run lives in runs/.
    for label, relative in [('volume', 'results/volume_trajectory'), ('volume_quick', 'runs/volume_trajectory_quick')]:
        path = source/relative/'predictions.csv'
        record(path)
        meta_path = path.with_name('metrics.json')
        record(meta_path)
        meta = json.loads(meta_path.read_text())
        rows = read_csv(path)
        lookup = {(t.patient_id, str(t.last.week), t.target.source_name): t for t in cohort.transitions}
        for r in rows:
            t = lookup[(r['patient_id'], r['last_week'], r['target_source'])]
            r['last_source'] = t.last.source_name
            require(int(r['target_week']) == t.target.week and int(r['dt_weeks']) == t.dt_weeks and int(r['n_history']) == len(t.history), 'volume transition metadata mismatch')
            for prefix, scan in [('true', t.target), ('persistence', t.last)]:
                truth = np.log1p(np.array([scan.volumes_mm3[c] for c in ('Necrotic_NonEnhancing', 'Enhancing_Core', 'Edema_Compartment')])/1000)
                require(np.allclose(finite(r, [f'{prefix}_{c}' for c in COMPARTMENT_SHORT]), truth, atol=1e-10, rtol=0), 'volume reference mismatch')
            for method in ['neural_ode', 'ridge', 'gbm', 'gompertz', 'loglinear', 'persistence']:
                finite(r, [f'{method}_{c}' for c in (*COMPARTMENT_SHORT, 'total')])
            finite(r, ['true_total'])
        validate_folds(rows, cohort.patients, meta['protocol']['folds'], meta['protocol']['repeats'], expected)
        keys = ['repeat', 'patient_id', 'last_source', 'target_source']
        def errors(method):
            return [{**r, 'loss': float(np.mean([abs(float(r[f'{method}_{c}'])-float(r[f'true_{c}'])) for c in COMPARTMENT_SHORT]))} for r in rows]
        for method in ['ridge', 'gbm', 'gompertz', 'persistence']:
            report['effects'][f'{label}: neural_ode - {method} MAE'] = paired_effect(errors('neural_ode'), errors(method), keys, ['loss'], args.bootstrap, args.seed)
        report['checks'][label] = {'rows': len(rows), 'protocol': meta['protocol'], 'duplicates': 0, 'missing_cases': 0, 'nonfinite_predictions': 0}
    import torch
    from mask_forecast_experiment import fold_split, MaskForecastNet, MaskForecastDataset, transition_metrics
    from register_step2_pairs import needed_pairs
    torch.set_num_threads(2)
    # Reconstruct disjoint fit/validation/test memberships, persist exact manifest.
    splits = []
    for f in range(5):
        parts = [sorted({t.patient_id for t in ts}) for ts in fold_split(cohort.transitions, 5, f)]
        require(all(not (set(parts[i]) & set(parts[j])) for i in range(3) for j in range(i)), 'split leakage')
        splits.append(dict(fold=f, fit=parts[0], validation=parts[1], test=parts[2]))
    report['mask_splits'] = splits
    pairs = needed_pairs(cohort)
    methods = Counter()
    for pid, mov, fix, role in pairs:
        stem = source/'data/lumiere_step2_aligned'/pid/f'{mov.source_name}__in__{fix.source_name}'
        for suffix in ('.npy', '.h5', '.json'):
            require(stem.with_suffix(suffix).exists(), f'missing aligned artifact: {stem}{suffix}')
        m = json.loads(stem.with_suffix('.json').read_text())
        require((m['patient_id'], m['moving_source'], m['fixed_source']) == (pid, mov.source_name, fix.source_name), 'registration reference mismatch')
        finite(m['qc'], ['ncc_T1_atlas', 'ncc_T1_registered', 'brain_dice_atlas', 'brain_dice_registered'])
        methods[m['qc']['method']] += 1
        record(stem.with_suffix('.json'))
        record(stem.with_suffix('.h5'))
    report['registration'] = {'pairs': len(pairs), 'methods': dict(methods)}
    for frame, directory in [('atlas', 'mask_forecast'), ('registered', 'mask_forecast_registered')]:
        variants = {}
        for variant in ['ode', 'static']:
            rows = []
            for f in range(5):
                directory_path = source/'runs'/directory/variant/f'fold{f}'
                path = directory_path/'test_predictions.csv'
                require(path.exists(), f'missing fold: {path}')
                record(path)
                rr = read_csv(path)
                for r in rr:
                    r.update(fold=str(f), repeat='0')
                    finite(r, [k for k in r if k.startswith(('model_', 'persistence_'))])
                rows.extend(rr)
                ckpt = directory_path/'best.pt'
                require(ckpt.exists(), f'missing checkpoint: {ckpt}')
                record(ckpt)
                if (directory_path/'final.pt').exists():
                    record(directory_path/'final.pt')
                log = directory_path/'train_log.csv'
                record(log)
                logs = np.loadtxt(log, delimiter=',', ndmin=2)
                require(np.isfinite(logs).all() and len(set(logs[:, 0])) == len(logs), 'nonfinite/duplicate training log')
                state = torch.load(ckpt, map_location='cpu', weights_only=True)
                zero_heads = all(torch.count_nonzero(state[k]).item() == 0 for k in ['flow.weight', 'flow.bias', 'resid.weight', 'resid.bias'])
                persistence_delta = max(abs(float(r[k])-float(r[k.replace('model_', 'persistence_', 1)])) for r in rr for k in r if k.startswith('model_') and not k.startswith('model_true_'))
                require(not zero_heads or persistence_delta < 1e-9, 'zero-head checkpoint disagrees with persistence CSV')
                report['checkpoints'].append({'max_model_persistence_difference': persistence_delta, 'frame': frame, 'variant': variant, 'fold': f, 'zero_output_heads': zero_heads, 'logged_epochs': len(logs), 'best_logged_epoch': int(logs[np.argmax(logs[:, 2]), 0]), 'best_logged_validation_dice_rounded': float(logs[:, 2].max()), 'final_exists': (directory_path/'final.pt').exists()})
                if args.smoke and f == 1:
                    t = expected[(rr[0]['patient_id'], rr[0]['last_source'], rr[0]['target_source'])]
                    ds = MaskForecastDataset([t], source/'data/lumiere_atlas2mm', False, frame, source/'data/lumiere_step2_aligned')
                    batch = ds[0]
                    base = state['enc.0.0.weight'].shape[0]
                    model = MaskForecastNet(12, base, variant).eval()
                    model.load_state_dict(state)
                    with torch.inference_mode():
                        logits, _ = model(*(batch[k].unsqueeze(0) for k in ['x', 'last_mask', 'dt']))
                    metrics = transition_metrics(logits[0].numpy() > 0, batch['last_mask'].numpy() > .5, batch['y'].numpy() > .5)
                    delta = max(abs(v-float(rr[0]['model_'+k])) for k, v in metrics.items())
                    report['smoke'].append({'frame': frame, 'variant': variant, 'patient': t.patient_id, 'target': t.target.source_name, 'metric_differences_cpu_minus_csv': {k: v-float(rr[0]['model_'+k]) for k, v in metrics.items()}, 'max_absolute_metric_difference': delta, 'exact_within_1e-6': delta < 1e-6})
            validate_folds(rows, cohort.patients, 5, 1, expected)
            variants[variant] = rows
        a = index_rows(variants['ode'], ['patient_id', 'last_source', 'target_source'])
        b = index_rows(variants['static'], ['patient_id', 'last_source', 'target_source'])
        require(a.keys() == b.keys(), 'variant case mismatch')
        for key, r in a.items():
            t = expected[key]
            require(int(r['dt_weeks']) == t.dt_weeks and int(r['n_history']) == len(t.history), 'mask transition metadata mismatch')
            for col in ['dt_weeks', 'n_history', 'target_rating']:
                require(r[col] == b[key][col], f'paired metadata mismatch {key} {col}')
            cols = [k for k in r if k.startswith(('persistence_', 'model_true_'))]
            require(np.allclose(finite(r, cols), finite(b[key], cols), atol=1e-10, rtol=0), f'paired reference mismatch {key}')
        for metric in ['dice', 'change_dice', 'logvol_err']:
            for region in ['mean', 'wt', 'tc', 'et']:
                cols = [f'model_{metric}_{r}' for r in (['wt', 'tc', 'et'] if region == 'mean' else [region])]
                report['effects'][f'{frame}: ode - static {metric}_{region}'] = paired_effect(variants['ode'], variants['static'], ['patient_id', 'last_source', 'target_source'], cols, args.bootstrap, args.seed)
        report['checks'][frame] = {'rows_per_variant': len(a), 'missing_folds': 0, 'duplicates': 0, 'reference_mismatches': 0, 'nonfinite_metrics': 0}
    report['status'] = 'table_checks_passed; exact_CPU_prediction_reproduction_failed' if any(not x['exact_within_1e-6'] for x in report['smoke']) else 'table_checks_passed; provenance_not_fully_verified'
    report['versions'] = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Directly paired patient effects', '', 'Difference = ODE minus comparator. Negative favors ODE for MAE/log-volume error; positive favors ODE for Dice. Equal patient weights; percentile patient bootstrap; repeated volume losses averaged within patient, not independent replicates.', '', f'Bootstrap draws: {args.bootstrap}; seed: {args.seed}. CIs are descriptive, unadjusted for multiple comparisons and conditional on saved fitted models.', '', '| Comparison | Patients | Rows | Difference | 95% CI |', '|---|---:|---:|---:|---:|']
    for name, e in report['effects'].items():
        lines.append(f"| {name} | {e['patients']} | {e['paired_rows']} | {e['difference']:+.6f} | [{e['ci95'][0]:+.6f}, {e['ci95'][1]:+.6f}] |")
    (args.output/'effects.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'checks': report['checks'], 'registration': report['registration'], 'smoke': report['smoke']}, indent=2))


if __name__ == '__main__':
    main()
