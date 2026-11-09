"""Validate a historical measurement manifest and report current-visit changes.

No model inference, future targets, clinical classification, or label lookup.
"""
import argparse
import json
import math
from pathlib import Path

COMPARTMENTS = ('Necrotic_NonEnhancing', 'Enhancing_Core', 'Edema_Compartment')


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite number')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be finite and nonnegative')
    return float(value)


def measure(payload):
    if payload.get('version') != 1:
        raise ValueError('manifest version must be 1')
    history = payload.get('history')
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError('At least two visits in one surgery-free segment are required')
    weeks, volumes = [], []
    for i, scan in enumerate(history):
        if not isinstance(scan, dict):
            raise ValueError(f'visit {i} must be an object')
        week = number(scan.get('week'), f'visit {i} week')
        if weeks and week <= weeks[-1]:
            raise ValueError('Visits must have strictly increasing weeks')
        if i and scan.get('rating', '').startswith('Post-Op'):
            raise ValueError('History crosses surgery; start a new segment at the postoperative visit')
        if scan.get('rating', '').startswith('Pre-Op'):
            raise ValueError('Preoperative scans are outside this workflow')
        if i == len(history) - 1 and scan.get('rating') not in (None, '', 'unknown'):
            raise ValueError('Current-visit rating must be withheld from assessment inputs')
        raw = scan.get('volumes_mm3')
        if not isinstance(raw, dict) or set(raw) != set(COMPARTMENTS):
            raise ValueError(f'visit {i} requires exactly the three non-overlapping compartments')
        values = {c: number(raw[c], f'visit {i} {c}') / 1000 for c in COMPARTMENTS}
        values['Total'] = sum(values.values())
        volumes.append(values)
        weeks.append(week)
    changes = {}
    for name, current in volumes[-1].items():
        previous = volumes[-2][name]
        nadir = min(v[name] for v in volumes[:-1])
        changes[name] = {
            'current_ml': current, 'previous_ml': previous, 'historical_nadir_ml': nadir,
            'delta_previous_ml': current - previous,
            'change_previous_percent': 100 * (current / previous - 1) if previous > 0 else None,
            'change_nadir_percent': 100 * (current / nadir - 1) if nadir > 0 else None,
        }
    warnings = ['Automated measurements require anatomical and segmentation quality review.',
                'Physical scan alignment and segmentation accuracy have not been checked by this command.']
    if any(x['change_previous_percent'] is None or x['change_nadir_percent'] is None for x in changes.values()):
        warnings.append('Some relative changes are undefined because reference volume is zero.')
    return {
        'schema_version': 1, 'task': 'current_visit_measurement',
        'patient_id': payload.get('patient_id'), 'assessed_week': weeks[-1],
        'interval_weeks': weeks[-1] - weeks[-2], 'history_visits': len(history),
        'measurements': changes,
        'assessment': {'status': 'insufficient_evidence', 'probability_pd': None,
                       'reason': 'Measurement-only baseline; no validated assessment model is attached.'},
        'quality': {'input_checks': 'passed', 'anatomical_review': 'not_performed'},
        'warnings': warnings,
    }


def markdown(report):
    lines = ['# Current-visit tumor measurements', '',
             f"Visit: week {report['assessed_week']:g}; interval: {report['interval_weeks']:g} weeks.", '',
             '| Compartment | Current mL | Previous mL | Change mL |',
             '| --- | ---: | ---: | ---: |']
    for name, x in report['measurements'].items():
        lines.append(f"| {name} | {x['current_ml']:.3f} | {x['previous_ml']:.3f} | {x['delta_previous_ml']:+.3f} |")
    lines += ['', 'Assessment: insufficient evidence. No assessment model is attached.', '']
    lines += [f'- {warning}' for warning in report['warnings']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    try:
        raw = args.history.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError('manifest must be an object')
        result = measure(payload)
        import hashlib
        result['input_sha256'] = hashlib.sha256(raw).hexdigest()
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (args.output / 'report.md').write_text(markdown(result))
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')
    print(f'Wrote measurement report to {args.output}')
