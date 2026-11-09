"""Frozen current/history-only feature definition shared by training and inference."""
import numpy as np
from .report import COMPARTMENTS, measure

BASE_NAMES = ['interval_weeks', 'elapsed_weeks', 'history_count'] + [
    f'{c}_{suffix}' for c in COMPARTMENTS for suffix in
    ('log_current_ml', 'log_previous_ml', 'log_nadir_ml', 'delta_previous_ml',
     'delta_nadir_ml', 'log_change_previous', 'log_change_nadir', 'slope_ml_week')]
RATINGS = ('SD', 'PR', 'CR', 'PD')


def features(payload, variant='volumes'):
    if set(payload) - {'version', 'patient_id', 'history'}:
        raise ValueError('Model manifest accepts only version, patient_id and history; keep current labels and other data separate')
    for scan in payload.get('history', []):
        if set(scan) - {'week', 'volumes_mm3', 'rating'}:
            raise ValueError('Visit contains unsupported fields; current labels/future/clinical flags must stay separate')
    report = measure(payload)  # validates timing, surgery, withheld current label and finite volumes
    history = payload['history']
    x = [report['interval_weeks'], history[-1]['week'] - history[0]['week'], len(history) - 1]
    for c in COMPARTMENTS:
        m = report['measurements'][c]
        cur, prev, nadir = (m[k] for k in ('current_ml', 'previous_ml', 'historical_nadir_ml'))
        x.extend([np.log1p(cur), np.log1p(prev), np.log1p(nadir), cur-prev, cur-nadir,
                  np.log1p(cur)-np.log1p(prev), np.log1p(cur)-np.log1p(nadir),
                  (cur-prev)/report['interval_weeks']])
    if variant == 'previous_rating':
        rating = history[-2].get('rating')
        if rating not in RATINGS:
            raise ValueError('Optional previous-rating model requires an observed SD/PR/CR/PD at the previous visit')
        x.extend(float(rating == r) for r in RATINGS)
    elif variant != 'volumes':
        raise ValueError('Unknown feature variant')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite extracted features')
    return np.asarray(x, dtype=float)


def names(variant):
    return BASE_NAMES + ([f'previous_{r}' for r in RATINGS] if variant == 'previous_rating' else [])


def rule_scores(payload):
    m = measure(payload)['measurements']['Enhancing_Core']
    cur = m['current_ml']
    # 1 mL pseudocount is predefined, finite at zero; these are volume heuristics, not RANO.
    return {ref: float((cur + 1) / (cur + m[f'{ref}_ml'] + 2))
            for ref in ('previous', 'historical_nadir')}
