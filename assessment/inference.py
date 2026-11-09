"""Offline scoring with the exact frozen evaluation features; no label lookup."""
import hashlib
import pickle
from pathlib import Path
from .features import features, names
from .model import predict
from .report import measure


def load_bundle(path, trusted=False):
    if not trusted:
        raise ValueError('Pickle may execute code: explicitly pass --trust-model only for a trusted local bundle')
    with Path(path).open('rb') as f:
        bundle = pickle.load(f)
    if bundle.get('schema_version') != 1 or not bundle.get('research_only'):
        raise ValueError('Unsupported assessment research bundle')
    if bundle.get('feature_names') != names(bundle['variant']):
        raise ValueError('Bundle feature definition mismatch')
    return bundle


def assess(payload, bundle=None):
    try:
        report = measure(payload)
        features(payload, bundle['variant'] if bundle else 'volumes')
    except (ValueError, TypeError, KeyError) as error:
        return dict(schema_version=1,task='current_visit_assessment',patient_id=payload.get('patient_id'),
                    assessed_week=None,interval_weeks=None,measurements={},
                    assessment=dict(status='insufficient_evidence',probability_pd=None,
                                    reason=str(error),research_band='defer'),
                    quality=dict(input_checks='failed', anatomical_review='unverified'),
                    missingness=[str(error)],warnings=['No usable assessment: input validation failed.'])
    report['task'] = 'current_visit_assessment'
    report['missingness'] = []
    report['quality']['anatomical_review'] = 'unverified'
    if bundle is not None:
        raw, probability, bands = predict(bundle, [payload])
        report['assessment'] = dict(status='insufficient_evidence',
            probability_pd=float(probability[0]),raw_probability_pd=float(raw[0]),
            research_band=str(bands[0]), research_binary_prediction=int(probability[0] >= bundle['thresholds']['binary']),
            thresholds=bundle['thresholds'],
            reason='Anatomical alignment and segmentation review unverified; research estimate of expert rating only, not a diagnosis.',
            statistical_deferral=bool(bands[0] == 'defer'), quality_deferral=True)
        report['model'] = dict(method=bundle['method'],variant=bundle['variant'],
            freeze_sha256=bundle['freeze_sha256'],training_scope=bundle['training_scope'])
        report['warnings'].append('All-cohort research refit. No independent validation or clinical readiness claim.')
    return report
