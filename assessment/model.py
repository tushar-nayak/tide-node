"""Training-only patient OOF selection, Platt calibration and decision recipes."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, balanced_accuracy_score
from .features import features, rule_scores, names

GRID = {'logistic': [0.1, 1.0, 10.0], 'boosted': [1.0, 10.0],
        'enhancing_previous': [None], 'enhancing_nadir': [None], 'previous_rating_rule': [None]}


def splits(y, groups, n, seed):
    result = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Patient leakage')
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError(f'Both classes required in every split (seed={seed}, folds={n})')
    return result


def estimator(method, parameter):
    if method == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=parameter, max_iter=2000, random_state=71))
    if method == 'boosted':
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
                    learning_rate=0.05, l2_regularization=parameter, early_stopping=False, random_state=71)
    return None


def raw_predict(model, method, x, payloads):
    if model is not None:
        return model.predict_proba(x)[:, 1]
    if method == 'previous_rating_rule':
        return np.array([float(p['history'][-2]['rating'] == 'PD') for p in payloads])
    key = 'previous' if method == 'enhancing_previous' else 'historical_nadir'
    return np.array([rule_scores(p)[key] for p in payloads])


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def thresholds(y, p):
    candidates = np.unique(np.r_[0., p, np.nextafter(p, 2.), 1.0000001])
    sensitivity = np.array([np.mean(p[y == 1] >= t) for t in candidates])
    specificity = np.array([np.mean(p[y == 0] < t) for t in candidates])
    # Sensitivity floor for low-risk exclusion, specificity floor for high-risk inclusion.
    low = float(candidates[sensitivity >= .9].max())
    high = float(candidates[specificity >= .9].min())
    low, high = min(low, high), max(low, high)
    bal = (sensitivity + specificity)/2
    binary = float(candidates[np.argmax(bal)])  # deterministic smallest threshold for ties
    return dict(low=low, high=high, binary=binary,
                training_sensitivity=float(np.mean(p[y == 1] >= binary)),
                training_specificity=float(np.mean(p[y == 0] < binary)))


def band(p, t):
    return np.where(p < t['low'], 'lower', np.where(p >= t['high'], 'higher', 'defer'))


def fit(rows, variant, method, seed, inner_splits=None):
    x = np.stack([features(r['payload'], variant) for r in rows])
    y = np.array([r['target'] for r in rows])
    groups = np.array([r['patient_id'] for r in rows])
    folds = inner_splits if inner_splits is not None else splits(y, groups, 3, seed)
    candidates = []
    for parameter in GRID[method]:
        oof = np.full(len(y), np.nan)
        for train, test in folds:
            model = estimator(method, parameter)
            if model is not None:
                model.fit(x[train], y[train])
            oof[test] = raw_predict(model, method, x[test], [rows[i]['payload'] for i in test])
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete inner OOF')
        candidates.append((brier_score_loss(y, oof), parameter, oof))
    score, parameter, oof = min(candidates, key=lambda c: c[0])
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=71).fit(logit(oof), y)
    calibrated = calibrator.predict_proba(logit(oof))[:, 1]
    t = thresholds(y, calibrated)
    model = estimator(method, parameter)
    if model is not None:
        model.fit(x, y)
    bundle = dict(feature_names=names(variant), schema_version=1, variant=variant, method=method, parameter=parameter,
                  model=model, calibrator=calibrator, thresholds=t,
                  inner_raw_brier=float(score), research_only=True,
                  anatomical_quality='unverified')
    audit = dict(training_patients=sorted(set(groups)), thresholds=t, selected_parameter=parameter,
                 candidate_brier=[dict(parameter=p, brier=float(s)) for s,p,_ in candidates],
                 inner_folds=[dict(train_patients=sorted(set(groups[a])),
                                   validation_patients=sorted(set(groups[b]))) for a,b in folds],
                 inner_oof=[dict(patient_id=r['patient_id'], visit=r['visit'], target=int(y[i]),
                                 raw=float(oof[i]), calibrated=float(calibrated[i])) for i,r in enumerate(rows)])
    return bundle, audit


def predict(bundle, payloads):
    x = np.stack([features(p, bundle['variant']) for p in payloads])
    raw = raw_predict(bundle['model'], bundle['method'], x, payloads)
    calibrated = bundle['calibrator'].predict_proba(logit(raw))[:, 1]
    return raw, calibrated, band(calibrated, bundle['thresholds'])
