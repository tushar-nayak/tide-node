"""Run frozen nested, patient-disjoint internal assessment benchmark (CPU only)."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import pickle
import platform
import shutil
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .cohort import build, sha
from .features import names
from .model import GRID, fit, predict, splits

SEEDS = [71, 1729, 2026]
METHODS = ['enhancing_previous', 'enhancing_nadir', 'logistic', 'boosted']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/lumiere/extracted/Imaging'))
    parser.add_argument('--ratings', type=Path, default=Path('data/lumiere/clinical/LUMIERE-ExpertRating-v202211.csv'))
    parser.add_argument('--results', type=Path, default=Path('results/assessment_v1'))
    parser.add_argument('--run', type=Path, default=Path('runs/assessment_v1'))
    args = parser.parse_args()
    out, run = args.results, args.run
    out.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    if (out/'freeze.json').exists():
        raise RuntimeError('Frozen run already exists. Use fresh result/run paths; never overwrite a frozen evaluation.')
    rows, excluded, sources = build(args.data, args.ratings)
    write(out/'cohort.json', rows)
    write(out/'exclusions.json', excluded)
    write(out/'input_manifest.json', sources)
    config = dict(version='assessment_v1', seeds=SEEDS, outer_folds=5, inner_folds=3,
                  grid=GRID, feature_names={v:names(v) for v in ['volumes','previous_rating']},
                  ablation='Volumes on previous-rating-available cohort uses identical outer/inner patient splits and training visits to optional-rating models',
                  target='current expert PD versus SD/PR/CR; postoperative visits excluded',
                  rule='enhancing (current+1mL)/(current+reference+2mL); fixed 25% increase separately reported',
                  preprocessing='log1p mL and signed changes; StandardScaler fitted inside logistic pipeline only',
                  calibration='Platt logistic C=1 on selected candidate patient inner OOF logit scores; no class weighting',
                  selection='minimum pooled inner OOF raw Brier; fixed grid order breaks ties',
                  thresholds='inner OOF calibrated probabilities: max low threshold sensitivity>=.90; min high threshold specificity>=.90; ordered min/max; binary max balanced accuracy, smallest tie',
                  calibration_limitation='Calibration and bands fitted on same training OOF labels after candidate selection; only outer evaluation estimates their achieved performance',
                  bootstrap=dict(unit='whole patient, pooled across all predefined repeats', resamples=2000, seed=90210,
                                 intervals='percentile 2.5/97.5; one-class samples omitted for ranking metrics'),
                  quality='structural eligibility only; anatomical alignment/segmentation UNVERIFIED; clinical-admitted coverage zero',
                  duplicate_ratings='identical ratings collapsed; conflicting targets excluded; any surgery resets history even with ambiguous label',
                  discovery='highest acquisition suffix per week regardless of modalities; boundaries reset before missing-volume exclusion',
                  exclusions='preop, surgery assessments, missing invalid volumes, missing current expert label, no prior valid visit in segment',
                  prohibited='no future scans/labels, survival, clinical flag columns, mask margins, registration caches',
                  refit='all-cohort logistic volumes; same fixed inner recipe with seed71; no independent performance claim')
    write(out/'config.json', config)
    # Every planned outer and inner split is checked, materialized and hashed BEFORE any estimator fits.
    plans = {}
    for variant in ['volumes', 'previous_rating']:
        selected = [r for r in rows if variant == 'volumes' or r['previous_available']]
        y = np.array([r['target'] for r in selected]); groups = np.array([r['patient_id'] for r in selected])
        plans[variant] = []
        for seed in SEEDS:
            for fold, (train, test) in enumerate(splits(y, groups, 5, seed)):
                inner = splits(y[train], groups[train], 3, seed+fold+100)
                plans[variant].append(dict(seed=seed, fold=fold, train=train.tolist(), test=test.tolist(),
                    inner=[dict(train=a.tolist(), validation=b.tolist()) for a,b in inner],
                    train_patients=sorted(set(groups[train])), test_patients=sorted(set(groups[test]))))
    plans['volumes_matched_previous'] = plans['previous_rating']
    # Also validate refit split before fitting.
    refit_inner = splits(np.array([r['target'] for r in rows]), np.array([r['patient_id'] for r in rows]), 3, 71)
    plans['refit'] = [dict(train=a.tolist(), validation=b.tolist()) for a,b in refit_inner]
    write(out/'splits.json', plans)
    write(run/'environment.json', dict(python=sys.version, platform=platform.platform(),
          git_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          thread_limits={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
    (run/'requirements.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    source_files = sorted(Path('assessment').glob('*.py')) + [Path('scripts/lumiere_weeks.py')]
    source_hashes = {str(p):sha(p) for p in source_files}
    for p in source_files:
        destination = run/'frozen_source'/p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    write(out/'freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(out/'config.json'), cohort_sha256=sha(out/'cohort.json'),
         inputs_sha256=sha(out/'input_manifest.json'), splits_sha256=sha(out/'splits.json'),
         ratings_file_sha256=sha(args.ratings), source_sha256=source_hashes,
         scope='All available eligible LUMIERE patients, internally evaluated; no untouched independent cohort'))
    print(f'FROZEN {len(rows)} assessments / {len(set(r["patient_id"] for r in rows))} patients; {len(excluded)} exclusions', flush=True)
    predictions = []
    for variant in ['volumes', 'volumes_matched_previous', 'previous_rating']:
        feature_variant = 'previous_rating' if variant == 'previous_rating' else 'volumes'
        selected = [r for r in rows if variant == 'volumes' or r['previous_available']]
        methods = METHODS + (['previous_rating_rule'] if variant == 'previous_rating' else [])
        for plan in plans[variant]:
            train = [selected[i] for i in plan['train']]; test = [selected[i] for i in plan['test']]
            inner = [(np.array(f['train']), np.array(f['validation'])) for f in plan['inner']]
            for method in methods:
                bundle, audit = fit(train, feature_variant, method, plan['seed']+plan['fold']+100, inner)
                raw, prob, bands = predict(bundle, [r['payload'] for r in test])
                tag = f'{variant}_{method}_seed{plan["seed"]}_fold{plan["fold"]}'
                write(run/f'{tag}_training.json', audit)
                for i,r in enumerate(test):
                    fixed = None
                    if method.startswith('enhancing_'):
                        ref = 'previous' if method == 'enhancing_previous' else 'historical_nadir'
                        from .report import measure
                        m = measure(r['payload'])['measurements']['Enhancing_Core']
                        fixed = int(m['current_ml'] >= 1.25*m[f'{ref}_ml'] and m['current_ml']-m[f'{ref}_ml'] >= 1.)
                    elif method == 'previous_rating_rule':
                        fixed = int(raw[i] >= .5)
                    predictions.append(dict(patient_id=r['patient_id'], visit=r['visit'], week=r['week'],
                        segment=r['segment'], target=r['target'], previous_rating=r['previous_rating'],
                        previous_available=r['previous_available'], new_progression_eligible=r['new_progression_eligible'],
                        variant=variant, method=method, seed=plan['seed'], fold=plan['fold'],
                        raw_probability=float(raw[i]), probability=float(prob[i]), band=bands[i],
                        binary_prediction=int(prob[i] >= bundle['thresholds']['binary']),
                        low_threshold=bundle['thresholds']['low'], high_threshold=bundle['thresholds']['high'],
                        binary_threshold=bundle['thresholds']['binary'], fixed_rule_prediction=fixed,
                        quality_status='anatomical_unverified', assessment_status='insufficient_evidence',
                        quality_admitted=False, training_audit=f'{tag}_training.json'))
                print(f'Completed {tag}', flush=True)
                pd.DataFrame(predictions).to_csv(out/'outer_predictions.csv', index=False)
    # Frozen all-cohort refit, explicitly separate from evaluation weights; no test use for selection.
    bundle, audit = fit(rows, 'volumes', 'logistic', 71, refit_inner)
    bundle['freeze_sha256'] = sha(out/'freeze.json')
    bundle['training_scope'] = 'all-cohort refit; no independent performance estimate'
    with (run/'all_cohort_logistic.pkl').open('wb') as f:
        pickle.dump(bundle, f)
    write(run/'all_cohort_refit_training.json', audit)
    smoke = rows[0]['payload']
    write(run/'real_history_no_current_label.json', smoke)
    from .summarize import summarize
    summarize(out)
    # Checksums include generated plots, predictions, frozen definitions, environment and local bundle.
    write(run/'checksums.json', {str(p):sha(p) for root in [out,run] for p in sorted(root.rglob('*'))
                                  if p.is_file() and p.name not in ['checksums.json','execution.log']})
    print('Evaluation, bootstrap, plots and all-cohort refit complete.', flush=True)

if __name__ == '__main__':
    main()
