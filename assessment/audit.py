"""Read-only verification of completed assessment artifacts, splits and training cutoffs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .cohort import sha
from .features import features
from .model import thresholds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/assessment_v1'))
    p.add_argument('--run',type=Path,default=Path('runs/assessment_v1'))
    args=p.parse_args();out=args.results;run=args.run
    freeze=json.loads((out/'freeze.json').read_text())
    for key,filename in [('config_sha256','config.json'),('cohort_sha256','cohort.json'),
                         ('inputs_sha256','input_manifest.json'),('splits_sha256','splits.json')]:
        assert sha(out/filename)==freeze[key],filename
    for path,expected in freeze['source_sha256'].items():
        assert sha(path)==expected,path
        assert sha(run/'frozen_source'/path)==expected,path
    source=json.loads((out/'input_manifest.json').read_text())
    for item in source:
        if item['measurement_sha256']:
            assert sha(item['measurement_path'])==item['measurement_sha256']
    rows=json.loads((out/'cohort.json').read_text())
    plans=json.loads((out/'splits.json').read_text())
    for row in rows:
        assert row['payload']['history'][-1].get('rating') is None
        assert row['payload']['history'][-1]['week']==row['week']
        assert all(h['week']<=row['week'] for h in row['payload']['history'])
        features(row['payload'])
    frame=pd.read_csv(out/'outer_predictions.csv')
    assert frame[['patient_id','visit','variant','method','seed']].duplicated().sum()==0
    assert frame.groupby(['patient_id','visit','variant','method']).size().eq(3).all()
    assert len(frame)==3*(371*4+252*9),len(frame)
    for variant in frame.variant.unique():
        selected=[r for r in rows if variant=='volumes' or r['previous_available']]
        for plan in plans[variant]:
            train=[selected[i] for i in plan['train']];test=[selected[i] for i in plan['test']]
            train_patients={r['patient_id'] for r in train};test_patients={r['patient_id'] for r in test}
            assert not train_patients & test_patients
            subset=frame[(frame.variant==variant)&(frame.seed==plan['seed'])&(frame.fold==plan['fold'])]
            assert set(subset.patient_id)==test_patients
            for method,pred in subset.groupby('method'):
                audit=json.loads((run/pred.training_audit.iloc[0]).read_text())
                assert set(audit['training_patients'])==train_patients
                assert set(zip(pred.patient_id,pred.visit))=={(r['patient_id'],r['visit']) for r in test}
                for f in audit['inner_folds']:
                    assert not set(f['train_patients']) & set(f['validation_patients'])
                    assert set(f['train_patients']) | set(f['validation_patients'])==train_patients
                oo=audit['inner_oof']
                t=thresholds(np.array([a['target'] for a in oo]),np.array([a['calibrated'] for a in oo]))
                assert t==audit['thresholds']
                for key in ['low','high','binary']:
                    np.testing.assert_allclose(pred[f'{key}_threshold'],t[key],rtol=1e-14,atol=1e-14)
                np.testing.assert_array_equal(pred.binary_prediction,pred.probability>=t['binary'])
    assert not frame.quality_admitted.any()
    assert np.isfinite(frame[['probability','raw_probability']]).all().all()
    checksums=json.loads((run/'checksums.json').read_text())
    for path,expected in checksums.items():
        assert sha(path)==expected,path
    print(f'PASS: {len(rows)} histories, {len(source)} input records, {len(frame)} outer predictions; frozen hashes, source snapshots, patient separation, inner-derived thresholds, repeat coverage and artifact checksums verified.')

if __name__=='__main__':main()
