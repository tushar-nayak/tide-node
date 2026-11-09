import copy
import json
import pickle
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from assessment.features import features, BASE_NAMES
from assessment.model import fit, predict, splits, thresholds
from assessment.inference import assess, load_bundle
from assessment.cohort import build


class AssessmentV1Tests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(Path('assessment/example.json').read_text())

    def test_zero_features_and_timing(self):
        p = copy.deepcopy(self.payload)
        for s in p['history']:
            s['volumes_mm3'] = dict.fromkeys(s['volumes_mm3'], 0)
        self.assertTrue(np.isfinite(features(p)).all())
        self.assertEqual(len(features(p)), len(BASE_NAMES))
        p['history'][-1]['week'] = 0
        with self.assertRaises(ValueError): features(p)
        p = copy.deepcopy(self.payload); p['history'][-1]['rating'] = 'PD'
        with self.assertRaises(ValueError): features(p)
        p = copy.deepcopy(self.payload); p['target'] = 1
        with self.assertRaises(ValueError): features(p)

    def test_none_rating_is_missing_and_future_append_does_not_change_prior_features(self):
        p = copy.deepcopy(self.payload)
        p['history'][-1]['rating'] = None
        self.assertTrue(np.isfinite(features(p)).all())
        original = features(p).copy()
        future = copy.deepcopy(p['history'][-1])
        future['week'] += 12
        future['rating'] = 'PD'
        p['history'].append(future)
        np.testing.assert_array_equal(features(dict(p, history=p['history'][:-1])), original)
        with self.assertRaises(ValueError): features(p)

    def test_optional_missingness_and_deferral(self):
        p = copy.deepcopy(self.payload)
        with self.assertRaises(ValueError): features(p, 'previous_rating')
        del p['history'][-1]['volumes_mm3']['Enhancing_Core']
        result = assess(p)
        self.assertEqual(result['assessment']['status'], 'insufficient_evidence')
        self.assertEqual(result['quality']['input_checks'], 'failed')
        self.assertTrue(result['missingness'])

    def test_split_separation_and_class_fail(self):
        y=np.tile([0,1],15); g=np.repeat(np.arange(15),2)
        for train,test in splits(y,g,5,71):
            self.assertFalse(set(g[train]) & set(g[test]))
        with self.assertRaises(ValueError): splits(np.zeros(30),g,5,71)

    def test_threshold_training_recipe(self):
        y=np.array([0]*10+[1]*10); p=np.linspace(.01,.99,20)
        t=thresholds(y,p)
        self.assertGreaterEqual(t['training_sensitivity'],.9)
        self.assertGreaterEqual(t['training_specificity'],.9)
        self.assertLessEqual(t['low'],t['high'])
        changed_test_labels=np.array([1,0])  # outer labels never appear in API
        self.assertEqual(t,thresholds(y,p))
        self.assertEqual(t['binary'],float(np.nextafter(p[9],2.)))

    def test_nested_serialization_and_inference_equivalence(self):
        rows=[]
        for patient in range(18):
            for target in [0,1]:
                p=copy.deepcopy(self.payload); p['patient_id']=str(patient)
                p['history'][-1]['volumes_mm3']['Enhancing_Core']=1000+target*2000+patient*10
                rows.append(dict(patient_id=str(patient),visit=str(target),target=target,payload=p))
        bundle,audit=fit(rows,'volumes','logistic',71)
        bundle.update(freeze_sha256='test', training_scope='test all-cohort refit')
        raw,prob,bands=predict(bundle,[self.payload])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle.pkl';path.write_bytes(pickle.dumps(bundle))
            with self.assertRaises(ValueError):load_bundle(path)
            loaded=load_bundle(path,True)
            result=assess(self.payload,loaded)
            self.assertAlmostEqual(result['assessment']['probability_pd'],prob[0],places=14)
            self.assertEqual(result['assessment']['research_band'],bands[0])
            self.assertTrue(result['assessment']['quality_deferral'])
            history_path=Path(tmp)/'history.json'
            history_path.write_text(json.dumps(self.payload))
            output=Path(tmp)/'cli'
            subprocess.run([sys.executable,'-m','assessment','--history',str(history_path),
                            '--model',str(path),'--trust-model','--output',str(output)],check=True,capture_output=True)
            cli=json.loads((output/'report.json').read_text())
            self.assertEqual(cli['assessment'],result['assessment'])
        for f in audit['inner_folds']:
            self.assertFalse(set(f['train_patients']) & set(f['validation_patients']))

    def test_missing_surgery_volume_still_resets_history_and_conflicts_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); patient=root/'Patient-001';patient.mkdir()
            for i in range(5):
                visit=patient/f'week-{i:03d}'
                location=visit/'DeepBraTumIA-segmentation/atlas/segmentation'
                location.mkdir(parents=True)
                if i != 2:
                    (location/'measured_volumes_in_mm3.json').write_text(json.dumps(self.payload['history'][0]['volumes_mm3']))
            ratings=root/'ratings.csv'
            ratings.write_text('Patient,Date,flag,flag2,rating\nPatient-001,week-000,,,Post-Op\nPatient-001,week-001,,,SD\nPatient-001,week-002,,,Post-Op\nPatient-001,week-003,,,SD\nPatient-001,week-004,,,SD\nPatient-001,week-004,,,PD\n')
            rows,excluded,_=build(root,ratings)
            self.assertEqual([r['week'] for r in rows],[1])
            self.assertIn('conflicting_duplicate_rating',[e['reason'] for e in excluded])
            self.assertIn('no_prior_valid_scan_in_segment',[e['reason'] for e in excluded])

    def test_summary_metrics_match_sklearn_with_ties_and_weights(self):
        import pandas as pd
        from sklearn.metrics import roc_auc_score, average_precision_score
        from assessment.summarize import stats
        frame=pd.DataFrame(dict(target=[0,1,1,0,1],probability=[.2,.2,.8,.7,.9],
              binary_prediction=[0,0,1,1,1],band=['lower','lower','higher','defer','higher'],
              fixed_rule_prediction=[np.nan]*5))
        weights=np.array([2,1,3,0,1])
        m=stats(frame,weights)
        self.assertAlmostEqual(m['auroc'],roc_auc_score(frame.target,frame.probability,sample_weight=weights))
        self.assertAlmostEqual(m['average_precision'],average_precision_score(frame.target,frame.probability,sample_weight=weights))
