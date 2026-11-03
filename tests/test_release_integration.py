import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prediction as p
import package_models as pm
from mask_forecast_experiment import MaskForecastNet
from lumiere_cohort import COMPARTMENTS
from volume_trajectory_experiment import history_features, static_features, y_of

class ReleaseIntegrationTest(unittest.TestCase):
    def test_packaged_cv_ensemble(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            members = []
            for i in range(5):
                torch.manual_seed(i)
                model = MaskForecastNet(12, 1, 'static').eval()
                torch.nn.init.normal_(model.resid.weight, std=.01)
                torch.nn.init.normal_(model.flow.weight, std=.01)
                ck = root / f'{i}.pt'
                torch.save(model.state_dict(), ck)
                pm.export_mask(ck, root / str(i), 'atlas', {'all_cohort': False, 'fold': i})
                members.append(str(i))
            descriptor = root / 'ensemble.json'
            descriptor.write_text(json.dumps(dict(version=1, kind='mask_cv_ensemble', members=members,
                label='cross-validation ensemble; no independent ensemble test estimate', aggregation='mean_logits')))
            model, meta = p.load_bundle(descriptor)
            x = torch.randn(1,12,16,16,16)
            lm = torch.rand(1,3,16,16,16)
            dt = torch.tensor([.2])
            with torch.no_grad():
                expected = torch.stack([m(x,lm,dt)[0] for m in model.members]).mean(0)
                actual = model(x,lm,dt)[0]
            torch.testing.assert_close(actual, expected)
            self.assertEqual(meta['frame'], 'atlas')
            bad = json.loads(descriptor.read_text())
            bad['members'][-1] = bad['members'][0]
            descriptor.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                p.load_bundle(descriptor)

    def test_volume_history_contract_and_serialization(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(4)
            models = [HistGradientBoostingRegressor(max_iter=2, early_stopping=False).fit(
                rng.normal(size=(40,22)), rng.normal(size=40)) for _ in range(3)]
            pm.export_volume(models, root/'bundle', {'all_cohort': False})
            scans = [SimpleNamespace(week=w, volumes_mm3=dict(zip(COMPARTMENTS, [1000+w,2000+w,3000+w]))) for w in [0,8]]
            doc = dict(version=1, patient_id='example', history=[dict(week=s.week, volumes_mm3=s.volumes_mm3) for s in scans])
            history = root/'history.json'
            history.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'trust'):
                p.predict_volume_history(history,12,root/'bundle',root/'untrusted')
            p.predict_volume_history(history,12,root/'bundle',root/'out',trusted=True)
            t = SimpleNamespace(history=scans,last=scans[-1],dt_weeks=12)
            x = history_features(t, static_features('example', {}))[None]
            expected = np.expm1(np.maximum(y_of(scans[-1])+[m.predict(x)[0] for m in models],0))
            result = json.loads((root/'out/volumes.json').read_text())
            np.testing.assert_array_equal([result[c] for c in COMPARTMENTS],expected)
            doc['target'] = 'forbidden'
            history.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                p.predict_volume_history(history,12,root/'bundle',root/'bad',trusted=True)
