import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import package_models as pm  # noqa: E402
import prediction as p  # noqa: E402
import volume_intervals as vi  # noqa: E402
from lumiere_cohort import COMPARTMENTS  # noqa: E402

COL = {name: i for i, name in enumerate(pm.VOLUME_FEATURES)}


def synthetic_x(rng, n):
    x = rng.normal(size=(n, len(pm.VOLUME_FEATURES)))
    dt = rng.integers(2, 40, size=n).astype(float)
    x[:, COL['log1p_dt_weeks']] = np.log1p(dt)
    x[:, COL['dt_years']] = dt / 52
    x[:, COL['history_count']] = rng.integers(1, 5, size=n)
    last = rng.uniform(0, 4, size=(n, 3))
    last[rng.random((n, 3)) < 0.1] = 0
    x[:, :3] = last
    return x, dt


def synthetic_data(rng, n_patients=24, per_patient=5):
    n = n_patients * per_patient
    x, dt = synthetic_x(rng, n)
    last3 = x[:, :3]
    true3 = np.maximum(last3 + rng.normal(size=(n, 3)) * (0.2 + 0.15 * np.log1p(dt))[:, None], 0)
    return {'x': x, 'dt': dt, 'delta': true3 - last3, 'patient': np.repeat([f'P{i:02d}' for i in range(n_patients)], per_patient),
            'last': np.column_stack([last3, vi.total_log(last3)]), 'true': np.column_stack([true3, vi.total_log(true3)])}


class ConformalMathTest(unittest.TestCase):
    def test_order_statistics_use_n_plus_one_correction(self):
        scores = np.arange(1, 100, dtype=float)[::-1]  # n = 99, unsorted on purpose
        self.assertEqual(vi.conformal_multipliers(scores, 0.8), (10.0, 90.0))
        self.assertEqual(vi.conformal_multipliers(scores, 0.5), (25.0, 75.0))
        with self.assertRaisesRegex(ValueError, 'too few'):
            vi.conformal_multipliers(np.arange(20.0), 0.95)

    def test_total_design_uses_summed_compartment_volume(self):
        x = np.zeros((1, len(pm.VOLUME_FEATURES)))
        x[0, :3] = np.log1p([1.0, 2.0, 3.0])
        x[0, COL['history_count']] = 1
        x[0, COL['log1p_dt_weeks']] = np.log1p(12)
        np.testing.assert_allclose(vi.scale_design(x, 'total'), [[np.log1p(12), np.log1p(6.0), 1.0]])
        np.testing.assert_allclose(vi.scale_design(x, 'edema'), [[np.log1p(12), np.log1p(3.0), 0.0, 1.0]])

    def test_scaled_ranges_hold_coverage_and_widen_with_horizon(self):
        rng = np.random.default_rng(0)
        x, _ = synthetic_x(rng, 6000)
        design = vi.scale_design(x, 'enhancing')
        r = (0.2 + 0.3 * design[:, 0]) * rng.standard_t(5, size=6000)
        cal, test = slice(0, 3000), slice(3000, None)
        scale = vi.fit_scale(design[cal], np.abs(r[cal]))
        scores = r[cal] / vi.scale_of(scale, design[cal])
        entry = {'scale': scale, 'multipliers': {vi.level_key(lv): vi.conformal_multipliers(scores, lv) for lv in vi.LEVELS}}
        ranges = vi.apply_ranges(entry, np.full(3000, 10.0), design[test])  # far from 0, so clipping is inactive
        y = 10.0 + r[test]
        for key, (lo, hi) in ranges.items():
            self.assertAlmostEqual(((lo <= y) & (y <= hi)).mean(), int(key[:-1]) / 100, delta=0.03)
        lo, hi = ranges['80%']
        long = design[test][:, 0] > np.median(design[test][:, 0])
        self.assertGreater((hi - lo)[long].mean(), (hi - lo)[~long].mean())

    def test_calibrate_is_patient_grouped_and_nested(self):
        data = synthetic_data(np.random.default_rng(1))
        params = dict(max_depth=2, learning_rate=0.1, max_iter=5, loss='absolute_error', random_state=0, early_stopping=False)
        cal = vi.calibrate(data, np.arange(len(data['patient'])), params, seeds=(0,))
        self.assertEqual(set(cal), set(vi.METHODS))
        for method in vi.METHODS:
            for q in vi.QUANTITIES:
                entry = cal[method][q]
                self.assertEqual(entry['n_scores'], len(data['patient']))
                self.assertEqual(entry['scale'] is None, method.endswith('constant'))
                bounds = [entry['multipliers'][vi.level_key(lv)] for lv in vi.LEVELS]
                self.assertTrue(all(a[0] >= b[0] and a[1] <= b[1] for a, b in zip(bounds, bounds[1:])))


class VolumeRangesCliTest(unittest.TestCase):
    def test_predict_writes_nested_ranges_tied_to_the_bundle(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = [HistGradientBoostingRegressor(max_iter=2, early_stopping=False).fit(
                rng.normal(size=(40, 22)), rng.normal(size=40)) for _ in range(3)]
            pm.export_volume(models, root / 'bundle', {'all_cohort': False})
            data = synthetic_data(rng)
            params = dict(max_depth=2, learning_rate=0.1, max_iter=5, loss='absolute_error', random_state=0, early_stopping=False)
            cal = vi.calibrate(data, np.arange(len(data['patient'])), params, seeds=(0,))[vi.DEPLOYED]
            doc = vi.calibration_document(cal, bundle=root / 'bundle', contract=dict(contract_id='test', **params),
                                          dt=data['dt'], patients=24)
            calibration = root / 'cal.json'
            calibration.write_text(json.dumps(doc))
            scans = [SimpleNamespace(week=w, volumes_mm3=dict(zip(COMPARTMENTS, [1000 + w, 2000 + w, 3000 + w]))) for w in (0, 8)]
            history = root / 'history.json'
            history.write_text(json.dumps(dict(version=1, patient_id='example',
                                               history=[dict(week=s.week, volumes_mm3=s.volumes_mm3) for s in scans])))

            ranges = p.predict_volume_history(history, 12, root / 'bundle', root / 'out', trusted=True, intervals=calibration)
            written = json.loads((root / 'out/intervals.json').read_text())
            self.assertEqual(written['volumes'], ranges['volumes'])
            point = json.loads((root / 'out/volumes.json').read_text())
            for name in COMPARTMENTS:
                self.assertAlmostEqual(written['volumes'][name]['forecast'], point[name])
            self.assertAlmostEqual(written['volumes']['Total']['forecast'], sum(point.values()))
            for v in written['volumes'].values():
                spans = [v['ranges'][k] for k in written['levels']]
                self.assertTrue(all(0 <= lo <= hi for lo, hi in spans))
                self.assertTrue(all(a[0] >= b[0] and a[1] <= b[1] for a, b in zip(spans, spans[1:])))
            self.assertIsNone(written['warning'])
            meta = json.loads((root / 'out/metadata.json').read_text())
            self.assertEqual(meta['intervals']['sha256'], pm.sha256(calibration))

            long = p.predict_volume_history(history, 500, root / 'bundle', root / 'long', trusted=True, intervals=calibration)
            self.assertIn('longer than any', long['warning'])

            bad = dict(doc, bundle_estimators_sha256='0' * 64)
            calibration.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'different volume bundle'):
                p.predict_volume_history(history, 12, root / 'bundle', root / 'bad', trusted=True, intervals=calibration)
            self.assertFalse((root / 'bad').exists())
            broken = json.loads(json.dumps(doc))
            broken['quantities']['Total']['scale_features'] = ['log1p_dt_weeks']
            calibration.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'scale features'):
                vi.load_calibration(calibration, root / 'bundle')
            del broken['horizon_weeks']
            broken['quantities']['Total']['scale_features'] = list(vi.SCALE_FEATURES['total'])
            calibration.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'Malformed'):
                vi.load_calibration(calibration, root / 'bundle')


if __name__ == '__main__':
    unittest.main()
