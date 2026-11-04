import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import progression_model as pm  # noqa: E402
from approach_screen import assess_features  # noqa: E402
from lumiere_cohort import COMPARTMENTS  # noqa: E402
from package_models import sha256  # noqa: E402
from volume_trajectory_experiment import static_features  # noqa: E402


def scan(week, nec, enh, ede, rating=None):
    return SimpleNamespace(week=week, volumes_mm3=dict(zip(COMPARTMENTS, (nec, enh, ede))), rating=rating)


def visit(week, nec, enh, ede, rating=None):
    v = {'week': week, 'volumes_mm3': dict(zip(COMPARTMENTS, (nec, enh, ede)))}
    if rating is not None:
        v['rating'] = rating
    return v


def toy_bundle(root: Path) -> Path:
    import sklearn
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, len(pm.FEATURES)))
    y = (x[:, pm.FEATURES.index('change_vs_nadir_enhancing')] + 0.3 * rng.normal(size=120) > 0).astype(int)
    model = pm.new_model().fit(x, y)
    manifest = {'kind': pm.KIND, 'schema_version': pm.SCHEMA, 'sklearn_version': sklearn.__version__, 'features': pm.FEATURES,
                'held_out': {'auroc': 0.8, 'thresholds': pm.thresholds(model.predict_proba(x)[:, 1], y)},
                'source_sha256': {n: sha256(pm.SCRIPTS / n) for n in pm.SOURCES}}
    pm.write_bundle(model, manifest, root / 'bundle', probe=x)
    return root / 'bundle'


class ProgressionModelTest(unittest.TestCase):
    def test_features_match_the_evaluated_experiment(self):
        static = static_features('p', {'p': {'Age at surgery (years)': '55', 'Sex': 'male', 'MGMT qualitative': 'methylated'}})
        history = [scan(0, 3000, 2500, 20000, 'Post-Op'), scan(12, 2800, 0, 15000, 'SD'), scan(20, 2900, 1800, 16000, 'CR')]
        current = scan(31, 4200, 5200, 26000)
        t = SimpleNamespace(patient_id='p', history=history, last=history[-1], target=current, dt_weeks=current.week - history[-1].week)
        np.testing.assert_array_equal(pm.features(history, current, static), assess_features(t, static))
        self.assertEqual(len(pm.features(history, current, static)), len(pm.FEATURES))

    def test_growth_constraints_cover_the_change_features(self):
        cst = pm.GBM_PARAMS['monotonic_cst']
        self.assertEqual(len(cst), len(pm.FEATURES))
        self.assertEqual({f for f, c in zip(pm.FEATURES, cst) if c == 1}, set(pm.MONOTONE_UP))

    def test_history_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'h.json'

            def check(history, message, **extra):
                path.write_text(json.dumps(dict(version=1, patient_id='p', history=history, **extra)))
                with self.assertRaisesRegex(ValueError, message):
                    pm.read_history(path)
            check([visit(0, 1, 1, 1)], 'at least one earlier scan')
            check([visit(0, 1, 1, 1, 'SD'), visit(10, 1, 1, 1, 'Post-Op'), visit(20, 1, 1, 1)], 'new segment')
            check([visit(0, 1, 1, 1), visit(10, 1, 1, 1)], 'rating is required')
            check([visit(0, 1, 1, 1, 'SD'), visit(10, 1, 1, 1, 'PD')], 'must not carry a rating')
            check([visit(10, 1, 1, 1, 'SD'), visit(10, 1, 1, 1)], 'increasing')
            check([visit(0, 1, 1, 1, 'maybe'), visit(10, 1, 1, 1)], 'rating must be one of')
            check([visit(0, 1, 1, 1, 'SD'), visit(10, 1, 1, 1)], 'clinical', clinical={'Age at surgery (years)': 50})
            path.write_text(json.dumps(dict(version=1, patient_id='p', history=[visit(0, 1, 2, 3, 'Post-Op'), visit(9, 4, 5, 6, 'unknown'),
                                                                                visit(20, 7, 8, 9)])))
            patient, scans, clinical = pm.read_history(path)
            self.assertEqual((patient, len(scans), scans[1].rating, clinical), ('p', 3, None, {}))

    def test_bundle_round_trip_and_cli_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = toy_bundle(root)
            history = root / 'h.json'
            history.write_text(json.dumps(dict(version=1, patient_id='p', history=[
                visit(0, 3000, 2500, 20000, 'Post-Op'), visit(12, 2800, 2000, 15000, 'SD'), visit(24, 4200, 5200, 26000)])))
            with self.assertRaisesRegex(ValueError, 'trust'):
                pm.predict_progression(history, bundle, root / 'untrusted')
            result = pm.predict_progression(history, bundle, root / 'out', trusted=True)
            written = json.loads((root / 'out/progression.json').read_text())
            self.assertEqual(written['probability_pd'], result['probability_pd'])
            self.assertTrue(0 <= result['probability_pd'] <= 1)
            t80, t90 = (result['thresholds'][k]['threshold'] for k in ('specificity_80', 'sensitivity_90'))
            expected = 'likely PD' if result['probability_pd'] >= t80 else ('uncertain' if result['probability_pd'] >= t90 else 'PD less likely')
            self.assertEqual(result['call'], expected)
            rates = pm.tier_rates(np.array([0.1, 0.5, 0.95, 0.99]), np.array([0, 1, 1, 0]), result['thresholds'])
            self.assertAlmostEqual(sum(v['fraction_of_scans'] for v in rates.values()), 1.0)
            self.assertAlmostEqual(result['measurements_ml']['enhancing_change_vs_previous_pct'], 160.0)
            self.assertEqual(result['previous_rating'], 'SD')
            (bundle / 'estimator.pkl').write_bytes((bundle / 'estimator.pkl').read_bytes() + b'0')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                pm.load(bundle, trusted=True)


if __name__ == '__main__':
    unittest.main()
