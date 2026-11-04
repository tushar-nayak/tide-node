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


def visit(week, nec, enh, ede, rating=None, segmentation=None):
    v = {'week': week, 'volumes_mm3': dict(zip(COMPARTMENTS, (nec, enh, ede)))}
    if rating is not None:
        v['rating'] = rating
    if segmentation is not None:
        v['segmentation'] = segmentation
    return v


def toy_bundle(root: Path) -> Path:
    import sklearn
    rng = np.random.default_rng(0)
    models, probes, entries = {}, {}, {}
    for name, feats, masks in (('volumes', pm.FEATURES, False), ('volumes_masks', pm.FEATURES_MASKS, True)):
        x = rng.normal(size=(150, len(feats)))
        y = (x[:, pm.FEATURES.index('change_vs_nadir_enhancing')] + 0.3 * rng.normal(size=150) > 0).astype(int)
        models[name], probes[name] = pm.new_model(masks).fit(x, y), x
        scores = models[name].predict_proba(x)[:, 1]
        thr = pm.thresholds(scores, y)
        entries[name] = {'features': feats, 'held_out': {'auroc': 0.8, 'thresholds': thr, 'tiers': pm.tier_rates(scores, y, thr)}}
    manifest = {'kind': pm.KIND, 'schema_version': pm.SCHEMA, 'sklearn_version': sklearn.__version__, 'models': entries,
                'mask_contract': {'margin_voxels': pm.MASK_MARGIN_VOXELS}, 'source_sha256': {n: sha256(pm.SCRIPTS / n) for n in pm.SOURCES}}
    pm.write_bundle(models, manifest, root / 'bundle', probes=probes)
    return root / 'bundle'


def write_seg(path: Path, blobs):
    import nibabel as nib
    seg = np.zeros((182, 218, 182), np.uint8)
    for (x, y, z), size, label in blobs:
        seg[x:x + size, y:y + size, z:z + size] = label
    nib.save(nib.Nifti1Image(seg, np.eye(4)), path)
    return str(path)


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
        self.assertEqual(len(pm.GBM_PARAMS_MASKS['monotonic_cst']), len(pm.FEATURES_MASKS))

    def test_new_tumor_outside_the_margin_is_counted_and_inside_is_not(self):
        prev = np.zeros((3, 40, 40, 40), bool)
        prev[:, 10:14, 10:14, 10:14] = True  # WT, TC, ET
        near, far = prev.copy(), prev.copy()
        near[[0, 2], 15:18, 10:14, 10:14] = True  # touches the prior core within the 8 mm margin
        far[[0, 2], 30:34, 30:34, 30:34] = True  # a new 4x4x4-voxel lesion far from it
        f_near = dict(zip(pm.MASK_FEATURES, pm.mask_features(prev, near, pm.MASK_MARGIN_VOXELS)))
        f_far = dict(zip(pm.MASK_FEATURES, pm.mask_features(prev, far, pm.MASK_MARGIN_VOXELS)))
        self.assertEqual((f_near['et_new_ml'], f_near['et_new_components']), (0.0, 0.0))
        self.assertAlmostEqual(f_far['et_new_ml'], 64 * pm.VOXEL_ML)
        self.assertEqual((f_far['et_new_components'], f_far['et_components'], f_far['et_components_prev']), (1.0, 2.0, 1.0))
        self.assertAlmostEqual(f_far['wt_new_ml'], 64 * pm.VOXEL_ML)
        self.assertEqual(pm.mask_features(prev, np.zeros_like(prev), 4)[3], 1.0)  # no enhancing left: fraction defined as 1

    def test_pooled_masks_follow_the_label_convention(self):
        seg = np.zeros((4, 4, 4), np.float32)
        seg[:2, :2, :2] = 1  # enhancing -> WT, TC, ET
        seg[2:, 2:, 2:] = 3  # edema -> WT only
        m = pm.pooled_masks(seg)
        self.assertEqual(m.shape, (3, 2, 2, 2))
        self.assertTrue(m[:, 0, 0, 0].all())
        self.assertEqual(m[:, 1, 1, 1].tolist(), [True, False, False])

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
            check([visit(0, 1, 1, 1, 'SD', segmentation=''), visit(10, 1, 1, 1)], 'segmentation')
            path.write_text(json.dumps(dict(version=1, patient_id='p', history=[
                visit(0, 1, 2, 3, 'Post-Op'), visit(9, 4, 5, 6, 'unknown', segmentation='a.nii.gz'), visit(20, 7, 8, 9)])))
            patient, scans, clinical = pm.read_history(path)
            self.assertEqual((patient, len(scans), scans[1].rating, clinical), ('p', 3, None, {}))
            self.assertEqual(scans[1].segmentation, (Path(td) / 'a.nii.gz').resolve())

    def test_bundle_round_trip_and_both_cli_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = toy_bundle(root)
            history = root / 'h.json'
            visits = [visit(0, 3000, 2500, 20000, 'Post-Op'), visit(12, 2800, 2000, 15000, 'SD'), visit(24, 4200, 5200, 26000)]
            history.write_text(json.dumps(dict(version=1, patient_id='p', history=visits)))
            with self.assertRaisesRegex(ValueError, 'trust'):
                pm.predict_progression(history, bundle, root / 'untrusted')
            result = pm.predict_progression(history, bundle, root / 'out', trusted=True)
            written = json.loads((root / 'out/progression.json').read_text())
            self.assertEqual((written['probability_pd'], written['model']), (result['probability_pd'], 'volumes'))
            self.assertIsNone(result['new_tumor_outside_prior'])
            t80, t90 = (result['thresholds'][k]['threshold'] for k in ('specificity_80', 'sensitivity_90'))
            p = result['probability_pd']
            self.assertEqual(result['call'], 'likely PD' if p >= t80 else ('uncertain' if p >= t90 else 'PD less likely'))
            self.assertAlmostEqual(result['measurements_ml']['enhancing_change_vs_previous_pct'], 160.0)

            prev_seg = write_seg(root / 'prev.nii.gz', [((80, 100, 80), 10, 1)])
            cur_seg = write_seg(root / 'cur.nii.gz', [((80, 100, 80), 10, 1), ((140, 40, 40), 8, 1)])
            visits[1]['segmentation'], visits[2]['segmentation'] = prev_seg, cur_seg
            history.write_text(json.dumps(dict(version=1, patient_id='p', history=visits)))
            masked = pm.predict_progression(history, bundle, root / 'masked', trusted=True)
            self.assertEqual(masked['model'], 'volumes_masks')
            self.assertEqual(masked['new_tumor_outside_prior']['new_enhancing_components'], 1)
            self.assertAlmostEqual(masked['new_tumor_outside_prior']['new_enhancing_ml'], 64 * pm.VOXEL_ML)
            del visits[1]['segmentation']
            history.write_text(json.dumps(dict(version=1, patient_id='p', history=visits)))
            with self.assertRaisesRegex(ValueError, 'both the previous and the current'):
                pm.predict_progression(history, bundle, root / 'half', trusted=True)

            (bundle / 'estimator_volumes.pkl').write_bytes((bundle / 'estimator_volumes.pkl').read_bytes() + b'0')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                pm.load(bundle, trusted=True)


if __name__ == '__main__':
    unittest.main()
