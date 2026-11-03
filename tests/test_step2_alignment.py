import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from register_step2_pairs import as_single_affine, pair_stem  # noqa: E402


def scan(pid, name):
    return SimpleNamespace(patient_id=pid, source_name=name)


class SingleAffineTest(unittest.TestCase):
    def test_composite_collapses_exactly(self):
        import SimpleITK as sitk

        rigid = sitk.Euler3DTransform((90, 109, 90), 0.05, -0.03, 0.1, (2.0, -3.0, 1.5))
        aff = sitk.AffineTransform(3)
        aff.SetMatrix([1.02, 0.01, 0, 0, 0.98, 0.02, 0, 0, 1.01])
        aff.SetTranslation((0.5, 0.2, -0.4))
        comp = sitk.CompositeTransform([rigid, aff])
        flat = as_single_affine(comp)
        rng = np.random.default_rng(0)
        for p in rng.uniform(0, 200, (20, 3)):
            np.testing.assert_allclose(flat.TransformPoint(tuple(p)), comp.TransformPoint(tuple(p)), atol=1e-6)

    def test_identity(self):
        import SimpleITK as sitk

        flat = as_single_affine(sitk.Euler3DTransform())
        np.testing.assert_allclose(flat.TransformPoint((10.0, 20.0, 30.0)), (10.0, 20.0, 30.0), atol=1e-9)


class FramePathTest(unittest.TestCase):
    def test_paths(self):
        from mask_forecast_experiment import in_last_frame_path

        last, tgt, prev = scan("Patient-001", "week-044"), scan("Patient-001", "week-056"), scan("Patient-001", "week-000-2")
        t = SimpleNamespace(last=last, target=tgt, history=[prev, last])
        cache, aligned = Path("/c"), Path("/a")
        self.assertEqual(in_last_frame_path("atlas", cache, aligned, t, tgt), Path("/c/Patient-001/week-056.npy"))
        self.assertEqual(in_last_frame_path("registered", cache, aligned, t, last), Path("/c/Patient-001/week-044.npy"))
        self.assertEqual(in_last_frame_path("registered", cache, aligned, t, tgt),
                         Path("/a/Patient-001") / f"{pair_stem('week-056', 'week-044')}.npy")
        self.assertEqual(in_last_frame_path("registered", cache, aligned, t, prev),
                         Path("/a/Patient-001/week-000-2__in__week-044.npy"))


if __name__ == "__main__":
    unittest.main()
