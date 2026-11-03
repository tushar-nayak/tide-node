import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import package_models as packaging
from mask_forecast_experiment import MaskForecastNet
from lumiere_cohort import COMPARTMENTS


def scan(volumes, week):
    return SimpleNamespace(volumes_mm3=dict(zip(COMPARTMENTS, volumes)), week=week)


def transitions():
    result = []
    for i in range(48):
        first = scan([1000 + i * 70, 500 + i * 20, 3000 + i * 40], 0)
        last = scan([1300 + i * 80, 700 + i * 20, 3200 + i * 40], 4)
        result.append(SimpleNamespace(patient_id=f"p{i}", history=[first, last], last=last, dt_weeks=8))
    return result


class PackagingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mask_variants_and_integrity(self):
        for variant in ("ode", "static"):
            model = MaskForecastNet(12, 2, variant)
            # Exercise learned residuals rather than only persistence initialization.
            with torch.no_grad():
                model.resid.weight.fill_(0.02)
                model.flow.bias.fill_(0.01)
            checkpoint = self.root / f"{variant}.pt"
            torch.save(model.state_dict(), checkpoint)
            output = self.root / variant
            packaging.export_mask(checkpoint, output, "registered", {"all_cohort": False})
            restored, manifest = packaging.load_mask(output)
            self.assertEqual(manifest["architecture"]["constructor"]["variant"], variant)
            self.assertEqual(manifest["preprocessing"]["channels"][:4], ["FLAIR", "T1", "T2", "CT1"])
            self.assertEqual(manifest["preprocessing"]["frame"], "registered")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                packaging.export_mask(checkpoint, output, "atlas", {})
            with (output / "weights.pt").open("ab") as f:
                f.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                packaging.load_mask(output)

    def test_reject_incompatible_mask(self):
        checkpoint = self.root / "bad.pt"
        torch.save(MaskForecastNet(11, 2, "static").state_dict(), checkpoint)
        with self.assertRaisesRegex(ValueError, "channel"):
            packaging.export_mask(checkpoint, self.root / "out", "atlas", {})
        self.assertFalse((self.root / "out").exists())

    def test_volume_preprocessing_and_round_trip(self):
        ts = transitions()
        clinical = {"p0": {"Age at surgery (years)": "72", "Sex": "male", "MGMT qualitative": "methylated"}}
        x = packaging.volume_features(ts, clinical)
        self.assertEqual(x.shape, (48, 22))
        last = np.log1p(np.array([1300, 700, 3200]) / 1000)
        first = np.log1p(np.array([1000, 500, 3000]) / 1000)
        np.testing.assert_allclose(x[0, :3], last)
        np.testing.assert_allclose(x[0, 3:6], (last - first) / (4 / 52))
        np.testing.assert_allclose(x[0, 6:9], (last - first) * 2)
        np.testing.assert_array_equal(x[0, -5:], [1, 0, 1, 1, 1])
        np.testing.assert_array_equal(x[1, -5:], [0, 1, 0, 0, 0])
        ts[1].history = [ts[1].last]
        np.testing.assert_array_equal(packaging.volume_features(ts, clinical)[1, 3:12], np.zeros(9))
        models = [HistGradientBoostingRegressor(max_iter=3, min_samples_leaf=2, random_state=7)
                  .fit(x, np.sin(x[:, 0]) * (k + 1)) for k in range(3)]
        output = self.root / "volume"
        packaging.export_volume(models, output, {"training_scope": "synthetic_test", "all_cohort": False})
        with self.assertRaisesRegex(ValueError, "trust"):
            packaging.load_volume(output)
        restored, manifest = packaging.load_volume(output, trusted=True)
        self.assertEqual(manifest["preprocessing"]["features"], packaging.VOLUME_FEATURES)
        expected = packaging.predict_volume(models, ts, clinical)
        np.testing.assert_array_equal(packaging.predict_volume(restored, ts, clinical), expected)
        self.assertTrue(np.isfinite(expected).all())
        (output / "unexpected").write_text("x")
        with self.assertRaisesRegex(ValueError, "inventory"):
            packaging.verify(output)

    def test_source_guard(self):
        with self.assertRaisesRegex(ValueError, "Implementation differs"):
            packaging.check_source({"source_sha256": {"lumiere_cohort.py": "invalid"}}, ["lumiere_cohort.py"])

    def test_all_cohort_command_on_synthetic_fixture(self):
        ts = transitions()
        for i, t in enumerate(ts):
            for j, item in enumerate(t.history):
                item.source_name = f"week-{j * 4}"
            t.target = scan([1700 + i * 90, 800 + i * 30, 3600 + i * 20], 12)
            t.target.source_name = "week-12"
        cohort = SimpleNamespace(transitions=ts, patients=[t.patient_id for t in ts], excluded={})
        clinical_dir = self.root / "clinical"
        clinical_dir.mkdir()
        (clinical_dir / "LUMIERE-Demographics_Pathology.csv").write_text("Patient,Sex\n")
        contract = self.root / "contract.json"
        contract.write_text(json.dumps(dict(contract_id="synthetic-only", max_depth=2, learning_rate=0.05,
                                           max_iter=3, loss="absolute_error", random_state=0, early_stopping=False)))
        args = SimpleNamespace(contract=contract, clinical_dir=clinical_dir, data_dir=self.root,
                               output=self.root / "fitted")
        with patch.object(packaging, "cohort_inputs", return_value=(cohort, {})):
            packaging.fit_volume(args)
            with self.assertRaises(FileExistsError):
                packaging.fit_volume(args)
        models, manifest = packaging.load_volume(args.output, trusted=True)
        self.assertTrue(manifest["provenance"]["all_cohort"])
        self.assertEqual(len(manifest["provenance"]["transitions"]), len(ts))
        self.assertEqual(manifest["environment_scope"], "fitting and export environment")
        self.assertEqual(packaging.predict_volume(models, ts, {}).shape, (48, 3))

    def test_all_cohort_requires_fixed_contract(self):
        path = self.root / "contract.json"
        path.write_text(json.dumps({"max_iter": 3}))
        with self.assertRaisesRegex(ValueError, "Contract must"):
            packaging.fit_volume(SimpleNamespace(contract=path))


if __name__ == "__main__":
    unittest.main()
