import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prediction as p
from mask_forecast_experiment import MaskForecastDataset, MaskForecastNet


class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / 'input.nii'
        self.file.touch()
        self.manifest = dict(version=1, patient_id='test', frame='atlas', history=[
            dict(week=w, modalities={m: str(self.file) for m in p.MODS}, segmentation=str(self.file)) for w in (0, 8)])
        self.path = self.root / 'history.json'
        self.affine = np.array([[-1., 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
        self.volume = np.ones((182, 218, 182), np.float32)

    def prepare(self, manifest=None, horizon=12, frame='atlas'):
        self.path.write_text(json.dumps(manifest or self.manifest))
        with patch.object(p, 'load_volume', return_value=(self.volume, self.affine)):
            return p.prepare_history(self.path, horizon, frame)

    def test_dataset_inputs_equivalent_and_target_never_read(self):
        batch, last, meta = self.prepare()
        pid = self.root / 'test'
        pid.mkdir()
        for name in ('prev', 'last', 'target'):
            np.save(pid / f'{name}.npy', last)
        scans = [SimpleNamespace(patient_id='test', source_name=n, week=w) for n, w in zip(('prev', 'last', 'target'), (0, 8, 20))]
        transition = SimpleNamespace(last=scans[1], history=scans[:2], target=scans[2], dt_weeks=12)
        expected = MaskForecastDataset([transition], self.root, False)[0]
        for k in ('x', 'last_mask', 'dt'):
            torch.testing.assert_close(batch[k][0], expected[k], rtol=0, atol=0)
        # Inference has no transition or target argument and still works after target deletion.
        (pid / 'target.npy').unlink()
        again, _, _ = self.prepare()
        torch.testing.assert_close(batch['x'], again['x'], rtol=0, atol=0)
        np.testing.assert_allclose(np.array(meta['affine']) @ [0, 0, 0, 1], self.affine @ [.5, .5, .5, 1])

    def test_reject_future_fields_and_invalid_history(self):
        for mutate in (
            lambda m: m.update(target='future.nii'),
            lambda m: m['history'][0].update(future_mask='future.nii'),
            lambda m: m['history'][0].pop('segmentation'),
            lambda m: m['history'][0].update(segmentation='missing.nii'),
            lambda m: m['history'][1].update(week=0),
            lambda m: m.update(history=[]),
        ):
            m = copy.deepcopy(self.manifest)
            mutate(m)
            with self.assertRaises(ValueError):
                self.prepare(m)
        for horizon in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.prepare(horizon=horizon)

    def test_registered_requires_explicit_history_alignment(self):
        m = copy.deepcopy(self.manifest)
        m['frame'] = 'registered'
        with self.assertRaisesRegex(ValueError, 'aligned'):
            self.prepare(m, frame='registered')

    def test_registered_cache_is_used_without_target(self):
        m = copy.deepcopy(self.manifest)
        m['frame'] = 'registered'
        cache = np.zeros(p.CACHE_SHAPE, np.float16)
        cache[4:, 20:30, 20:30, 20:30] = .75
        np.save(self.root / 'previous.npy', cache)
        m['history'][0].update(cache='previous.npy', aligned_to_week=8)
        batch, _, _ = self.prepare(m, frame='registered')
        np.testing.assert_array_equal(batch['x'][0, 7:10, :91, :109, :91].numpy(), cache[4:])
        cache[4, 0, 0, 0] = float('nan')
        np.save(self.root / 'previous.npy', cache)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.prepare(m, frame='registered')

    def test_single_history_flags_and_export_geometry(self):
        m = copy.deepcopy(self.manifest)
        m['history'] = m['history'][-1:]
        batch, last, meta = self.prepare(m)
        self.assertEqual(torch.count_nonzero(batch['x'][:, 7:]).item(), 0)
        masks = last[4:7] > .5
        output = self.root / 'output'
        p.export_prediction(output, masks, last, meta)
        img = nib.load(output / 'wt.nii.gz')
        np.testing.assert_allclose(img.affine, meta['affine'])
        np.testing.assert_allclose(img.get_qform(), meta['affine'])
        self.assertEqual(img.shape, (91, 109, 91))
        self.assertEqual(img.header.get_xyzt_units()[0], 'mm')
        volumes = json.loads((output / 'volumes.json').read_text())
        self.assertAlmostEqual(volumes['wt']['predicted_ml'], masks[0].sum() * .008)
        self.assertTrue((output / 'overlay.png').stat().st_size > 0)
        with self.assertRaises(FileExistsError):
            p.export_prediction(output, masks, last, meta)

    def test_geometry_and_label_validation(self):
        image = self.root / 'bad.nii'
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), image)
        with self.assertRaises(ValueError):
            p.load_volume(image)
        self.volume[0, 0, 0] = 4
        with self.assertRaisesRegex(ValueError, 'labels'):
            self.prepare()

    def test_both_architectures_checkpoint_and_forward(self):
        torch.set_num_threads(2)
        for variant in ('static', 'ode'):
            model = MaskForecastNet(12, 1, variant).eval()
            ckpt = self.root / f'{variant}.pt'
            torch.save(model.state_dict(), ckpt)
            bundle = dict(version=1, variant=variant, base=1, frame='atlas',
                          preprocessing=p.PREPROCESSING, checkpoint=ckpt.name)
            descriptor = self.root / 'bundle.json'
            descriptor.write_text(json.dumps(bundle))
            restored, _ = p.load_bundle(descriptor)
            x = torch.randn(1, 12, 16, 16, 16)
            lm = (torch.rand(1, 3, 16, 16, 16) > .5).float()
            dt = torch.tensor([12/52])
            with torch.inference_mode():
                a, _ = model(x, lm, dt)
                b, _ = restored(x, lm, dt)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            np.testing.assert_array_equal(p.infer(restored, dict(x=x, last_mask=lm, dt=dt)), lm[0].numpy().astype(bool))


if __name__ == '__main__':
    unittest.main()
