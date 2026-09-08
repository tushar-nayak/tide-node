from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_neural_ode_pipeline import (
    adaptive_flair_mask,
    binary_dice,
    flatten_metric_row,
    flow_smoothness_penalty,
    get_device,
    mask_region,
    warp_baseline,
    write_metric_rows,
)


class RunNeuralODEPipelineMetricsTest(unittest.TestCase):
    def test_get_device_accepts_explicit_cpu(self) -> None:
        self.assertEqual(str(get_device("cpu")), "cpu")

    def test_get_device_rejects_unavailable_cuda(self) -> None:
        try:
            device = get_device("cuda")
        except RuntimeError as error:
            self.assertIn("CUDA was requested", str(error))
        else:
            self.assertEqual(str(device), "cuda")

    def test_flatten_metric_row_expands_modality_metrics(self) -> None:
        row = {
            "patient_id": "Patient-001",
            "history_weeks": [0, 4],
            "target_week": 8,
            "mse": 0.1,
            "mae": 0.2,
            "focal_mse": 0.15,
            "focal_mae": 0.25,
            "relative_flair_volume_diff": 0.3,
            "wt_dice": 0.8,
            "tc_dice": None,
            "et_dice": None,
            "true_tumor_voxels": 42.0,
            "per_modality_mse": {"FLAIR": 0.11, "T1": 0.12, "T2": 0.13, "CT1": 0.14},
            "per_modality_mae": {"FLAIR": 0.21, "T1": 0.22, "T2": 0.23, "CT1": 0.24},
        }

        flat = flatten_metric_row(row)

        self.assertEqual(flat["history_weeks"], "0,4")
        self.assertEqual(flat["flair_mse"], 0.11)
        self.assertEqual(flat["ct1_mae"], 0.24)
        self.assertEqual(flat["focal_mse"], 0.15)
        self.assertEqual(flat["focal_mae"], 0.25)
        self.assertEqual(flat["wt_dice"], 0.8)
        self.assertIsNone(flat["tc_dice"])
        self.assertEqual(flat["true_tumor_voxels"], 42.0)

    def test_write_metric_rows_writes_header_for_empty_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "metrics.csv"
            write_metric_rows([], output_path)

            with output_path.open(newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("patient_id", rows[0])
        self.assertIn("ct1_mae", rows[0])
        self.assertIn("focal_mse", rows[0])
        self.assertIn("focal_mae", rows[0])
        self.assertIn("wt_dice", rows[0])
        self.assertIn("true_tumor_voxels", rows[0])

    def test_mask_region_and_binary_dice(self) -> None:
        # Labels: 1=Enhancing Core, 2=Necrotic/Non-Enhancing, 3=Edema (verified against
        # LUMIERE's measured_volumes_in_mm3.json voxel counts, not assumed BraTS ordering).
        mask = torch.tensor([[1, 2], [3, 0]])
        whole_tumor = mask_region(mask, (1, 2, 3))
        self.assertTrue(torch.equal(whole_tumor, torch.tensor([[True, True], [True, False]])))

        # Perfect overlap.
        self.assertAlmostEqual(binary_dice(whole_tumor, whole_tumor), 1.0)

        # No overlap at all.
        empty = torch.zeros_like(whole_tumor, dtype=torch.bool)
        self.assertAlmostEqual(binary_dice(whole_tumor, empty), 0.0)

        # Both empty is a correct "no tumor" prediction, not an undefined 0/0.
        self.assertAlmostEqual(binary_dice(empty, empty), 1.0)

        # Partial overlap: pred={0,1}, true={1,2} (flat indices) -> intersection size 1.
        pred = torch.tensor([True, True, False])
        true = torch.tensor([False, True, True])
        self.assertAlmostEqual(binary_dice(pred, true), 2 * 1 / (2 + 2))

    def test_adaptive_flair_mask_thresholds_relative_to_peak(self) -> None:
        # A fixed absolute cutoff (the old `> 0.5`) never fires on a volume whose peak
        # intensity is below it, even though it clearly has a bright "hyperintense" region
        # relative to its own scale. This is exactly what real LUMIERE volumes looked like.
        flair = np.array([0.05, 0.1, 0.4, 0.42, 0.05])
        mask = adaptive_flair_mask(flair, fraction_of_max=0.45)
        # peak=0.42, threshold=0.189 -> indices 2,3 qualify, nothing would under a fixed >0.5.
        self.assertTrue(np.array_equal(mask, np.array([False, False, True, True, False])))
        self.assertEqual(int((flair > 0.5).sum()), 0)

        # An all-zero volume has no peak to threshold against; must not divide by zero.
        empty = np.zeros(4, dtype=np.float32)
        self.assertFalse(adaptive_flair_mask(empty).any())

    def test_warp_baseline_identity_flow_reproduces_input(self) -> None:
        # slice_count=2, MODALITIES=4 -> 8 channels; a zero flow field must be a no-op.
        baseline = torch.rand(1, 8, 16, 16)
        zero_flow = torch.zeros(1, 4, 16, 16)
        warped = warp_baseline(baseline, zero_flow, slice_count=2)
        self.assertTrue(torch.allclose(warped, baseline, atol=1e-5))

    def test_warp_baseline_shifts_content_in_expected_direction(self) -> None:
        # slice_count=1, MODALITIES=4 -> 4 channels sharing one flow field.
        img = torch.zeros(1, 4, 16, 16)
        img[0, :, 8, 8] = 1.0
        flow = torch.zeros(1, 2, 16, 16)
        flow[:, 0, :, :] = 0.25  # shift the sampling grid in +x -> content appears to move -x
        warped = warp_baseline(img, flow, slice_count=1)
        peak_row, peak_col = divmod(int(warped[0, 0].argmax()), 16)
        self.assertEqual(peak_row, 8)
        self.assertLess(peak_col, 8)

    def test_flow_smoothness_penalty_zero_for_constant_field(self) -> None:
        const_flow = torch.full((1, 2, 16, 16), 0.1)
        self.assertAlmostEqual(flow_smoothness_penalty(const_flow).item(), 0.0, places=6)

        step_flow = torch.zeros(1, 2, 16, 16)
        step_flow[:, :, :, 8:] = 1.0  # a hard discontinuity halfway across
        self.assertGreater(flow_smoothness_penalty(step_flow).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
