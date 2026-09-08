from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_neural_ode_pipeline import (
    binary_dice,
    flatten_metric_row,
    get_device,
    mask_region,
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


if __name__ == "__main__":
    unittest.main()
