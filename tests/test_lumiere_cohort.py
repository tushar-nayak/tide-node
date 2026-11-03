from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lumiere_cohort import build_cohort, patient_folds  # noqa: E402


def make_scan(root: Path, patient: str, name: str) -> None:
    base = root / patient / name / "DeepBraTumIA-segmentation" / "atlas"
    (base / "skull_strip").mkdir(parents=True)
    for mod in ("flair", "t1", "t2", "ct1"):
        (base / "skull_strip" / f"{mod}_skull_strip.nii.gz").write_bytes(b"")
    (base / "segmentation").mkdir(parents=True)
    (base / "segmentation" / "seg_mask.nii.gz").write_bytes(b"")
    (base / "segmentation" / "measured_volumes_in_mm3.json").write_text(
        json.dumps({"Necrotic_NonEnhancing": 1.0, "Enhancing_Core": 2.0, "Edema_Compartment": 3.0})
    )


class BuildCohortTest(unittest.TestCase):
    def test_pre_op_dropped_and_no_forecast_across_surgery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Imaging"
            for name in ("week-000-1", "week-000-2", "week-010", "week-020", "week-030", "week-040"):
                make_scan(root, "Patient-001", name)
            ratings = Path(tmp) / "ratings.csv"
            ratings.write_text(
                "Patient,Date,a,b,Rating,c\n"
                "Patient-001,week-000-1,,,Pre-Op,\n"
                "Patient-001,week-000-2,,,Post-Op,\n"
                "Patient-001,week-010,,,SD,\n"
                "Patient-001,week-020,,,PD,\n"
                "Patient-001,week-030,,,Post-Op,\n"
                "Patient-001,week-040,,,SD,\n"
            )
            cohort = build_cohort(root, ratings)
            self.assertEqual([s.source_name for s in cohort.scans["Patient-001"]],
                             ["week-000-2", "week-010", "week-020", "week-030", "week-040"])
            pairs = [([h.week for h in t.history], t.target.week) for t in cohort.transitions]
            # week 30 is a re-resection: never a target, and it starts a new history segment
            self.assertEqual(pairs, [([0], 10), ([0, 10], 20), ([30], 40)])
            self.assertEqual(cohort.excluded["surgery_targets"], 1)

    def test_patient_folds_are_disjoint_and_complete(self) -> None:
        patients = [f"Patient-{i:03d}" for i in range(23)]
        folds = patient_folds(patients, 5, seed=0)
        flat = [p for f in folds for p in f]
        self.assertEqual(sorted(flat), patients)
        self.assertEqual(folds, patient_folds(patients, 5, seed=0))


if __name__ == "__main__":
    unittest.main()
