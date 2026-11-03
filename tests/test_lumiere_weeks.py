from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.lumiere_weeks import (
    registered_cache_matches_sources,
    resolve_patient_weeks,
    write_registration_marker,
)


def make_scan(patient_dir: Path, name: str, modalities=("flair", "t1", "t2", "ct1")) -> None:
    skull = patient_dir / name / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
    skull.mkdir(parents=True)
    for mod in modalities:
        (skull / f"{mod}_skull_strip.nii.gz").write_bytes(b"")


class ResolvePatientWeeksTest(unittest.TestCase):
    def test_duplicate_week_picks_latest_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "Patient-001"
            for name in ("week-000-2", "week-000-1", "week-044"):
                make_scan(patient, name)
            weeks = resolve_patient_weeks(patient)
            self.assertEqual(list(weeks), [0, 44])
            self.assertEqual(weeks[0].source_name, "week-000-2")
            self.assertEqual(weeks[0].duplicates, ("week-000-1",))
            self.assertEqual(weeks[44].duplicates, ())

    def test_incomplete_latest_scan_falls_back_to_earlier_complete_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "Patient-002"
            make_scan(patient, "week-000-1")
            make_scan(patient, "week-000-2", modalities=("flair", "t1"))
            weeks = resolve_patient_weeks(patient)
            self.assertEqual(weeks[0].source_name, "week-000-1")

    def test_cache_marker_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "Patient-003"
            for name in ("week-000-1", "week-000-2", "week-010"):
                make_scan(patient, name)
            weeks = resolve_patient_weeks(patient)
            dest = Path(tmp) / "cache"
            dest.mkdir()
            # Legacy cache (no marker) for a duplicated week is not trusted.
            self.assertFalse(registered_cache_matches_sources(dest, weeks[0], weeks[10]))
            write_registration_marker(dest, weeks[0], weeks[10])
            self.assertTrue(registered_cache_matches_sources(dest, weeks[0], weeks[10]))
            # Legacy cache for weeks without duplicates is trusted.
            self.assertTrue(registered_cache_matches_sources(Path(tmp) / "nomarker", weeks[10], weeks[10]))


if __name__ == "__main__":
    unittest.main()
