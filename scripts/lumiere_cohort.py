"""Cohort definition shared by the volume-trajectory (step 1) and mask-forecast (step 2) experiments.

Rules (all derived from LUMIERE's own expert RANO table, LUMIERE-ExpertRating-v202211.csv):

* One scan per week number, chosen by ``lumiere_weeks.resolve_patient_weeks``.
* Scans rated ``Pre-Op`` are dropped: the pre-op -> post-op change is a resection, not
  tumor dynamics, and no imaging-only model can forecast a surgery.
* A scan rated ``Post-Op`` (including later re-resections) starts a new *segment*.
  History never crosses a segment boundary, and a transition whose target is a Post-Op
  scan is never used (surgery happened inside that interval).
* Scans with no rating row are treated as ordinary follow-up scans.

A *transition* is (history scans within one segment, next scan in that segment).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from lumiere_weeks import WeekScan, resolve_patient_weeks

COMPARTMENTS = ("Necrotic_NonEnhancing", "Enhancing_Core", "Edema_Compartment")
COMPARTMENT_SHORT = ("necrosis", "enhancing", "edema")


@dataclass
class CohortScan:
    patient_id: str
    week: int
    scan: WeekScan
    rating: str | None
    segment: int
    volumes_mm3: dict[str, float] | None = None

    @property
    def source_name(self) -> str:
        return self.scan.source_name


@dataclass
class Transition:
    patient_id: str
    history: list[CohortScan]
    target: CohortScan

    @property
    def last(self) -> CohortScan:
        return self.history[-1]

    @property
    def dt_weeks(self) -> int:
        return self.target.week - self.last.week


@dataclass
class Cohort:
    scans: dict[str, list[CohortScan]] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def patients(self) -> list[str]:
        return sorted({t.patient_id for t in self.transitions})


def load_ratings(path: Path) -> dict[tuple[str, str], str]:
    ratings: dict[tuple[str, str], str] = {}
    if not path.exists():
        return ratings
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                ratings[(row[0].strip(), row[1].strip())] = row[4].strip()
    return ratings


def load_clinical(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["Patient"].strip(): row for row in csv.DictReader(handle)}


def build_cohort(
    imaging_root: Path,
    ratings_csv: Path,
    require_volumes: bool = True,
    require_seg_mask: bool = False,
    min_history: int = 1,
) -> Cohort:
    ratings = load_ratings(ratings_csv)
    cohort = Cohort()
    counts = {"pre_op_scans": 0, "missing_volumes": 0, "missing_seg": 0, "surgery_targets": 0}

    for patient_dir in sorted(imaging_root.glob("Patient-*")):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        kept: list[CohortScan] = []
        segment = 0
        for week, scan in resolve_patient_weeks(patient_dir).items():
            rating = ratings.get((pid, scan.source_name))
            norm = (rating or "").replace(" ", "")
            if norm == "Pre-Op":
                counts["pre_op_scans"] += 1
                continue
            if require_seg_mask and scan.seg_mask_path is None:
                counts["missing_seg"] += 1
                continue
            volumes = None
            if scan.volumes_json_path is not None:
                volumes = {k: float(v) for k, v in json.loads(scan.volumes_json_path.read_text()).items()}
            if require_volumes and (volumes is None or not all(c in volumes for c in COMPARTMENTS)):
                counts["missing_volumes"] += 1
                continue
            if norm.startswith("Post-Op") and kept:
                segment += 1
            kept.append(CohortScan(pid, week, scan, rating, segment, volumes))
        cohort.scans[pid] = kept

        for idx in range(1, len(kept)):
            target = kept[idx]
            if (target.rating or "").replace(" ", "").startswith("Post-Op"):
                counts["surgery_targets"] += 1
                continue
            history = [s for s in kept[:idx] if s.segment == target.segment]
            if len(history) < min_history:
                continue
            cohort.transitions.append(Transition(pid, history, target))

    cohort.excluded = counts
    return cohort


def patient_folds(patients: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """Deterministic patient-level folds (no patient appears in two folds)."""
    import random

    order = sorted(patients)
    random.Random(seed).shuffle(order)
    return [order[i::n_folds] for i in range(n_folds)]
