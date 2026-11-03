"""Deterministic LUMIERE week-directory resolution.

LUMIERE stores some study dates as several scans that share one week number, e.g.
``week-000-1`` (pre-operative) and ``week-000-2`` (post-operative). Parsing only the
number makes these collide, and the old loaders silently kept whichever directory
``glob`` returned last (filesystem order). That mixed pre- and post-resection anatomy
into the "week 0" slot for roughly half the cohort.

Rule used everywhere in this repo: when several complete scans share a week number,
keep the one with the highest acquisition suffix (the latest scan of that week). For
week 0 that is the post-operative scan, which is the correct starting point for
follow-up forecasting: the pre-op -> post-op change is the resection, not tumor dynamics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MODALITIES = ("FLAIR", "T1", "T2", "CT1")
MODALITY_FILE_PREFIX = {"FLAIR": "flair", "T1": "t1", "T2": "t2", "CT1": "ct1"}
WEEK_DIR_PATTERN = re.compile(r"^week-(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class WeekScan:
    week: int
    suffix: int
    directory: Path
    modality_paths: dict[str, Path]
    seg_mask_path: Path | None
    volumes_json_path: Path | None
    duplicates: tuple[str, ...]  # names of other directories that shared this week number

    @property
    def source_name(self) -> str:
        return self.directory.name


def parse_week_dir_name(name: str) -> tuple[int, int] | None:
    match = WEEK_DIR_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _scan_files(week_dir: Path) -> tuple[dict[str, Path], Path | None, Path | None]:
    skull_strip = week_dir / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
    paths: dict[str, Path] = {}
    if skull_strip.exists():
        for mod, prefix in MODALITY_FILE_PREFIX.items():
            matches = sorted(skull_strip.glob(f"{prefix}_skull_strip.nii*"))
            if matches:
                paths[mod] = matches[0]
    seg_dir = week_dir / "DeepBraTumIA-segmentation" / "atlas" / "segmentation"
    seg = seg_dir / "seg_mask.nii.gz"
    vols = seg_dir / "measured_volumes_in_mm3.json"
    return paths, (seg if seg.exists() else None), (vols if vols.exists() else None)


def resolve_patient_weeks(patient_dir: Path, require_modalities: bool = True) -> dict[int, WeekScan]:
    """Map week number -> the single scan used for that week (see module docstring)."""
    candidates: dict[int, list[tuple[int, Path]]] = {}
    for week_dir in patient_dir.glob("week-*"):
        if not week_dir.is_dir():
            continue
        parsed = parse_week_dir_name(week_dir.name)
        if parsed is None:
            continue
        candidates.setdefault(parsed[0], []).append((parsed[1], week_dir))

    resolved: dict[int, WeekScan] = {}
    for week, entries in candidates.items():
        entries.sort(key=lambda item: item[0])
        names = [d.name for _, d in entries]
        for suffix, week_dir in reversed(entries):
            paths, seg, vols = _scan_files(week_dir)
            if require_modalities and not all(mod in paths for mod in MODALITIES):
                continue
            resolved[week] = WeekScan(
                week=week,
                suffix=suffix,
                directory=week_dir,
                modality_paths=paths,
                seg_mask_path=seg,
                volumes_json_path=vols,
                duplicates=tuple(n for n in names if n != week_dir.name),
            )
            break
    return dict(sorted(resolved.items()))


SOURCE_MARKER = "registration_source.json"


def write_registration_marker(dest_dir: Path, moving: WeekScan, target: WeekScan) -> None:
    import json

    (dest_dir / SOURCE_MARKER).write_text(
        json.dumps({"moving_source": moving.source_name, "target_source": target.source_name}, indent=2)
    )


def registered_cache_matches_sources(dest_dir: Path, moving: WeekScan | None, target: WeekScan | None) -> bool:
    """True if a cached registration was produced from the scans the resolver now selects.

    Cache entries written before the duplicate-week fix carry no marker. Those are only
    trusted when neither week had duplicate scans (so there was nothing to get wrong).
    """
    import json

    if moving is None or target is None:
        return True  # non-LUMIERE layouts: nothing to validate
    marker = dest_dir / SOURCE_MARKER
    if marker.exists():
        data = json.loads(marker.read_text())
        return data.get("moving_source") == moving.source_name and data.get("target_source") == target.source_name
    return not moving.duplicates and not target.duplicates
