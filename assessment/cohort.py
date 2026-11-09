"""Assessment-specific discovery: no forecasting transitions or image cache selection."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from .features import features, RATINGS
from .report import COMPARTMENTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lumiere_weeks import parse_week_dir_name  # audited filename parser only


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(imaging, ratings_path):
    ratings = {}
    with Path(ratings_path).open(newline='') as f:
        for row in list(csv.reader(f))[1:]:
            key = tuple(row[:2])
            ratings.setdefault(key, []).append(row[4].strip())  # deliberately never parse flags, rationale or survival
    rows, excluded, sources = [], [], []
    def exclude(pid, visit, reason):
        excluded.append(dict(patient_id=pid, visit=visit, reason=reason))
    for patient in sorted(Path(imaging).glob('Patient-*')):
        by_week = {}
        for path in sorted(patient.glob('week-*')):
            parsed = parse_week_dir_name(path.name)
            if parsed and path.is_dir():
                by_week.setdefault(parsed[0], []).append((parsed[1], path))
        segment, history = 0, []
        for week, entries in sorted(by_week.items()):
            entries.sort()
            path = entries[-1][1]
            pid, visit = patient.name, path.name
            labels = [r for _, p in entries for r in ratings.get((pid, p.name), [])]
            current_labels = ratings.get((pid, visit), [])
            unique_labels = set(current_labels)
            ambiguous = len(unique_labels) > 1
            surgery = any(r.replace(' ', '').startswith('Post-Op') for r in labels)
            preop = 'Pre-Op' in unique_labels and not surgery
            # Boundaries occur BEFORE volume validation. An invalid postoperative scan still resets history.
            if surgery or preop:
                history = []
                segment += 1
            for _, discarded in entries[:-1]:
                exclude(pid, discarded.name, 'duplicate_week_lower_suffix')
            rating = current_labels[0] if len(unique_labels) == 1 else ''
            if len(current_labels) > 1:
                exclude(pid, visit, 'conflicting_duplicate_rating' if ambiguous else 'identical_duplicate_rating_collapsed')
            volume_path = path / 'DeepBraTumIA-segmentation/atlas/segmentation/measured_volumes_in_mm3.json'
            record = dict(patient_id=pid, visit=visit, week=week, segment=segment,
                          rating=rating, measurement_path=str(volume_path),
                          measurement_sha256=sha(volume_path) if volume_path.exists() else None,
                          same_week_ratings=labels, current_rating_rows=current_labels)
            sources.append(record)
            if preop:
                exclude(pid, visit, 'preoperative')
                continue
            try:
                raw = json.loads(volume_path.read_text())
                volumes = {c: raw[c] for c in COMPARTMENTS}
                # Validate a single scan via duplicate synthetic earlier measurement; not used as a feature.
                from .report import number
                for c, value in volumes.items():
                    number(value, c)
            except (OSError, KeyError, ValueError, TypeError) as error:
                exclude(pid, visit, 'invalid_or_missing_volumes: ' + str(error))
                continue
            scan = dict(week=week, volumes_mm3=volumes)
            if history and not surgery and rating in RATINGS:
                payload = dict(version=1, patient_id=pid, history=history + [scan])
                features(payload)
                previous = history[-1].get('rating')
                rows.append(dict(patient_id=pid, visit=visit, week=week, segment=segment,
                                 target=int(rating == 'PD'), target_rating=rating,
                                 previous_rating=previous,
                                 previous_available=previous in RATINGS,
                                 new_progression_eligible=previous in ('SD', 'PR', 'CR'),
                                 payload=payload))
            else:
                exclude(pid, visit, 'surgery_visit' if surgery else
                        'no_prior_valid_scan_in_segment' if not history else 'no_supervised_current_rating')
            # Current label lives only in targets, then becomes legitimate historical information for later visits.
            history.append(dict(scan, rating='Post-Op' if surgery else rating or 'unknown'))
    return rows, excluded, sources
