import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import progression_model as pm  # noqa: E402
from lumiere_cohort import COMPARTMENTS  # noqa: E402
from progression_errors import non_pd_reason, pd_reason  # noqa: E402
from progression_robustness import N_TWO, TWO_MONOTONE, two_compartment_features  # noqa: E402


def scan(week, nec, enh, ede, rating=None, source=None):
    return SimpleNamespace(patient_id='p', source_name=source or f'week-{week:03d}', week=week, rating=rating,
                           volumes_mm3=dict(zip(COMPARTMENTS, (nec, enh, ede))))


class TwoCompartmentTest(unittest.TestCase):
    def test_matches_released_features_when_compartments_are_merged(self):
        history = [scan(0, 3000, 2500, 20000, 'Post-Op'), scan(12, 2800, 900, 15000, 'SD')]
        current = scan(24, 4200, 5200, 26000)
        vol = {('p', s.source_name): (s.volumes_mm3['Enhancing_Core'],
                                      s.volumes_mm3['Necrotic_NonEnhancing'] + s.volumes_mm3['Edema_Compartment'])
               for s in history + [current]}
        two = dict(zip(range(N_TWO), two_compartment_features(history, current, vol)))
        full = dict(zip(pm.FEATURES, pm.features(history, current, np.zeros(5))))
        self.assertAlmostEqual(two[2], full['change_vs_previous_enhancing'])
        self.assertAlmostEqual(two[4], full['change_vs_nadir_enhancing'])
        for i, name in ((6, 'total_log1p_ml'), (7, 'total_change_vs_previous'), (8, 'total_change_vs_nadir'),
                        (9, 'interval_years'), (10, 'history_count'), (11, 'years_since_segment_start')):
            self.assertAlmostEqual(two[i], full[name], msg=name)
        self.assertEqual(len(TWO_MONOTONE), N_TWO)


class VolumeRuleTest(unittest.TestCase):
    def test_rule_tracks_the_nadir_fires_only_on_rated_scans_and_falls_back_to_reoperation(self):
        from volume_rule_robustness import rule_week

        def s(week, enh, rating, segment=0):
            return SimpleNamespace(week=week, rating=rating, segment=segment, enh=enh)
        enh = lambda x: x.enh  # noqa: E731
        scans = [s(0, 3000, 'Post-Op'), s(10, 1000, 'SD'), s(20, 1300, 'SD'), s(30, 2500, None), s(40, 2600, 'PD'),
                 s(50, 500, 'Post-Op', 1)]
        cohort = SimpleNamespace(scans={'p': scans})
        # nadir 1000 at week 10; week 30 crosses but is unrated; week 40 (2600 >= 1400 and +1600 mL) fires
        self.assertEqual(rule_week(cohort, 'p', enh, 0.40, 1000.0), 40)
        # +30% relative only at week 20 (1300 vs 1000) with no absolute floor
        self.assertEqual(rule_week(cohort, 'p', enh, 0.25, 0.0), 20)
        # never reaches +5 mL: falls back to the re-operation week
        self.assertEqual(rule_week(cohort, 'p', enh, 0.40, 5000.0), 50)


class ReasonGroupTest(unittest.TestCase):
    def test_pd_reasons(self):
        self.assertEqual(pd_reason('T2-Progr., new non-target L.'), 'new lesion')
        self.assertEqual(pd_reason('Target L.: 13mm x 13mm'), 'measurable lesion growth')
        self.assertEqual(pd_reason('TL 10 x 16 mm'), 'measurable lesion growth')
        self.assertEqual(pd_reason('T2-Progr.'), 'T2/FLAIR progression only')
        self.assertEqual(pd_reason('Non-Target L.: 1'), 'non-measurable / non-target lesion')
        self.assertEqual(pd_reason('T2-Progr., Non-Target L.: 1'), 'non-measurable / non-target lesion')
        self.assertEqual(pd_reason(''), 'other or no text')

    def test_non_pd_reasons(self):
        treat = 'growth attributed to treatment (<3 months / irradiated field)'
        self.assertEqual(non_pd_reason('Less than 3 months, new target L. 11mm x 11mm'), treat)
        self.assertEqual(non_pd_reason('Progression probably within the irradiated area'), treat)
        self.assertEqual(non_pd_reason('New non-Target L. in the high dose field'), treat)
        self.assertEqual(non_pd_reason(''), 'no rationale')
        self.assertEqual(non_pd_reason('with Avastin'), 'other rationale')


if __name__ == '__main__':
    unittest.main()
