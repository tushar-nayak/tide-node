import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mu_freeze_model import rule_fires  # noqa: E402
from mu_validation import (Patient, Scan, concordance_censored, cox_fit_censored, features,  # noqa: E402
                           label_scans)
from progression_robustness import two_compartment_features  # noqa: E402
from progression_survival import concordance, cox_fit, z_matrix  # noqa: E402
from xlsx_stdlib import read  # noqa: E402


class LabelTest(unittest.TestCase):
    def test_pd_scan_is_closest_within_window_and_later_scans_are_excluded(self):
        days = [90, 150, 240, 250, 400]
        self.assertEqual(label_scans(days, 245), ['non-PD', 'non-PD', 'PD', 'excluded', 'excluded'])  # tie -> earlier
        self.assertEqual(label_scans(days, 300), ['non-PD', 'non-PD', 'non-PD', 'non-PD', 'excluded'])  # no scan in window
        self.assertEqual(label_scans(days, None), ['non-PD'] * 5)
        self.assertEqual(label_scans(days, 95), ['PD', 'excluded', 'excluded', 'excluded', 'excluded'])

    def test_volume_rule(self):
        self.assertTrue(rule_fires([3000, 1000], 2500))  # +150% and +1.5 mL over the 1 mL nadir
        self.assertFalse(rule_fires([3000, 1000], 1900))  # +0.9 mL only
        self.assertFalse(rule_fires([10000], 13000))  # +3 mL but only +30%


class FeatureTest(unittest.TestCase):
    def test_segment_start_is_the_surgery_day_and_the_rest_matches_lumiere_features(self):
        scans = [Scan(1, 70.0, 900.0, 20000.0), Scan(2, 160.0, 600.0, 15000.0), Scan(3, 250.0, 2600.0, 24000.0)]
        p = Patient('p', 'GBM', 0.0, False, None, '0', False, False, None, 300.0, 60.0, scans)
        x = features(p, 2)
        ns = [SimpleNamespace(patient_id='p', source_name=f'tp{s.k}', week=s.day / 7) for s in scans]
        ref = two_compartment_features(ns[:2], ns[2], {('p', f'tp{s.k}'): (s.enh, s.non) for s in scans})
        np.testing.assert_allclose(x[:11], ref[:11])
        self.assertAlmostEqual(x[11], 250 / 7 / 52)  # weeks since surgery / 52, not since the first MU scan
        self.assertAlmostEqual(x[4], np.log1p(2.6) - np.log1p(0.6))  # enhancing change vs nadir


class CensoredSurvivalTest(unittest.TestCase):
    def test_reduces_to_lumiere_functions_without_censoring(self):
        rng = np.random.default_rng(0)
        entry = rng.uniform(0, 5, 30)
        exit_ = entry + rng.uniform(1, 50, 30)
        prog = np.where(rng.random(30) < 0.6, entry + rng.uniform(0, 30, 30), np.nan)
        z = z_matrix(prog, exit_)
        event = np.ones(30, dtype=bool)
        self.assertAlmostEqual(concordance_censored(entry, exit_, event, z), concordance(entry, exit_, z))
        np.testing.assert_allclose(cox_fit_censored(entry, exit_, event, z[None]), cox_fit(entry, exit_, z[None]), rtol=1e-6)

    def test_censored_concordance_matches_brute_force(self):
        rng = np.random.default_rng(1)
        n = 25
        entry = np.zeros(n)
        exit_ = rng.uniform(1, 40, n)
        event = rng.random(n) < 0.6
        prog = np.where(rng.random(n) < 0.5, rng.uniform(0, 30, n), np.nan)
        z = z_matrix(prog, exit_)
        conc = total = 0.0
        for i in range(n):
            if not event[i]:
                continue
            for j in range(n):
                if exit_[j] > exit_[i]:
                    total += 1
                    conc += 1.0 if z[i, i] > z[j, i] else 0.5 if z[i, i] == z[j, i] else 0.0
        self.assertAlmostEqual(concordance_censored(entry, exit_, event, z), conc / total)


class XlsxTest(unittest.TestCase):
    def test_reads_shared_and_numeric_cells_with_gaps(self):
        files = {
            '[Content_Types].xml': '<Types/>',
            'xl/workbook.xml': '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                               'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                               '<sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
            'xl/_rels/workbook.xml.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                          '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
            'xl/sharedStrings.xml': '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                    '<si><t>Patient_ID</t></si><si><t>Days</t></si><si><t>PatientID_0003</t></si></sst>',
            'xl/worksheets/sheet1.xml': '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                                        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="s"><v>1</v></c></row>'
                                        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>286</v></c></row>'
                                        '</sheetData></worksheet>'}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.xlsx'
            with zipfile.ZipFile(path, 'w') as z:
                for name, text in files.items():
                    z.writestr(name, text)
            self.assertEqual(read(path), {'S': [['Patient_ID', '', 'Days'], ['PatientID_0003', '', '286']]})


if __name__ == '__main__':
    unittest.main()
