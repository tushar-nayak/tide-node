import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from release_audit import paired_effect, validate_folds


class PairedAuditTest(unittest.TestCase):
    def rows(self):
        return [{'patient_id': 'a', 'case': str(i), 'score': 1.0} for i in range(5)] + [{'patient_id': 'b', 'case': '0', 'score': 3.0}]

    def effect(self, a, b):
        return paired_effect(a, b, ['patient_id', 'case'], ['score'], 500, 42)

    def test_equal_patient_weight_and_pairing(self):
        a = self.rows()
        b = [{**r, 'score': 0.0} for r in reversed(a)]
        e = self.effect(a, b)
        self.assertEqual(e['difference'], 2.0)  # not row mean 4/3
        self.assertEqual(e, self.effect(list(reversed(a)), b))
        self.assertEqual(e['ci95'], [1.0, 3.0])

    def test_identical_models_have_zero_interval(self):
        e = self.effect(self.rows(), self.rows())
        self.assertEqual(e['ci95'], [0.0, 0.0])

    def test_duplicates_rejected(self):
        a = self.rows()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.effect(a + [a[0]], a)

    def test_unpaired_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            self.effect(self.rows(), self.rows()[:-1])

    def test_nonfinite_rejected(self):
        for bad in [np.nan, np.inf, -np.inf]:
            a = self.rows()
            a[0]['score'] = bad
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.effect(a, self.rows())

    def test_missing_fold_and_repeat_rejected(self):
        rows = [{'repeat': '0', 'fold': '0'}]
        with self.assertRaisesRegex(ValueError, 'folds'):
            validate_folds(rows, ['a', 'b'], 2, 1, {})
        with self.assertRaisesRegex(ValueError, 'repeats'):
            validate_folds(rows, ['a', 'b'], 1, 2, {})

    def test_wrong_fold_and_missing_case_rejected(self):
        from lumiere_cohort import patient_folds
        ps = patient_folds(['a', 'b'], 2, 0)
        rows = [{'repeat': '0', 'fold': str(f), 'patient_id': pid,
                 'last_source': 'x', 'target_source': 'y'} for f, group in enumerate(ps) for pid in group]
        expected = {(p, 'x', 'y'): None for p in ['a', 'b']}
        validate_folds(rows, ['a', 'b'], 2, 1, expected)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_folds(rows, ['a', 'b'], 2, 1, {**expected, ('a', 'y', 'z'): None})
        rows[0]['fold'], rows[1]['fold'] = rows[1]['fold'], rows[0]['fold']
        with self.assertRaisesRegex(ValueError, 'fold mismatch'):
            validate_folds(rows, ['a', 'b'], 2, 1, expected)


if __name__ == '__main__':
    unittest.main()
