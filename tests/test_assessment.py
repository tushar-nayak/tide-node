import copy
import json
from pathlib import Path
import unittest

from assessment.report import measure


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((Path(__file__).resolve().parents[1] / 'assessment/example.json').read_text())

    def test_units_and_deferral(self):
        result = measure(self.payload)
        self.assertAlmostEqual(result['measurements']['Enhancing_Core']['current_ml'], 2.6)
        self.assertAlmostEqual(result['measurements']['Enhancing_Core']['change_previous_percent'], 30)
        self.assertEqual(result['assessment']['status'], 'insufficient_evidence')
        self.assertIsNone(result['assessment']['probability_pd'])

    def test_current_label_rejected(self):
        self.payload['history'][-1]['rating'] = 'PD'
        with self.assertRaises(ValueError):
            measure(self.payload)

    def test_invalid_volume_rejected(self):
        for value in [float('nan'), -1, True]:
            payload = copy.deepcopy(self.payload)
            payload['history'][-1]['volumes_mm3']['Enhancing_Core'] = value
            with self.assertRaises(ValueError):
                measure(payload)

    def test_surgery_and_duplicate_time_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload['history'][-1]['rating'] = 'Post-Op'
        with self.assertRaises(ValueError):
            measure(payload)
        self.payload['history'][-1]['week'] = 0
        with self.assertRaises(ValueError):
            measure(self.payload)
