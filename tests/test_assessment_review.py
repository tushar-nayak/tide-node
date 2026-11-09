import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from assessment.review import geometry, select_cases


class ReviewTests(unittest.TestCase):
    def test_missing_and_mismatched_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, mask = root/'flair.nii.gz', root/'mask.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5),dtype=np.float32),np.eye(4)),ref)
            affine=np.eye(4); affine[0,3]=8
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5),dtype=np.float32),affine),mask)
            flags,_=geometry({'flair':ref,'mask':mask,'ct1':None})
            self.assertIn('missing_ct1',flags)
            self.assertIn('within_scan_grid_mismatch_mask',flags)

    def test_deterministic_selection_and_patient_diversity(self):
        rows=[{'patient_id':f'Patient-{i//2}', 'visit':f'week-{i}', 'target':'1', 'binary_prediction':'0'} for i in range(10)]
        groups, selected=select_cases(rows,3)
        self.assertEqual(len(groups['false_negative']),10)
        self.assertEqual(len({r['patient_id'] for _,r in selected}),3)
        self.assertEqual(selected,select_cases(list(reversed(rows)),3)[1])
