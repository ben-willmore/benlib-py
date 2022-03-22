# pylint: disable=invalid-name

import unittest
import numpy as np
import utils
import strf

class TestSTRFs(unittest.TestCase):

    def test_tensorize(self):
        X_ft = np.ones((5, 6))
        X_fht = strf.tensorize(X_ft, 2)
        self.assertTrue(X_fht.shape == (5, 2, 6))
        self.assertTrue(X_fht.sum() == 55)

    def disabled_matlab(self):
        import matlab.engine
        eng = matlab.engine.start_matlab()
        X_fht_mat = np.array(eng.tensorize(matlab.double(X_ft.tolist()), 15))

        assert(X_fht.shape==X_fht_mat.shape)
        assert((X_fht == X_fht_mat).all())

    def test_reconstruct_gaps_t(self):
        X = np.array([1, 1, 2, 2, 3, 3])
        segment_lengths = [2, 2, 2]
        X_full = strf.reconstruct_gaps_t(X, segment_lengths, 2)
        self.assertTrue(X_full.shape == (9,))

if __name__ == '__main__':
    unittest.main()
