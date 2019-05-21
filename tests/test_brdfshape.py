#!/usr/bin/env python

"""
test function for pre_modis_brdf computations
"""
import numpy as np
import unittest 
from swfo import brdf_shape


class BrdfShape(unittest.TestCase):
    """ this class handles all testing of computation of 
    BRDF shape functions.
    """

    def test_constants(self):
        """ test to check if constants are not changed"""
        constants = brdf_shape.CONSTANTS

        self.assertEqual(0.015683596, constants['c11'])
        self.assertEqual(0.055165295, constants['c12'])
        self.assertEqual(0.371423479, constants['c22'])
        self.assertEqual(0.189184, constants['gvol'])
        self.assertEqual(-1.377622, constants['ggeo'])
        self.assertEqual(0.037839423386594996, constants['ra'])
        self.assertEqual(-0.07706873100489496, constants['rb'])
        self.assertEqual(0.1957082863758389, constants['rc'])

    def test_get_rms_indices(self):
        """ tests computation of rms index"""
        alpha1 = alpha2 = np.array([1, 2, 3, 4, 5])
        rms_val = np.sqrt((0.015683596 * alpha1**2) + (2 * 0.055165295 * alpha1 * alpha2) + (0.371423479 * alpha2**2))
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)
        np.testing.assert_array_equal(rms_val, rms)

    def test_get_afx_indices(self):
        """ tests computation of afx index"""
        alpha1 = alpha2 = np.array([1, 2, 3, 4, 5])
        afx_val = 1 + 0.189184 * alpha1 + (-1.377622 * alpha2)
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        np.testing.assert_array_equal(afx_val, afx)

    def test_mean_shape_param(self):
        """ tests computation of brdf shape function using lognormal model"""
        fiso = np.ones((3, 5, 5), dtype='int16') * 100
        fvol = np.ones((3, 5, 5), dtype='int16') * 50
        fgeo = np.ones((3, 5, 5), dtype='int16') * 10
        cov_iso = np.std(fiso, axis=0)/np.mean(fiso, axis=0)

        alpha1, alpha2 = brdf_shape.get_mean_shape_param(np.mean(fiso, axis=0),
                                                         np.mean(fvol, axis=0),
                                                         np.mean(fgeo, axis=0), cov_iso)
        np.testing.assert_array_equal(alpha1, (np.ones((5, 5), dtype='float64') * 0.5))
        np.testing.assert_array_equal(alpha2, (np.ones((5, 5), dtype='float64') * 0.1))

    def test_cov_shape_param(self):
        """ tests computation of coefficients of shape parameters"""
        alpha1_cov, alpha2_cov = brdf_shape.get_cov_shape_param(0., 0., 0.)
        self.assertEqual(alpha2_cov, 0.0)
        self.assertEqual(alpha1_cov, 0.0)
        alpha1_cov, alpha2_cov = brdf_shape.get_cov_shape_param(1., 1., 1.)
        self.assertEqual(alpha1_cov, np.sqrt(3))
        self.assertEqual(alpha2_cov, np.sqrt(3))

    def test_std_afx_indices(self):
        """ tests computation of standard deviation of afx index"""
        afx_std = brdf_shape.get_std_afx_indices(1.0, 1.0, 0.0, 0.0)
        self.assertEqual(afx_std, 0.0)
        afx_std = brdf_shape.get_std_afx_indices(2.0, 1.0, 1.0, 1.0)
        self.assertEqual(afx_std, 1.4286373641718879)

    def test_std_rms_indices(self):
        """ tests computation of standard deviation of rms index"""
        rms_std = brdf_shape.get_std_rms_indices(1.0, 1.0, 1.0, 1.0, 1.0)
        rms_val = np.sqrt((0.015683596 + 0.055165295)**2 + (0.055165295 + 0.371423479)**2)
        self.assertEqual(rms_std, rms_val)

    def test_get_unfeasible_mask(self):
        """ tests generation of infeasibility index mask"""
        rms = np.array([1, 2, 3, 4, 5])
        afx = np.array([10, 2, 3, 4, 50])
        mask = brdf_shape.get_unfeasible_mask(rms, afx)
        np.testing.assert_array_equal(mask, np.array([True, False, False, False,True]))


if __name__ == '__main__':
    unittest.main()
