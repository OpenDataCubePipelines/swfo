#!/usr/bin/env python

"""
test function for pre_modis_brdf computations
"""
import numpy as np
import unittest 
import pre_modis_brdf as premod
import brdf_shape


def gen_test_dataset():
    data_band = {}
    data_qual = {}

    keys = ["2011.01.01", "2011.01.02", "2011.01.03", "2011.01.04", "2011.01.05", "2011.01.06", "2011.01.07", 
            "2011.01.08", "2011.01.09", "2011.01.10"]
    for k in keys:
        data = np.ones(shape=(3, 5, 5), dtype='float32')
        data[:, :, 0] = 5
        data[:, :, 4] = 10
        data[:, 0, :] = np.nan
        data_band[k] = data
        
        dataq = np.zeros(shape=(5, 5), dtype='float32')
        dataq[4, 4] = 1
        dataq[0, :] = np.nan
        data_qual[k] = dataq 

    return data_band, data_qual    


class PreModisBrdfTest(unittest.TestCase):
    
    def test_generate_tile_spatial_stats(self): 
        """ test to checks if spatial statistics of an array are computed correctly """
        data_band, data_qual = gen_test_dataset()
        stats = premod.generate_tile_spatial_stats(data_band)
        
        for key in stats.keys(): 
            self.assertEqual(stats[key]['mean'][0], np.nanmean(data_band[key][0]))
            self.assertEqual(stats[key]['mean'][1], np.nanmean(data_band[key][1]))
            self.assertEqual(stats[key]['mean'][2], np.nanmean(data_band[key][2])) 
            self.assertEqual(stats[key]['std'][0], np.nanstd(data_band[key][0]))
            self.assertEqual(stats[key]['std'][1], np.nanstd(data_band[key][1]))
            self.assertEqual(stats[key]['std'][2], np.nanstd(data_band[key][2]))
 
    def test_get_qualityband_count(self):
        """test to check if band quality counts are computed correctly"""

        data_band, data_qual = gen_test_dataset()
        num_val_pixels = premod.get_qualityband_count(data_qual)
        result = np.full((5, 5), 10, dtype='float32')
        result[0, :] = np.nan
        result[4, 4] = 0.

        self.assertEqual(num_val_pixels.all(), result.all())

    def test_get_threshold(self): 
        """test to check if threshold values are computed correctly"""
        
        data_band, data_qual = gen_test_dataset()
        stats = premod.generate_tile_spatial_stats(data_band)
        threshold_iso, threshold_vol, threshold_geo = premod.get_threshold(stats)
        iso_threshold = np.mean([stats[key]['std'][0] for key in stats.keys()])
        vol_threshold = np.mean([stats[key]['std'][1] for key in stats.keys()])
        geo_threshold = np.mean([stats[key]['std'][2] for key in stats.keys()])

        self.assertEqual(iso_threshold, threshold_iso)
        self.assertEqual(vol_threshold, threshold_vol)
        self.assertEqual(geo_threshold, threshold_geo)

    def test_apply_threshold(self): 
        """test to check if data with failed threshold test are filled with local median value"""
        
        data_band, data_qual = gen_test_dataset()
        keys = [k for k in data_band.keys()]
        num_val_pixels = premod.get_qualityband_count(data_qual)
        min_numpix_required = float(int((10.0/100.) * len(keys)))

        data_band[keys[0]][0][4, 4] = 6.0
        data_band[keys[1]][0][4, 4] = 6.0
        data_band[keys[2]][0][4, 4] = 3.0
        data_band[keys[7]][0][4, 4] = 11.0
        data_band[keys[8]][0][4, 4] = 11.0
        data_band[keys[9]][0][4, 4] = 12.0

        data_band[keys[0]][0][1, 0] = 1.0
        data_band[keys[8]][0][1, 0] = 1.0
        data_band[keys[9]][0][1, 0] = 10.0

        stats = premod.generate_tile_spatial_stats(data_band)
        clean_data = premod.apply_threshold(data_band, 2, stats, num_val_pixels, min_numpix_required)

        # test to check if bad band quality idx are replaced with median values
        self.assertEqual(clean_data[keys[0]]['iso_clean'][4, 4], 6.0)
        self.assertEqual(clean_data[keys[1]]['iso_clean'][4, 4], 6.0)
        self.assertEqual(clean_data[keys[2]]['iso_clean'][4, 4], 6.0)
        self.assertEqual(clean_data[keys[5]]['iso_clean'][4, 4], 10.0)
        self.assertEqual(clean_data[keys[7]]['iso_clean'][4, 4], 11.0)
        self.assertEqual(clean_data[keys[8]]['iso_clean'][4, 4], 11.0)
        self.assertEqual(clean_data[keys[9]]['iso_clean'][4, 4], 11.0)

        # test to check if threshold failed index are replaced with median values
        self.assertEqual(clean_data[keys[0]]['iso_clean'][1, 0], 5.0)
        self.assertEqual(clean_data[keys[9]]['iso_clean'][1, 0], 5.0)
        self.assertEqual(clean_data[keys[8]]['iso_clean'][1, 0], 5.0)

    def test_temporal_average(self):
        """ test to check the result of temporal average computation """

        data_band, data_qual = gen_test_dataset()

        data_band['2012.01.01'] = np.ones(shape=(3, 5, 5), dtype='float32')
        data_qual['2012.01.01'] = np.zeros(shape=(5, 5), dtype='float32')

        keys = [k for k in data_band.keys()]

        data_band[keys[0]][0][1, 1] = 12.0
        data_band[keys[1]][0][1, 1] = 12.0
        data_band[keys[2]][0][1, 1] = 12.0
        data_band[keys[7]][0][1, 1] = 12.0
        data_band[keys[8]][0][1, 1] = 12.0
        data_band[keys[9]][0][1, 1] = 12.0
        data_band[keys[10]][0][0, 0] = 6.0
        data_band[keys[10]][0][1, 0] = 2.5

        num_val_pixels = premod.get_qualityband_count(data_qual)
        min_numpix_required = float(int((10.0/100.) * len(keys)))

        stats = premod.generate_tile_spatial_stats(data_band)
        clean_data = premod.apply_threshold(data_band, 2, stats, num_val_pixels, min_numpix_required)

        monthly_mean = premod.temporal_average(clean_data, tag="Monthly")
        daily_mean = premod.temporal_average(clean_data, tag="Daily")
        yearly_mean = premod.temporal_average(clean_data, tag="Yearly")

        self.assertEqual(monthly_mean[1]['iso_clean_mean'][1, 1], 8.0)
        self.assertEqual(daily_mean[1]['iso_clean_mean'][1, 0], 3.75)
        self.assertEqual(yearly_mean[2011]['iso_clean_mean'][1, 1], 7.6)

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
        """ tests computation of rms indices"""
        alpha1 = alpha2 = np.array([1, 2, 3, 4, 5])
        rms_val = np.sqrt((0.015683596 * alpha1**2) + (2 * 0.055165295 * alpha1 * alpha2) + (0.371423479 * alpha2**2))
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)
        self.assertEqual(rms_val.all(), rms.all())

    def test_get_afx_indices(self):
        """ tests computation of afx indices"""
        alpha1 = alpha2 = np.array([1, 2, 3, 4, 5])
        afx_val = 1 + 0.189184 * alpha1 + (-1.377622 * alpha2)
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        self.assertEqual(afx_val.all(), afx.all())

    def test_shape_param(self):
        """ tests computation of brdf shape function using lognormal model"""
        fiso = np.ones((3, 5, 5), dtype='int16') * 100
        fvol = np.ones((3, 5, 5), dtype='int16') * 50
        fgeo = np.ones((3, 5, 5), dtype='int16') * 10
        cov_iso = np.std(fiso, axis=0)/np.mean(fiso, axis=0)

        alpha1, alpha2 = brdf_shape.get_mean_shape_param(fiso, fvol, fgeo, cov_iso)

        self.assertEqual(alpha1.all(), (np.ones((5, 5), dtype='float32') * 0.5).all())
        self.assertEqual(alpha2.all(), (np.ones((5, 5), dtype='float32') * 0.1).all())


if __name__ == '__main__':
    
    unittest.main()
