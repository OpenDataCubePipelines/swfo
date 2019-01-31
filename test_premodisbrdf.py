#!/usr/bin/env python

"""
test function for pre_modis_brdf computations
"""
import numpy as np
import unittest 
import pre_modis_brdf as premod


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


class BrdfModuleTest(unittest.TestCase): 
    
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

        # test to check if threshold failed idx are replaced with median values
        self.assertEqual(clean_data[keys[0]]['iso_clean'][1, 0], 5.0)
        self.assertEqual(clean_data[keys[9]]['iso_clean'][1, 0], 5.0)
        self.assertEqual(clean_data[keys[8]]['iso_clean'][1, 0], 5.0)


if __name__ == '__main__': 
    
    unittest.main()
