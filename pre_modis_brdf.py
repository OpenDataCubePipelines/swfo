#!/usr/bin/env python

"""
BRDF data extraction utilities from hdf5 files
"""

import os
from os.path import join as pjoin
import h5py
import fnmatch
import datetime
import numpy as np
import json


SDS_ATTRS_PREFIX = {'crs_wkt': 'crs_wkt',
                    'geotransform': 'geotransform',
                    'no_data': 'nodata',
                    'offset': 'offsets',
                    'scale': 'scales'}

SDS_BAND_NAME = {'BAND1': 'BRDF_Albedo_Parameters_Band1',
                 'BAND2': 'BRDF_Albedo_Parameters_Band2',
                 'BAND3': 'BRDF_Albedo_Parameters_Band3',
                 'BAND4': 'BRDF_Albedo_Parameters_Band4',
                 'BAND5': 'BRDF_Albedo_Parameters_Band5',
                 'BAND6': 'BRDF_Albedo_Parameters_Band6',
                 'Band7': 'BRDF_Albedo_Parameters_Band7',
                 'QUAL_BAND1': 'BRDF_Albedo_Band_Mandatory_Quality_Band1',
                 'QUAL_BAND2': 'BRDF_Albedo_Band_Mandatory_Quality_Band2',
                 'QUAL_BAND3': 'BRDF_Albedo_Band_Mandatory_Quality_Band3',
                 'QUAL_BAND4': 'BRDF_Albedo_Band_Mandatory_Quality_Band4',
                 'QUAL_BAND5': 'BRDF_Albedo_Band_Mandatory_Quality_Band5',
                 'QUAL_BAND6': 'BRDF_Albedo_Band_Mandatory_Quality_Band6',
                 'QUAL_Band7': 'BRDF_Albedo_Band_Mandatory_Quality_Band7'
                 }
TILE = ['h29v10', 'h30v10', 'h31v10', 'h32v10', 'h27v11', 'h28v11', 'h29v11', 'h30v11',
        'h31v11', 'h27v12', 'h28v12', 'h29v12', 'h30v12', 'h31v12', 'h28v13', 'h29v13']

BRDF_PARAM = {'iso': 0, 'vol': 1, 'geo': 2}


class JsonEncoder(json.JSONEncoder):
    """
    A wrapper class to address the issue of json encoding error
    This class handles  the json serializing error for numpy
    datatype: 'float32', datetime and numpy arrays
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        else:
            raise OSError('Json Encoding Error')


def get_h5_info(brdf_dir=None):
    """
    A function to extract all the MODIS BRDF acquisition details
    from the root folder where BRDF data are stored.

    :param brdf_dir:
        A path name to where hdf5 formatted BRDF data are stored
        The BRDF directories are assumed to be yyyy.mm.dd naming convention.

    :return:
        a nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name

    """
    brdf_data = {}
    folder_fmt = '%Y.%m.%d'

    for item in os.listdir(brdf_dir):
        files = [f for f in os.listdir(pjoin(brdf_dir, item))]

        h5_names = {}

        for tile in TILE:
            try:
                filename = fnmatch.filter(files, '*{t}*.h5'.format(t=tile))[0]
            except IndexError:
                filename = None
            if filename:
                file_path = pjoin(brdf_dir, item, filename)
                h5_names[tile] = file_path
        try:
            h5_names['datetime'] = datetime.datetime.strptime(item, folder_fmt)
        except ValueError:
            h5_names['datetime'] = None

        brdf_data[item] = h5_names

    return brdf_data


def generate_tile_spatial_stats(data_dict):
    """
    A function to perform base brdf parameter data check by
    computing mean, standard deviation and coefficient of variation
    across whole spatial (x, y dimension) for each time step from a
    list of data.

    This function assumes data in a list is a spatial 2 dims (x, y)
    data
    """

    stats = {}

    for key in data_dict:
        temp = {}
        iso = data_dict[key][0]
        vol = data_dict[key][1]
        geo = data_dict[key][2]

        temp['mean'] = (np.nanmean(iso), np.nanmean(vol), np.nanmean(geo))
        temp['std'] = (np.nanstd(iso), np.nanstd(vol), np.nanstd(geo))
        temp['cov'] = ((temp['std'][0] / temp['mean'][0]) * 100,
                       (temp['std'][1] / temp['mean'][1]) * 100,
                       (temp['std'][2] / temp['mean'][2]) * 100)

        stats[key] = temp

    return stats


def get_sds_doy_dataset(dirs=None, dayofyear=None, sds_name=None, year=None, tile=None, apply_scale=False):
    """
    returns a list of numpy array of specific sds data
    for particular day for range of years( for all years > given year)
    from all the hdf5 files in a given directory
    """

    h5_info = get_h5_info(brdf_dir=dirs)
    # print(json.dumps(h5_info, cls=JsonEncoder, indent=4))

    keys = np.sort([k for k in h5_info])

    data_dict = {}

    for k in keys:

        dt = h5_info[k]['datetime']
        doy = dt.timetuple().tm_yday
        yr = dt.timetuple().tm_year

        if doy < dayofyear and yr > year:

            h5_file = (h5_info[k][tile])
            print(h5_file)
            with h5py.File(h5_file, 'r') as fid:
                for band in fid:
                    if band == sds_name:
                        ds = fid[band]
                        sds_data = ds[:]
                        if len(ds.shape) == 3:
                            scale_factor = float(ds.attrs['scale_factor'])
                            add_offset = float(ds.attrs['add_offset'])
                            nodata = float(ds.attrs['_FillValue'])
                            sds_data = sds_data.astype('float32')
                            sds_data[sds_data == float(nodata)] = np.nan

                        else:
                            scale_factor = 1.0
                            add_offset = 0.0
                            nodata = float(ds.attrs['_FillValue'])
                            sds_data = sds_data.astype('float32')
                            sds_data[sds_data == float(nodata)] = np.nan

                        if apply_scale:
                            sds_data = sds_data * scale_factor + add_offset

                        data_dict[k] = sds_data
    # print(json.dumps(data_dict, cls=JsonEncoder, indent=4))
    return data_dict


def get_qualityband_count(quality_data=None):
    """
    this function computes and returns the number of valid pixels
    in a time series stack.Parameter 'quality_data' is a dict, which
    contains numpy masked array (x, y dim) for each key.

    Computation here is little convoluted, since quality data are
    valid for 0 and invalid for for 1 integer values. so, first
    masked sum array operation is carried out for whole stacked
    and then total number of time series is subtracted to obtain
    the num of valid pixels
    """

    data = np.array([quality_data[key] for key in quality_data.keys()])
    data_sum = (np.nansum(data, axis=0))
    data_sum[np.all(np.isnan(data), axis=0)] = np.nan
    num_val_pixels = len(quality_data) - data_sum
    return np.array(num_val_pixels)


def get_threshold(data_dict):
    """
    A 'data_dict' contains spatial stats of each tile across different time
    A function to compute threshold based on the temporal average from each tile
    threshold is the time series average of the standard deviation of each tile
    """
    threshold_iso = np.mean([data_dict[key]['std'][0] for key in data_dict.keys()])
    threshold_vol = np.mean([data_dict[key]['std'][1] for key in data_dict.keys()])
    threshold_geo = np.mean([data_dict[key]['std'][2] for key in data_dict.keys()])

    return threshold_iso, threshold_vol, threshold_geo


def apply_threshold(data, filter_size, spatial_stats, quality_count, min_numpix_required):
    """
    This function applies median filter on the dataset, median filter with size of
    (2 * filter_size + 1) is applied as a running median filter with time steps centered
    the data value. For example, if time step included day of year [001, 002,...009] then
    median filer would consists of all dataset if filter_size is 4 and center time step
    would be day 005. For the edge cases, the filter size would only encompass the all
    available time step within the defined filter size. For example, if data were to be filtered
    for day 008 for [001, 002,...009] then the median filter would only encompass data from
    day [004, 005, 006, 007, 008 and 009] since day 010, 011, 012 are missing.

    """
    keys = [k for k in data.keys()]

    # get the data iso, vol and go from data which is a dict for all the keys and convert to numpy array
    data_iso = np.ma.array([np.ma.masked_invalid(data[key][0]) for key in keys])
    data_vol = np.ma.array([np.ma.masked_invalid(data[key][1]) for key in keys])
    data_geo = np.ma.array([np.ma.masked_invalid(data[key][2]) for key in keys])

    # get threshold values 
    iso_threshold, vol_threshold, geo_threshold = get_threshold(spatial_stats)

    # print(iso_threshold, vol_threshold, geo_threshold)

    # get the index where band_quality number is less the minimum number of valid pixels required
    bad_idx = np.where(quality_count < min_numpix_required)

    # print(bad_idx)
    # print(data_iso.mask)

    # set indexes to compute temporal median filtering of the dataset as defined
    # by a filter size with time step centered around data value
    data_clean = {}
    for i in range(len(keys)):
        temp = {}

        # set the index's for the filter median filters
        start_idx = i - filter_size
        end_idx = i + filter_size + 1
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(keys)-1:
            end_idx = len(keys)

        # print(start_idx, end_idx)

        # extract the value for a key and mask invalid data
        iso_clean = np.ma.masked_invalid(data_iso[i])
        vol_clean = np.ma.masked_invalid(data_vol[i])
        geo_clean = np.ma.masked_invalid(data_geo[i])

        # print(keys[i])
        # print(keys[start_idx:end_idx])
        # print('iso')
        # print(data_iso[start_idx:end_idx])

        # get temporal local median value as set by filter size
        iso_local_median = np.ma.median(data_iso[start_idx:end_idx], axis=0)
        vol_local_median = np.ma.median(data_vol[start_idx:end_idx], axis=0)
        geo_local_median = np.ma.median(data_geo[start_idx:end_idx], axis=0)

        # print('median')
        # print(iso_local_median)

        # apply threshold test to clean the data set
        threshold_iso_idx = np.ma.where(abs(iso_local_median - iso_clean) > iso_threshold)
        threshold_vol_idx = np.ma.where(abs(vol_local_median - vol_clean) > vol_threshold)
        threshold_geo_idx = np.ma.where(abs(geo_local_median - geo_clean) > geo_threshold)

        # print(threshold_iso_idx)
        # print(iso_clean.shape)

        # replace the data which did not pass threshold test with temporal local median value
        iso_clean[threshold_iso_idx] = iso_local_median[threshold_iso_idx]
        vol_clean[threshold_vol_idx] = vol_local_median[threshold_vol_idx]
        geo_clean[threshold_geo_idx] = geo_local_median[threshold_geo_idx]

        # replace the data which were initially filled with running median with local median
        iso_clean[bad_idx] = iso_local_median[bad_idx]
        vol_clean[bad_idx] = iso_local_median[bad_idx]
        geo_clean[bad_idx] = iso_local_median[bad_idx]

        # print(iso_clean[2200:2210, 2200:2210])
        # print(iso_local_median[2200:2210, 2200:2210])
        # print('clean')
        # print(iso_clean)
        
        temp['iso_clean'] = iso_clean
        temp['vol_clean'] = vol_clean
        temp['geo_clean'] = geo_clean

        data_clean[keys[i]] = temp

    return data_clean


def temporal_average(data):
    """
    This function computes temporal average of data sets for same day of year (doy),
    same month (monthly), and same year(yearly).
    """
    keys = np.array([k for k in data.keys()])

    key_fmt = '%Y.%m.%d'
    dt_list = [datetime.datetime.strptime(item, key_fmt) for item in data.keys()]

    set_doy = set([dt.timetuple().tm_yday for dt in dt_list])
    set_mnt = set([dt.timetuple().tm_mon for dt in dt_list])
    set_yr = set([dt.timetuple().tm_year for dt in dt_list])

    daily_mean, monthly_mean, yearly_mean = {}, {}, {}

    for d in set_doy:
        tmp = {}
        idx_doy = np.argwhere(np.array([dt.timetuple().tm_yday for dt in dt_list]) == d)
        tmp['iso_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['iso_clean'] for idx in idx_doy]), axis=0)
        tmp['vol_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['vol_clean'] for idx in idx_doy]), axis=0)
        tmp['geo_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['geo_clean'] for idx in idx_doy]), axis=0)
        daily_mean[d] = tmp

    for m in set_mnt:
        tmp = {}
        idx_mnt = np.argwhere(np.array([dt.timetuple().tm_mon for dt in dt_list]) == m)
        tmp['iso_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['iso_clean'] for idx in idx_mnt]), axis=0)
        tmp['vol_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['vol_clean'] for idx in idx_mnt]), axis=0)
        tmp['geo_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['geo_clean'] for idx in idx_mnt]), axis=0)
        monthly_mean[m] = tmp

    for y in set_yr:
        tmp = {}
        idx_yr = np.argwhere(np.array([dt.timetuple().tm_year for dt in dt_list]) == y)
        tmp['iso_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['iso_clean'] for idx in idx_yr]), axis=0)
        tmp['vol_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['vol_clean'] for idx in idx_yr]), axis=0)
        tmp['geo_mean'] = np.ma.mean(np.ma.array([data[keys[idx][0]]['geo_clean'] for idx in idx_yr]), axis=0)
        yearly_mean[y] = tmp

    return daily_mean, monthly_mean, yearly_mean


def main(brdf_dir=None, band=None, apply_scale=None, doy=None, year_from=None, tile=None):

    pthresh = 10
    filter_size = 4

    sds_databand_name = SDS_BAND_NAME['{b}'.format(b=band)]
    sds_qualityband_name = SDS_BAND_NAME['QUAL_{b}'.format(b=band)]

    # get a list data sets for given tile for particular date and tile
    sds_data = get_sds_doy_dataset(dirs=brdf_dir, dayofyear=doy, sds_name=sds_databand_name,
                                   year=year_from, tile=tile, apply_scale=apply_scale)

    qual_data = get_sds_doy_dataset(dirs=brdf_dir, dayofyear=doy, sds_name=sds_qualityband_name,
                                    year=year_from, tile=tile, apply_scale=apply_scale)

    # spatial stats required to set the threshold value
    tile_spatial_stats = generate_tile_spatial_stats(sds_data)
    # print(json.dumps(tile_spatial_stats, cls=JsonEncoder, indent=4))

    # generate number of valid pixel count required for analysis in a time series stack
    # as defined in David Jubb's BRDF document
    min_numpix_required = float(int((pthresh / 100.0) * len(sds_data)))
    # print(type(min_numpix_required))

    # get counts of good pixel quality
    quality_count = get_qualityband_count(quality_data=qual_data)
    # quality_count[2209, 2209] = -111.
    # print(quality_count[2200:2210, 2200:2210])

    quality_count = np.ma.masked_invalid(quality_count)

    data_clean = apply_threshold(sds_data, filter_size, tile_spatial_stats, quality_count, min_numpix_required)

    # compute daily, monthly and yearly mean from clean data sets
    daily_mean, monthly_mean, yearly_mean = temporal_average(data_clean)


if __name__ == "__main__":
    # brdf_dir = '/g/data/u46/users/pd1813/BRDF_PARAM/MCD43A1_C6_HDF5_TILE_DATASET/'
    brdf_dir = '/g/data/u46/users/ia1511/Work/data/brdf-collection-6/reprocessed/'
    band = "BAND1"
    tile = 'h29v10'
    apply_scale = True
    doy = 10  # subset for which doy to be processed
    year_from = 2002  # subset from which year to be processed
    main(brdf_dir=brdf_dir, band=band, apply_scale=apply_scale, doy=doy, year_from=year_from, tile=tile)


