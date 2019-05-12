#!/usr/bin/env python

"""
BRDF data extraction utilities from hdf5 files
"""

import sys
import os
from pathlib import Path
from os.path import join as pjoin, basename
import h5py
import fnmatch
import datetime
import numpy as np
import json
from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import write_h5_image, attach_attributes, attach_image_attributes
import brdf_shape

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

BRDF_PARAM_INDEX = {'iso': 0, 'vol': 1, 'geo': 2}
BRDF_PARAM_CLEAN_NAME = {'iso': 'iso_clean', 'vol': 'vol_clean', 'geo': 'geo_clean'}
BRDF_PARAM_MEAN = {'iso': 'iso_clean_mean', 'vol': 'vol_clean_mean', 'geo': 'geo_clean_mean'}
BRDF_PARAM_STDV = {'iso': 'iso_clean_std', 'vol': 'vol_clean_std', 'geo': 'geo_clean_std'}
BRDF_PARAM_NUM = {'iso': 'iso_clean_num', 'vol': 'vol_clean_num', 'geo': 'geo_clean_num'}


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
    less than particular day of year( for all years > given year)
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

        if yr >= year:
            h5_file = (h5_info[k][tile])
            print('loading', sds_name, 'from', h5_file)
            with h5py.File(h5_file, 'r') as fid:

                ds = fid[sds_name]
                # print([item for item in ds.attrs])
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
    # print(sys.getsizeof(data_dict))
    return data_dict, h5_info


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

    # get threshold values 
    iso_threshold, vol_threshold, geo_threshold = get_threshold(spatial_stats)

    # get the index where band_quality number is less the minimum number of valid pixels required
    bad_idx = np.where(quality_count < min_numpix_required)

    data_clean = {}
    for i in range(len(keys)):
        temp = {}
        # set the index's for the median filters
        start_idx = i - filter_size
        end_idx = i + filter_size + 1
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(keys)-1:
            end_idx = len(keys)

        for param in BRDF_PARAM_INDEX:
            # get the data iso, vol or geo from data which is a dict for all the keys and convert to numpy array
            data_param = np.ma.array([np.ma.masked_invalid(data[key][BRDF_PARAM_INDEX[param]]) for key in keys])

            # extract the value for a key and mask invalid data
            clean_data = np.ma.masked_invalid(data_param[i])

            # get temporal local median value as set by filter size
            local_median = np.ma.median(data_param[start_idx:end_idx], axis=0)

            # set parameter index and names and get threshold value
            if param is 'iso':
                threshold = iso_threshold
                param_name = BRDF_PARAM_CLEAN_NAME['iso']
            elif param is 'vol':
                threshold = vol_threshold
                param_name = BRDF_PARAM_CLEAN_NAME['vol']
            elif param is 'geo':
                threshold = geo_threshold
                param_name = BRDF_PARAM_CLEAN_NAME['geo']
            else:
                raise ValueError

            # apply threshold test to clean the data set
            threshold_idx = np.ma.where(abs(local_median - clean_data) > threshold)

            # replace the data which did not pass threshold test with temporal local median value
            clean_data[threshold_idx] = local_median[threshold_idx]

            # replace bad index data with local median
            clean_data[bad_idx] = local_median[bad_idx]
            temp[param_name] = clean_data

        data_clean[keys[i]] = temp

    return data_clean


def temporal_average(data, h5_info=None, tile=None, tag=None):
    """
    This function computes temporal average of data sets for same day of year (doy),
    same month (monthly), and same year(yearly) as determined from tags

    returns the stats on the average using the mean, standard deviation and the number
    of good quality data used in deriving the stats

    In David document, Mean, Stdv, Num and Masks are returned for each temporal average,
    here, we did not output mask because it can be inferred from the number of good
    quality data.

    """
    keys = np.array([k for k in data.keys()])
    key_fmt = '%Y.%m.%d'
    dt_list = [datetime.datetime.strptime(item, key_fmt) for item in data.keys()]

    set_doy = set([dt.timetuple().tm_yday for dt in dt_list])
    set_mnt = set([dt.timetuple().tm_mon for dt in dt_list])
    set_yr = set([dt.timetuple().tm_year for dt in dt_list])

    def get_temporal_stats(idxs):
        tmp = {}
        for param in BRDF_PARAM_CLEAN_NAME:
            data_param = np.ma.array([data[keys[idx][0]][BRDF_PARAM_CLEAN_NAME[param]] for idx in idxs])
            tmp['{par}_mean'.format(par=BRDF_PARAM_CLEAN_NAME[param])] = np.ma.mean(data_param, axis=0)
            tmp['{par}_std'.format(par=BRDF_PARAM_CLEAN_NAME[param])] = np.ma.std(data_param, axis=0)
            tmp['{par}_num'.format(par=BRDF_PARAM_CLEAN_NAME[param])] = data_param.count(axis=0)
        return tmp

    if tag is "Daily":
        daily_mean = {}
        for d in set_doy:
            idx_doy = np.argwhere(np.array([dt.timetuple().tm_yday for dt in dt_list]) == d)
            tmp = get_temporal_stats(idx_doy)
            if h5_info:
                tmp['data_id'] = {keys[idx][0]: h5_info[keys[idx][0]][tile] for idx in idx_doy}

            daily_mean[d] = tmp
        return daily_mean

    if tag is 'Monthly':
        monthly_mean = {}
        for m in set_mnt:
            idx_mnt = np.argwhere(np.array([dt.timetuple().tm_mon for dt in dt_list]) == m)
            tmp = get_temporal_stats(idx_mnt)
            if h5_info:
                tmp['data_id'] = {keys[idx][0]: h5_info[keys[idx][0]][tile] for idx in idx_mnt}
            monthly_mean[m] = tmp
        return monthly_mean

    if tag is 'Yearly':
        yearly_mean = {}
        for y in set_yr:
            idx_yr = np.argwhere(np.array([dt.timetuple().tm_year for dt in dt_list]) == y)
            tmp = get_temporal_stats(idx_yr)
            if h5_info:
                tmp['data_id'] = {keys[idx][0]: h5_info[keys[idx][0]][tile] for idx in idx_yr}
            yearly_mean[y] = tmp
        return yearly_mean

def brdf_indices_quality_check(avg_data=None):
    """
    This function performs the quality check on the temporal averages data.
    The quality check is performed in following steps:
        1. data are masked if any of the iso, vol and geo data are not valid.
        2. data are further masked if data are greater or less than valid range
           of brdf shape indices rms and afx (max and min values are sourced
           from MODIS_BRDF_MCD43A_Processing_v2' by David's Jubb 2018).
        3. Additional mask based on the requirement (b2 >= ac) is applied
           (parameters definition are based on BRDF_shape_parameters_and_indices'
           by David Jubb 2018.

    :param avg_data:
         A 'dict' type data set that contains the numpy array type dataset with
         temporal average of clean brdf parameters (iso, vol, and geo) and
         its associated standard deviation and number of observations used
         to generate the temporal average.

    :return filtered_data:
         A 'dict' type data that contains the filtered data of brdf
         shape function (alpha1 and alpha2) in lognormal space. Additional
         data, mean brdf iso parameter, shape indices (rms and afx), mask
         and number of observations used in generating shape function are
         also included.

    """
    filtered_data = {}
    for key in avg_data.keys():
        # set the mean brdf data from the avg_data dict for each keys
        iso_mean = avg_data[key][BRDF_PARAM_MEAN['iso']]
        vol_mean = avg_data[key][BRDF_PARAM_MEAN['vol']]
        geo_mean = avg_data[key][BRDF_PARAM_MEAN['geo']]

        # generate new mask where all the iso, vol and geo brdf parameters are valid
        mask_param = np.ma.mask_or(np.ma.mask_or(iso_mean.mask, vol_mean.mask), geo_mean.mask)

        min_num = np.min(np.array([avg_data[key][BRDF_PARAM_NUM['iso']],
                                   avg_data[key][BRDF_PARAM_NUM['vol']],
                                   avg_data[key][BRDF_PARAM_NUM['geo']]]), axis=0)

        # mask the brdf param with new mask that is generated from union of masks from
        # individual brdf parameters (iso, vol, and geo)
        iso_mean = np.ma.masked_array(iso_mean, mask=mask_param)
        vol_mean = np.ma.masked_array(vol_mean, mask=mask_param)
        geo_mean = np.ma.masked_array(geo_mean, mask=mask_param)

        iso_std = np.ma.masked_array(avg_data[key][BRDF_PARAM_STDV['iso']], mask=mask_param)
        # vol_std = np.ma.masked_array(avg_data[key][BRDF_PARAM_STDV['vol']], mask=mask_param)
        # geo_std = np.ma.masked_array(avg_data[key][BRDF_PARAM_STDV['geo']], mask=mask_param)

        # set coefficients of variation
        cov_iso = iso_std / iso_mean
        # cov_vol = vol_std / vol_mean
        # cov_geo = geo_std / geo_mean

        # set alpha1 and alpha2 in lognormal space
        alpha1, alpha2 = brdf_shape.get_mean_shape_param(iso_mean, vol_mean, geo_mean, cov_iso)

        # set afx and rms indices
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)

        # generate unfeasible afx and rms values masks from respective min and max values
        # max and min for rms and afx is generated sourced from David's brdf document
        rms_min_mask = np.ma.masked_where(rms < brdf_shape.CONSTANTS['rmsmin'], rms).mask
        rms_max_mask = np.ma.masked_where(rms > brdf_shape.CONSTANTS['rmsmax'], rms).mask 
        afx_min_mask = np.ma.masked_where(afx < brdf_shape.CONSTANTS['afxmin'], afx).mask 
        afx_max_mask = np.ma.masked_where(afx > brdf_shape.CONSTANTS['afxmax'], afx).mask
        rms_mask = np.ma.mask_or(rms_min_mask, rms_max_mask)
        afx_mask = np.ma.mask_or(afx_min_mask, afx_max_mask)
        rms_afx_mask = np.ma.mask_or(rms_mask, afx_mask)

        # get brdf infeasible  mask
        unfeasible_mask = brdf_shape.get_unfeasible_mask(rms, afx)

        # final mask composed of all previous masks
        combined_mask = np.ma.mask_or(rms_afx_mask, unfeasible_mask)

        temp = {}
        temp['data_id'] = avg_data[key]['data_id']
        temp['iso_mean'] = np.ma.masked_array(iso_mean, mask=combined_mask)
        temp['alpha1'] = np.ma.masked_array(alpha1, mask=combined_mask)
        temp['alpha2'] = np.ma.masked_array(alpha2, mask=combined_mask)
        temp['afx'] = np.ma.masked_array(rms, mask=combined_mask)
        temp['rms'] = np.ma.masked_array(afx, mask=combined_mask)
        temp['mask'] = np.array(combined_mask)
        temp['num'] = np.array(min_num)
        filtered_data[key] = temp

    return filtered_data


def write_h5(data_dict, band_name, tile, outdir, tag=None, filter_opts=None,
             compression=H5CompressionFilter.LZF):
    """
    write numpy array to to h5 files with user supplied attributes
    and compression
    """
    for key in data_dict.keys():

        if tag is 'Daily':
            tag1 = 'DOY'
            tag2 = '{:03}'.format(key)

        elif tag is 'Monthly':
            tag1 = 'MONTH'
            tag2 = '{:02}'.format(key)

        elif tag is 'Yearly':
            tag1 = 'YEAR'
            tag2 = key

        else:
            raise ValueError('tag is not defined: options are {Daily, Monthly, Yearly}')

        outfile = pjoin(outdir, 'MCD43A1.JLAV.{t1}.{t2}.{t3}.006.{b}.h5'.format(t1=tag1, t2=tag2, t3=tile,
                                                                                b=band_name))

        with h5py.File(outfile, 'w') as fid:
            h5_files = [data_dict[key]['data_id'][k] for k in data_dict[key]['data_id'].keys()]
            attrs, attrs_main, attrs_support, attrs_quality = {}, {}, {}, {}
            with h5py.File(h5_files[0], 'r') as f:
                ds = f[band_name]
                attrs['crs_wkt'] = ds.attrs['crs_wkt']
                attrs['geotransform'] = ds.attrs['geotransform']

            attrs_main['scale_factor'] = 0.0001
            attrs_main['add_offset'] = 0
            attrs_main['_FillValue'] = 32767
            attrs_main['description'] = ('BRDF albedo shape parameters (alpha1 and alpha2) derived from {b} '
                                         'in lognormal space'.format(b=band_name))
            attrs_main['bands'] = "alpha1: 1, alpha2: 2"
            attrs_main.update(**attrs)

            attrs_support['scale_factor'] = 0.0001
            attrs_support['add_offset'] = 0
            attrs_support['_FillValue'] = 32767
            attrs_support['description'] = ('BRDF Albedo ISO parameter and statistics (rms and afx) '
                                            'generated to support future validation work')
            attrs_support['bands'] = "iso_mean: 1, afx: 2, rms: 3"
            attrs_support.update(**attrs)

            attrs_quality['description'] = ('Mask and number of valid data used in generating BRDF Albedo '
                                            'shape parameters')
            attrs_quality['bands'] = "mask: 1, num: 2"
            attrs_quality.update(**attrs)

            chunks = (1, 240, 240)

            if not filter_opts:
                filter_opts = dict()
                filter_opts['chunks'] = chunks
            else:
                filter_opts = filter_opts.copy()

            if 'chunks' not in filter_opts:
                filter_opts['chunks'] = chunks

            data_main = np.ma.array([data_dict[key]['alpha1'], data_dict[key]['alpha2']])
            data_support = np.ma.array([data_dict[key]['iso_mean'], data_dict[key]['afx'], data_dict[key]['rms']])
            data_quality = np.array([data_dict[key]['mask'], data_dict[key]['num']])

            data_main = data_main * 10000
            data_main = data_main.filled(fill_value=32767).astype(np.int16)
            write_h5_image(data_main, 'BRDF_Albedo_Shape_Parameters', fid, attrs=attrs_main, compression=compression,
                           filter_opts=filter_opts)

            data_support = data_support * 10000
            data_support = data_support.filled(fill_value=32767).astype(np.int16)
            write_h5_image(data_support, 'BRDF_Albedo_Shape_Indices', fid, attrs=attrs_support, compression=compression,
                           filter_opts=filter_opts)

            data_quality = data_quality.astype(np.int16)
            write_h5_image(data_quality, 'BRDF_Albedo_Shape_Parameters_Quality', fid, attrs=attrs_quality,
                           compression=compression, filter_opts=filter_opts)


def main(brdf_dir=None, band=None, apply_scale=True, doy=None, year_from=None,
         tile=None, outdir=None, filter_size=None, pthresh=10.0):

    sds_databand_name = SDS_BAND_NAME['{b}'.format(b=band)]
    sds_qualityband_name = SDS_BAND_NAME['QUAL_{b}'.format(b=band)]

    # get a list data sets for given tile for particular date and tile
    sds_data, h5_info = get_sds_doy_dataset(dirs=brdf_dir, dayofyear=doy, sds_name=sds_databand_name,
                                            year=year_from, tile=tile, apply_scale=apply_scale)
    qual_data, _ = get_sds_doy_dataset(dirs=brdf_dir, dayofyear=doy, sds_name=sds_qualityband_name,
                                       year=year_from, tile=tile, apply_scale=apply_scale)

    # spatial stats required to set the threshold value
    tile_spatial_stats = generate_tile_spatial_stats(sds_data)

    # generate number of valid pixel count required for analysis in a time series stack
    # as defined in David Jubb's BRDF document
    min_numpix_required = float(int((pthresh / 100.0) * len(sds_data)))

    # get counts of good pixel quality
    quality_count = get_qualityband_count(quality_data=qual_data)

    quality_count = np.ma.masked_invalid(quality_count)

    data_clean = apply_threshold(sds_data, filter_size, tile_spatial_stats, quality_count, min_numpix_required)

    # free some memory
    sds_data, qual_data = None, None

    # compute daily, monthly and yearly mean from clean data sets
    tags = ["Daily"]
    for tag in tags:
        avg_data = temporal_average(data_clean, h5_info=h5_info, tile=tile, tag=tag)
        filtered_data = brdf_indices_quality_check(avg_data=avg_data)
        write_h5(filtered_data, sds_databand_name, tile, outdir, tag=tag)


if __name__ == "__main__":
    # brdf_dir = '/g/data/u46/users/pd1813/BRDF_PARAM/MCD43A1_C6_HDF5_TILE_DATASET/'
    brdf_dir = '/g/data/u46/users/ia1511/Work/data/brdf-collection-6/reprocessed/'
    outdir = '/g/data/u46/users/ia1511/Work/data/brdf-collection-6/optimize/old_results/'
    band = "BAND1"
    tile = 'h29v10'
    pthresh = 10.0
    apply_scale = True
    doy = 9  # subset for which doy to be processed
    year_from = 2015  # subset from which year to be processed
    main(brdf_dir=brdf_dir, band=band, apply_scale=apply_scale, doy=doy, year_from=year_from, tile=tile,
         outdir=outdir, filter_size=4, pthresh=pthresh)
