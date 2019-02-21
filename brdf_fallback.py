#!/usr/bin/env python

"""
BRDF data extraction utilities from hdf5 files
"""

import os
from os.path import join as pjoin
from datetime import datetime

import fnmatch
import h5py
import numpy as np

import click

from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import attach_image_attributes
from wagl.tiling import generate_tiles
from wagl.constants import BrdfParameters

import brdf_shape


BAND_LIST = ['Band{}'.format(band) for band in range(1, 8)]

TILES = ['h29v10', 'h30v10', 'h31v10', 'h32v10', 'h27v11', 'h28v11', 'h29v11', 'h30v11',
         'h31v11', 'h27v12', 'h28v12', 'h29v12', 'h30v12', 'h31v12', 'h28v13', 'h29v13']


def albedo_band_name(band):
    return 'BRDF_Albedo_Parameters_{}'.format(band)


def quality_band_name(band):
    return 'BRDF_Albedo_Band_Mandatory_Quality_{}'.format(band)


def folder_datetime(folder):
    return datetime.strptime(folder, '%Y.%m.%d')


def folder_doy(folder):
    return folder_datetime(folder).timetuple().tm_yday


def folder_year(folder):
    return folder_datetime(folder).timetuple().tm_year


def hdf5_files(brdf_dir, tile, year_from=None):
    """
    A function to extract relevant MODIS BRDF acquisition details
    from the root folder where BRDF data are stored.

    :param brdf_dir:
        A path name to where hdf5 formatted BRDF data are stored
        The BRDF directories are assumed to be yyyy.mm.dd naming convention.

    :return:
        a nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name

    """
    h5_info = {}

    for item in os.listdir(brdf_dir):
        if year_from is not None and folder_year(item) < year_from:
            continue

        files = os.listdir(pjoin(brdf_dir, item))

        try:
            filename = fnmatch.filter(files, '*.{}.*.h5'.format(tile))[0]
            h5_info[item] = pjoin(brdf_dir, item, filename)
        except IndexError:
            pass

    return h5_info


def read_brdf_dataset(ds, window=None):
    if window is None:
        window = slice(None)

    data = ds[window]

    nodata = ds.attrs['_FillValue']
    nodata_mask = (data == nodata)
    data = data.astype('float32')
    data[nodata_mask] = np.nan

    # quality data
    if len(ds.shape) != 3:
        return data

    # BRDF data
    scale_factor = ds.attrs['scale_factor']
    add_offset = ds.attrs['add_offset']
    return data * scale_factor + add_offset


def get_qualityband_count(h5_info, band_name):
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

    def read_quality_data(filename):
        with h5py.File(filename, 'r') as fid:
            return 1. - read_brdf_dataset(fid[band_name])

    first, *rest = list(h5_info)

    data_sum = read_quality_data(h5_info[first])

    for key in rest:
        new_data = read_quality_data(h5_info[key])

        # https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
        data_sum_nan = np.isnan(data_sum)
        new_data_nan = np.isnan(new_data)
        data_sum = np.where(data_sum_nan & new_data_nan,
                            np.nan,
                            np.where(data_sum_nan, 0., data_sum) + np.where(new_data_nan, 0., new_data))

    return data_sum


def apply_threshold(h5_info, dayofyear, band_name, window, filter_size, thresholds, bad_indices):
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

    all_data_keys = sorted(list(h5_info.keys()))
    doy_data_keys = sorted([key for key in h5_info if folder_doy(key) == dayofyear])

    def get_albedo_data(filename, window):
        with h5py.File(filename, 'r') as fid:
            return read_brdf_dataset(fid[band_name], window)

    data_clean = {}
    for index, key in enumerate(all_data_keys):
        if key not in doy_data_keys:
            continue

        # set the index's for the median filters
        start_idx = index - filter_size
        end_idx = index + filter_size + 1
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(all_data_keys) - 1:
            end_idx = len(all_data_keys)

        temp = {}
        for param_index, param in enumerate(BrdfParameters):
            # get the data iso, vol or geo from data which is a dict for all the keys and convert to numpy array
            data_param = np.ma.array([np.ma.masked_invalid(get_albedo_data(h5_info[date], (param_index,) + window))
                                      for date in all_data_keys[start_idx:end_idx]])

            # extract the value for a key and mask invalid data
            clean_data = np.ma.masked_invalid(data_param[index - start_idx])

            # get temporal local median value as set by filter size
            local_median = np.ma.median(data_param, axis=0)

            # apply threshold test to clean the data set
            threshold_idx = np.ma.where(np.abs(local_median - clean_data) > thresholds[param])

            # replace the data which did not pass threshold test with temporal local median value
            clean_data[threshold_idx] = local_median[threshold_idx]

            # replace bad index data with local median
            clean_data[bad_indices] = local_median[bad_indices]
            temp[param] = clean_data

        data_clean[key] = temp

    return data_clean


def temporal_average(data):
    """
    This function computes temporal average.

    returns the stats on the average using the mean, standard deviation and the number
    of good quality data used in deriving the stats

    In David document, Mean, Stdv, Num and Masks are returned for each temporal average,
    here, we did not output mask because it can be inferred from the number of good
    quality data.

    """
    keys = np.array([k for k in data.keys()])
    set_doy = {folder_doy(item) for item in data}

    def get_temporal_stats(idxs):
        tmp = {}
        for param in BrdfParameters:
            data_param = np.ma.array([data[keys[idx][0]][param] for idx in idxs])
            tmp[param] = dict(mean=np.ma.mean(data_param, axis=0),
                              std=np.ma.std(data_param, axis=0),
                              num=data_param.count(axis=0))
        return tmp

    daily_mean = {}
    for d in set_doy:
        idx_doy = np.argwhere(np.array([folder_doy(item) for item in data]) == d)
        tmp = get_temporal_stats(idx_doy)

        daily_mean[d] = tmp
    return daily_mean


def calculate_combined_mask(afx, rms):
    # generate unfeasible afx and rms values masks from respective min and max values
    # max and min for rms and afx is generated sourced from David's brdf document
    rms_min_mask = np.ma.masked_where(rms < brdf_shape.CONSTANTS['rmsmin'], rms).mask
    rms_max_mask = np.ma.masked_where(rms > brdf_shape.CONSTANTS['rmsmax'], rms).mask
    afx_min_mask = np.ma.masked_where(afx < brdf_shape.CONSTANTS['afxmin'], afx).mask
    afx_max_mask = np.ma.masked_where(afx > brdf_shape.CONSTANTS['afxmax'], afx).mask
    rms_mask = np.ma.mask_or(rms_min_mask, rms_max_mask, shrink=False)
    afx_mask = np.ma.mask_or(afx_min_mask, afx_max_mask, shrink=False)
    rms_afx_mask = np.ma.mask_or(rms_mask, afx_mask, shrink=False)

    # get brdf infeasible  mask
    unfeasible_mask = brdf_shape.get_unfeasible_mask(rms, afx)

    # final mask composed of all previous masks
    return np.ma.mask_or(rms_afx_mask, unfeasible_mask, shrink=False)


def brdf_indices_quality_check(avg_data=None):
    """
    This function performs the quality check on the temporal averages data.
    The quality check is performed in following steps:
        1. data are masked if any of the iso, vol and geo data are not valid.
        2. data are further masked if data are greater or less than valid range
           of brdf shape indices rms and afx (max and min values are sourced
           from MODIS_BRDF_MCD43A_Processing_v2' by David Jupp 2018).
        3. Additional mask based on the requirement (b2 >= ac) is applied
           (parameters definition are based on BRDF_shape_parameters_and_indices'
           by David Jupp 2018.

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
        iso_mean = avg_data[key][BrdfParameters.ISO]['mean']
        vol_mean = avg_data[key][BrdfParameters.VOL]['mean']
        geo_mean = avg_data[key][BrdfParameters.GEO]['mean']

        # generate new mask where all the iso, vol and geo brdf parameters are valid
        mask_param = np.ma.mask_or(np.ma.mask_or(iso_mean.mask, vol_mean.mask, shrink=False),
                                   geo_mean.mask, shrink=False)

        min_num = np.min(np.array([avg_data[key][BrdfParameters.ISO]['num'],
                                   avg_data[key][BrdfParameters.VOL]['num'],
                                   avg_data[key][BrdfParameters.GEO]['num']]), axis=0)

        # mask the brdf param with new mask that is generated from union of masks from
        # individual brdf parameters (iso, vol, and geo)
        iso_mean = np.ma.masked_array(iso_mean, mask=mask_param)
        vol_mean = np.ma.masked_array(vol_mean, mask=mask_param)
        geo_mean = np.ma.masked_array(geo_mean, mask=mask_param)

        iso_std = np.ma.masked_array(avg_data[key][BrdfParameters.ISO]['std'], mask=mask_param)

        # set coefficients of variation
        cov_iso = iso_std / iso_mean

        # set alpha1 and alpha2 in lognormal space
        alpha1, alpha2 = brdf_shape.get_mean_shape_param(iso_mean, vol_mean, geo_mean, cov_iso)

        # set afx and rms indices
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)

        combined_mask = calculate_combined_mask(afx, rms)

        temp = {}
        temp['iso_mean'] = np.ma.masked_array(iso_mean, mask=combined_mask)
        temp['alpha1'] = np.ma.masked_array(alpha1, mask=combined_mask)
        temp['alpha2'] = np.ma.masked_array(alpha2, mask=combined_mask)
        temp['afx'] = np.ma.masked_array(rms, mask=combined_mask)
        temp['rms'] = np.ma.masked_array(afx, mask=combined_mask)
        temp['mask'] = np.array(combined_mask)
        temp['num'] = np.array(min_num)
        filtered_data[key] = temp

    return filtered_data


def calculate_thresholds(h5_info, band_name):
    # spatial stats required to set the threshold value
    def std(filename, index):
        with h5py.File(filename, 'r') as fid:
            return np.nanstd(read_brdf_dataset(fid[band_name], (index,)))

    # get threshold values
    return {param: np.mean(np.array([std(filename, index) for filename in h5_info.values()]))
            for index, param in enumerate(BrdfParameters)}


def create_dataset(group, band_name, shape, attrs,
                   dtype=np.int16, chunks=(1, 240, 240), filter_opts=None,
                   compression=H5CompressionFilter.LZF):
    if filter_opts is None:
        filter_opts = {}
    else:
        filter_opts = filter_opts.copy()

    if 'chunks' not in filter_opts:
        filter_opts['chunks'] = chunks

    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    ds = group.create_dataset(band_name, shape=shape, dtype=dtype, **kwargs)

    attach_image_attributes(ds, attrs)

    return ds


def create_brdf_datasets(group, band_name, shape, common_attrs,
                         chunks=(1, 240, 240), filter_opts=None,
                         compression=H5CompressionFilter.LZF):

    attrs = dict(scale_factor=0.0001, add_offset=0,
                 _FillValue=32767, bands="alpha1: 1, alpha2: 2",
                 description=('BRDF albedo shape parameters (alpha1 and alpha2)'
                              'derived from {}'
                              'in lognormal space'.format(albedo_band_name(band_name))),
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Shape_Parameters_{}'.format(band_name),
                   (2,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)

    attrs = dict(scale_factor=0.0001, add_offset=0,
                 _FillValue=32767, bands="iso_mean: 1, afx: 2, rms: 3",
                 description=('BRDF Albedo ISO parameter and statistics (rms and afx)'
                              'generated to support future validation work'),
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Shape_Indices_{}'.format(band_name),
                   (3,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)

    attrs = dict(description=('Mask and number of valid data used'
                              'in generating BRDF Albedo shape parameters'),
                 bands="mask: 1, num: 2",
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Shape_Parameters_Quality_{}'.format(band_name),
                   (2,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)


def write_chunk(data_dict, fid, band_name, window):
    """
    write numpy array to to h5 files with user supplied attributes
    and compression
    """
    # refactor?
    assert len(data_dict) == 1
    key = list(data_dict.keys())[0]

    data_main = np.ma.array([data_dict[key]['alpha1'], data_dict[key]['alpha2']])
    data_support = np.ma.array([data_dict[key]['iso_mean'], data_dict[key]['afx'], data_dict[key]['rms']])
    data_quality = np.array([data_dict[key]['mask'], data_dict[key]['num']])

    data_main = data_main * 10000
    data_main = data_main.filled(fill_value=32767).astype(np.int16)
    fid['BRDF_Albedo_Shape_Parameters_{}'.format(band_name)][window] = data_main

    data_support = data_support * 10000
    data_support = data_support.filled(fill_value=32767).astype(np.int16)
    fid['BRDF_Albedo_Shape_Indices_{}'.format(band_name)][window] = data_support

    data_quality = data_quality.astype(np.int16)
    fid['BRDF_Albedo_Shape_Parameters_Quality_{}'.format(band_name)][window] = data_quality


def get_band_info(h5_info, band_name):
    for date in h5_info:
        with h5py.File(h5_info[date], 'r') as fid:
            ds = fid[band_name]
            return ds.shape, {key: ds.attrs[key] for key in ['crs_wkt', 'geotransform']}


def write_brdf_fallback_band(fid, band, h5_info, dayofyear,
                             min_numpix_required, filter_size, compute_chunks, data_chunks):
    # get counts of good pixel quality
    quality_count = get_qualityband_count(h5_info=h5_info, band_name=quality_band_name(band))
    quality_count = np.ma.masked_invalid(quality_count)

    # get the index where band_quality number is less the minimum number of valid pixels required
    bad_indices = (quality_count < min_numpix_required).filled(False)
    print('calculated bad pixels', np.where(bad_indices))

    thresholds = calculate_thresholds(h5_info, albedo_band_name(band))
    print('spatial stats', thresholds)

    shape, attrs = get_band_info(h5_info, albedo_band_name(band))
    shape = shape[-2:]
    create_brdf_datasets(fid, band, shape, attrs, chunks=data_chunks)

    for x, y in generate_tiles(shape[0], shape[1], compute_chunks[0], compute_chunks[1]):
        window = (slice(*y), slice(*x))
        print('processing [{}:{}] [{}:{}]'.format(y[0], y[1], x[0], x[1]))

        data_clean = apply_threshold(h5_info, dayofyear,
                                     albedo_band_name(band), window, filter_size, thresholds,
                                     bad_indices[window])

        # compute daily, monthly and yearly mean from clean data sets
        avg_data = temporal_average(data_clean)
        filtered_data = brdf_indices_quality_check(avg_data=avg_data)
        write_chunk(filtered_data, fid, band, window=(slice(None),) + window)


def write_brdf_fallback(brdf_dir, tile, dayofyear, outdir, filter_size,
                        pthresh=10.0, year_from=None, data_chunks=(1, 240, 240),
                        compute_chunks=(240, 240)):

    h5_info = hdf5_files(brdf_dir, tile=tile, year_from=year_from)
    print('got info for', len(h5_info), 'files')

    # generate number of valid pixel count required for analysis in a time series stack
    # as defined in David Jupp's BRDF document
    min_numpix_required = int((pthresh / 100.0) * len(h5_info))
    print('min pix required', min_numpix_required)

    outfile = pjoin(outdir, 'MCD43A1.JLAV.006.{}.DOY.{:03}.h5'.format(tile, dayofyear))
    with h5py.File(outfile, 'w') as fid:
        for band in BAND_LIST:
            write_brdf_fallback_band(fid, band, h5_info, dayofyear,
                                     min_numpix_required, filter_size, compute_chunks, data_chunks)


@click.command()
@click.option('--brdf-dir', default='/g/data/u46/users/ia1511/Work/data/brdf-collection-6/reprocessed/')
@click.option('--outdir', default='/g/data/u46/users/ia1511/Work/data/brdf-collection-6/optimize/fallback_results/')
@click.option('--tile', default='h29v10')
@click.option('--dayofyear', default=9)
@click.option('--year-from', default=2015)
@click.option('--filter-size', default=4)
def main(brdf_dir, outdir, tile, dayofyear, year_from, filter_size):
    write_brdf_fallback(brdf_dir, tile, dayofyear, outdir, filter_size, year_from=year_from)


if __name__ == "__main__":
    main()
