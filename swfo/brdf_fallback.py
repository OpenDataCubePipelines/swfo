#!/usr/bin/env python

"""
Pre-MODIS BRDF average implementation as described in
MODIS_BRDF_MCD43A1_Processing_v2.docx and
BRDF_shape_parameters_and_indices.docx
"""

import tempfile
import os
from os.path import join as pjoin
import datetime
import time
from sys import stdout
from contextlib import contextmanager

from multiprocessing import Pool as ProcessPool, Lock
import fnmatch
import h5py
import numpy as np
import click
from memory_profiler import profile

from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import attach_image_attributes
from wagl.tiling import generate_tiles
from wagl.constants import BrdfModelParameters

import brdf_shape

BAND_LIST = ['Band{}'.format(band) for band in range(1, 8)]

BRDF_AVG_FILE_FMT = 'MCD43A1.JLAV.006.{}.DOY.{:03}.{}.h5'

TILES = ['h29v10', 'h30v10', 'h31v10', 'h32v10', 'h27v11', 'h28v11', 'h29v11', 'h30v11',
         'h31v11', 'h27v12', 'h28v12', 'h29v12', 'h30v12', 'h31v12', 'h28v13', 'h29v13']

LOCKS = {}

@contextmanager
def timing(task_name):
    start_time = time.time()
    print('starting', task_name, 'at', time.ctime(int(start_time)), file=stdout)
    stdout.flush()
    yield
    print('finished', task_name, 'in', time.time() - start_time, file=stdout)
    stdout.flush()


def albedo_band_name(band):
    """
    :param band:
        A 'str' type: A MODIS band name.
    :return:
        A 'str' type: A subdataset (brdf parameter band name) for the specified band.
    """
    return 'BRDF_Albedo_Parameters_{}'.format(band)


def quality_band_name(band):
    """
    :param band:
        A 'str' type: A MODIS band name.
    :return:
        A 'str' type: A subdataset (brdf quality band) name for the specified band.
    """
    return 'BRDF_Albedo_Band_Mandatory_Quality_{}'.format(band)


def folder_datetime(folder):
    """
    :param folder:
        A 'str' type: A Folder name in format ('%Y'.%m.%d').
    :return:
        A 'date' object parsed from folder format.
    """
    return datetime.datetime.strptime(folder, '%Y.%m.%d')


def folder_doy(folder):
    """
    :param folder:
        A 'str' type: A Folder name in format ('%Y'.%m.%d').
    :return:
        A 'int' type: Day of the year that folder corresponds to.
    """
    return folder_datetime(folder).timetuple().tm_yday


def folder_year(folder):
    """
    :param folder:
        A 'str' type: A Folder name in format ('%Y'.%m.%d').
    :return:
        A 'int' type: A year that folder corresponds to.
    """
    return folder_datetime(folder).timetuple().tm_year


def gauss_filter(filter_size):
    """
    A Guassian filter where the weights are a normal exponential but value 1 at centre.
    Filter_rad is half the total filter length. The value of the normal distribution at
    a distance filter_size defines the filter. If the value of the exponential at filter_rad
    is 0.01 then sig can be defined as in the function.
    It is a backwards way to get a filter that is long enough to do the job.
    In this case you define the filter_size and sig gets worked out. It is roughly where the
    area between -filter_size and +filter_size is 99.9% of the total area.
    It just makes sure you have a normal distribution that is basically zero outside
    the length filter_rad to filter_rad.
    :param filter_size:
        A 'int' type: A length of a filter
    :return:
        A numpy array: A filter kernels
    """
    sig = 0.329505 * filter_size

    return np.array([np.exp(-0.5*((j-filter_size)/sig)**2) for j in range(2*filter_size+1)])


def hdf5_files(brdf_dir, tile, year_from=None):
    """
    A function to extract relevant MODIS BRDF acquisition details
    from the root folder where BRDF data are stored.
    :param brdf_dir:
        A 'str' type: path name to where hdf5 formatted BRDF data are stored
        The BRDF directories are assumed to be yyyy.mm.dd naming convention.
    :param tile:
        A 'str' type: The name of a MODIS tile.
    :param year_from:
        A 'int' type: The year from where processing should begin from.
        Default = None, will include all datasets in directory 'brdf_dir'.
    :return:
        A 'dict' type: a nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name.
    """
    h5_info = {}
    for item in os.listdir(brdf_dir):
        if year_from is not None and folder_year(item) < year_from:
            continue
        files = os.listdir(pjoin(brdf_dir, item))
        try:
            filename = fnmatch.filter(files, '*.{}*.h5'.format(tile))[0]
            h5_info[item] = pjoin(brdf_dir, item, filename)
        except IndexError:
            pass

    return h5_info


def read_brdf_dataset(ds, window=None):
    """
    :param ds:
        A 'file object' type: hdf5 file object containing the BRDF data set
    :param window:
        A 'slice object' type: contain the set of indices specified by
        range(start, stop, step). Default=None, results in reading whole
        data set
    :return:
        A 'array' type: slice of data set specified by window if set or all
        the data set. The BRDF parameter are scale and offset factor
        corrected and filled with numpy 'nan' for no data values.
    """
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
    return scale_factor * (data - add_offset)


def get_qualityband_count(h5_info, band_name):
    """
    This function computes and returns the number of valid pixels
    in a time series stack.
    :param h5_info:
        A 'dict' type: A nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name.
    :param band_name:
        A 'str' type: A name of quality band.
    :return:
        A numpy array with total quality band counts across all the datasets
        in h5_info.
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


def calculate_combined_mask(afx, rms):
    """
    A function to generate mask based on BRDF shape indicies using unfeasible afx
    and rms values derived from respective min and max values
    The max and min for rms and afx are sourced from David's brdf document.
    :param afx:
        A 'numpy array' type: The Root Mean Square (RMS) statistics.
    :param afx:
        A 'numpy array' type: The Anisotropic Flat Index (AFX) statistics.
    :return:
        A 'numpy array' type: A mask on the array which are considered unfeasible
        based on conditions defined.
    """

    rms_min_mask = np.ma.masked_where(rms < brdf_shape.CONSTANTS['rmsmin'], rms).mask
    rms_max_mask = np.ma.masked_where(rms > brdf_shape.CONSTANTS['rmsmax'], rms).mask
    afx_min_mask = np.ma.masked_where(afx < brdf_shape.CONSTANTS['afxmin'], afx).mask
    afx_max_mask = np.ma.masked_where(afx > brdf_shape.CONSTANTS['afxmax'], afx).mask
    rms_mask = np.ma.mask_or(rms_min_mask, rms_max_mask, shrink=False)
    afx_mask = np.ma.mask_or(afx_min_mask, afx_max_mask, shrink=False)
    rms_afx_mask = np.ma.mask_or(rms_mask, afx_mask, shrink=False)

    return rms_afx_mask


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
    for key in avg_data:
        # set the mean brdf data from the avg_data dict for each keys
        iso_mean = avg_data[key][BrdfModelParameters.ISO]['mean']
        vol_mean = avg_data[key][BrdfModelParameters.VOL]['mean']
        geo_mean = avg_data[key][BrdfModelParameters.GEO]['mean']

        # generate new mask where all the iso, vol and geo brdf parameters are valid
        mask_param = np.ma.mask_or(np.ma.mask_or(iso_mean.mask, vol_mean.mask, shrink=False),
                                   geo_mean.mask, shrink=False)

        min_num = np.min(np.array([avg_data[key][BrdfModelParameters.ISO]['num'],
                                   avg_data[key][BrdfModelParameters.VOL]['num'],
                                   avg_data[key][BrdfModelParameters.GEO]['num']]), axis=0)

        # mask the brdf param with new mask that is generated from union of masks from
        # individual brdf parameters (iso, vol, and geo)
        iso_mean = np.ma.masked_array(iso_mean, mask=mask_param)
        vol_mean = np.ma.masked_array(vol_mean, mask=mask_param)
        geo_mean = np.ma.masked_array(geo_mean, mask=mask_param)

        iso_std = np.ma.masked_array(avg_data[key][BrdfModelParameters.ISO]['std'], mask=mask_param)

        # set coefficients of variation
        cov_iso = iso_std / iso_mean

        # set alpha1 and alpha2 in lognormal space
        alpha1, alpha2 = brdf_shape.get_mean_shape_param(iso_mean, vol_mean, geo_mean, cov_iso)

        # set afx and rms indices
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)

        combined_mask = calculate_combined_mask(afx, rms)

        temp = {}
        temp[BrdfModelParameters.ISO.value] = np.ma.masked_array(iso_mean, mask=combined_mask)
        temp[BrdfModelParameters.VOL.value] = np.ma.masked_array(alpha1 * iso_mean, mask=combined_mask)
        temp[BrdfModelParameters.GEO.value] = np.ma.masked_array(alpha2 * iso_mean, mask=combined_mask)
        temp['afx'] = np.ma.masked_array(afx, mask=combined_mask)
        temp['rms'] = np.ma.masked_array(rms, mask=combined_mask)
        temp['mask'] = np.array(combined_mask)
        temp['num'] = np.array(min_num)
        filtered_data[key] = temp

    return filtered_data


def get_std_block(h5_info, band_name, index, window):
    """
    A function to compute a standard deviation for across a temporal axis.
    This function was written to facilitate parallel processing.
    """

    def __get_data(dat_filename, window):
        with h5py.File(dat_filename, 'r') as fid:
            dat = read_brdf_dataset(fid[band_name], window)
            return dat

    data = np.array([__get_data(filename, (index,) + window) for filename in h5_info.values()])
    run_median = np.nanmedian(data, axis=0, keepdims=False)

    median_filled_data = []
    for item in data:
        idx = np.where(np.isnan(item))
        item[idx] = run_median[idx]
        median_filled_data.append(item)

    return np.nanstd(np.array(median_filled_data), axis=0, ddof=1, keepdims=False)


def calculate_thresholds(h5_info, band_name, shape, compute_chunks, nprocs=None):
    """
    Computes threshold needed needed to clean temporal series
    """
    thresh_dict = {}
    for index, param in enumerate(BrdfModelParameters):
        args = []
        for x, y in generate_tiles(shape[0], shape[1], compute_chunks[0], compute_chunks[1]):
            window = (slice(*y), slice(*x))
            args.append([h5_info, band_name, index, window])

        if nprocs:
            with ProcessPool(processes=nprocs) as pool:
                results = pool.starmap(get_std_block, args)
        else:
            results = [get_std_block(*arg) for arg in args]

        thresh_dict[param] = np.nanmean(results)

    return thresh_dict


def create_dataset(group, band_name, shape, attrs,
                   dtype=np.int16, chunks=(1, 240, 240), filter_opts=None,
                   compression=H5CompressionFilter.BLOSC_ZSTANDARD):
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
                         compression=H5CompressionFilter.BLOSC_ZSTANDARD):

    attrs = dict(scale_factor=0.0001, add_offset=0,
                 _FillValue=32767, bands="{}: 1, {}: 2:, {}: 3".format(BrdfModelParameters.ISO.value,
                                                                       BrdfModelParameters.VOL.value,
                                                                       BrdfModelParameters.GEO.value),
                 description=('BRDF albedo parameters (iso, vol and geo)'
                              ' derived from {}'
                              ' in lognormal space'.format(albedo_band_name(band_name))),
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Parameters_{}'.format(band_name),
                   (3,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)

    attrs = dict(scale_factor=0.0001, add_offset=0,
                 _FillValue=32767, bands="afx: 1, rms: 2",
                 description=('BRDF shape indices (afx and rms)'
                              ' generated to support future validation work'),
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Shape_Indices_{}'.format(band_name),
                   (2,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)

    attrs = dict(description=('Mask and number of valid data used'
                              ' in generating BRDF Albedo shape parameters'),
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

    data_main = np.ma.array([data_dict[key][BrdfModelParameters.ISO.value],
                             data_dict[key][BrdfModelParameters.VOL.value],
                             data_dict[key][BrdfModelParameters.GEO.value]])

    data_support = np.ma.array([data_dict[key]['afx'], data_dict[key]['rms']])
    data_quality = np.array([data_dict[key]['mask'], data_dict[key]['num']])

    data_main = data_main * 10000
    data_main = np.rint(data_main).filled(fill_value=32767).astype(np.int16)

    fid['BRDF_Albedo_Parameters_{}'.format(band_name)][window] = data_main

    data_support = data_support * 10000
    data_support = np.rint(data_support).filled(fill_value=32767).astype(np.int16)
    fid['BRDF_Albedo_Shape_Indices_{}'.format(band_name)][window] = data_support

    data_quality = data_quality.astype(np.int16)
    fid['BRDF_Albedo_Shape_Parameters_Quality_{}'.format(band_name)][window] = data_quality


def get_band_info(h5_info, band_name):
    for date in h5_info:
        with h5py.File(h5_info[date], 'r') as fid:
            ds = fid[band_name]
            return ds.shape, {key: ds.attrs[key] for key in ['crs_wkt', 'geotransform']}


def temporal_average(data, doy):
    """
    This function computes temporal average.

    returns the stats on the average using the mean, standard deviation and the number
    of good quality data used in deriving the stats

    In David document, Mean, Stdv, Num and Masks are returned for each temporal average,
    here, we did not output mask because it can be inferred from the number of good
    quality data.

    """
    tmp = {}
    for param in BrdfModelParameters:
        data_param = np.ma.array([data[param][key] for key in data[param].keys() if folder_doy(key) == doy])

        tmp[param] = dict(mean=np.ma.mean(data_param, axis=0),
                          std=np.ma.std(data_param, ddof=0, axis=0),
                          num=data_param.count(axis=0))
    return {doy: tmp}


def apply_threshold(h5_info, band_name, window, filter_size, thresholds, bad_indices):
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

    def get_albedo_data(filename, window):
        with h5py.File(filename, 'r') as fid:
            return read_brdf_dataset(fid[band_name], window)

    result = {}
    for key in all_data_keys:
        y_shape = window[0].stop - window[0].start
        x_shape = window[1].stop - window[1].start
        result[key] = np.full(shape=(3, y_shape, x_shape), fill_value=32767, dtype=np.int16)

    # dictionary mapping date to data
    data_dict = {}

    for index, key in enumerate(all_data_keys):
        # set the index's for the median filters
        start_idx = index - filter_size
        end_idx = index + filter_size + 1

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(all_data_keys) - 1:
            end_idx = len(all_data_keys)

        # read the data relevant to this range
        for date in all_data_keys[start_idx:end_idx]:
            if date not in data_dict:
                data_dict[date] = np.ma.masked_invalid(get_albedo_data(h5_info[date], (slice(None),) + window))

        # clean up the data we don't need anymore
        for date in list(data_dict):
            if date not in all_data_keys[start_idx:end_idx]:
                del data_dict[date]

        data_all_params = np.ma.array([data_dict[date]
                                       for date in all_data_keys[start_idx:end_idx]])

        for param_index, param in enumerate(BrdfModelParameters):

            # get the data iso, vol or geo from data which is a dict for all the keys and convert to numpy array
            data_param = data_all_params[:, param_index, :, :]

            # extract the value for a key and mask invalid data
            clean_data = data_param[index - start_idx]

            # replace bad index data with nan
            clean_data[bad_indices] = np.nan

            clean_data = np.ma.masked_invalid(clean_data)

            # get temporal local median value as set by filter size
            local_median = np.ma.median(data_param, axis=0)

            # apply threshold test to clean the data set
            threshold_idx = np.ma.where(np.abs(local_median - clean_data) > thresholds[param])

            # replace the data which did not pass threshold test with temporal local median value
            clean_data[threshold_idx] = local_median[threshold_idx]

            # convert to int16
            clean_data = clean_data * 1000
            clean_data = clean_data.filled(fill_value=32767).astype(np.int16)

            result[key][param_index, :, :] = clean_data

    return (window, result)


def apply_convolution(filename, h5_info, window, filter_size, mask_indices):
    """
    This function applies convolution on the clean dataset from applied threshold
    method.
    """
    all_data_keys = sorted(list(h5_info.keys()))

    # define a filter to be used in convolution and normalize to sum 1.0
    filt = gauss_filter(filter_size)
    filt = filt / np.sum(filt)

    def __get_clean_data(data_filename, data_key, data_window):
        with h5py.File(data_filename, 'r') as fid:
            d = fid[data_key][data_window]
            d = d.astype('float32')
            d[mask_indices] = np.nan
            return d

    data_convolved = {}

    for param_index, param in enumerate(BrdfModelParameters):
        temp = {}

        # get clean dataset for all available dates for given window
        data_clean = np.array([__get_clean_data(filename, key, (param_index,) + window) for key in all_data_keys])

        # set data that needs to be padded at the end and front to perform convolution
        data_head = np.array([data_clean[0] for i in range(filter_size)])
        data_tail = np.array([data_clean[len(data_clean)-1] for i in range(filter_size)])

        # pad the data_head and tail
        data_padded = np.concatenate((data_head, data_clean, data_tail), axis=0)

        # mask where data are invalid
        invalid_data = data_padded == 32767.0
        data_padded = np.where(~invalid_data, data_padded, np.nan)

        # get mean across temporal axis to fill the np.nan in data_padded array
        median_data = np.nanmedian(data_padded, axis=0)
        for index in range(data_padded.shape[0]):
            data_padded[index, :, :] = np.where(invalid_data[index, :, :], median_data, data_padded[index, :, :])

        # convert data to floating point with hard coded scale factor of 0.001
        data_padded = data_padded * 0.001

        # perform convolution using Gaussian filter defined above
        data_conv = np.apply_along_axis(lambda m: np.ma.convolve(m, filt, mode='same'), axis=0, arr=data_padded)
        data_conv = np.ma.masked_invalid(data_conv[filter_size:len(data_clean)+filter_size])

        for index, key in enumerate(all_data_keys):
            temp[key] = data_conv[index]

        data_convolved[param] = temp

    return data_convolved


def post_cleanup_process(window, set_doys, h5_info, outdir, tile, clean_data_file,
                         filter_size, band, bad_indices):
    """
    This function implements gaussian smoothing of the cleaned dataset,
    temporal averaging of the gaussian smooth dataset,
    quality check based on brdf_shape indices and writes the final
    brdf averaged parameters to a h5 file.
    """

    data_convolved = apply_convolution(clean_data_file, h5_info, window, filter_size,
                                       bad_indices[window])

    for doy in set_doys:
        avg_data = temporal_average(data_convolved, doy)
        filtered_data = brdf_indices_quality_check(avg_data=avg_data)

        outfile = pjoin(outdir, BRDF_AVG_FILE_FMT.format(tile, doy, band))
        with LOCKS[outfile]:
            with h5py.File(outfile) as fid:
                write_chunk(filtered_data, fid, band, window=(slice(None),) + window)


def write_brdf_fallback_band(brdf_dir, tile, band, outdir, filter_size,
                             pthresh=10.0, year_from=None, data_chunks=(1, 240, 240),
                             compute_chunks=(100, 2400), nprocs=None, compression=H5CompressionFilter.BLOSC_ZSTANDARD):

    with timing('calculate thresholds'):
        h5_info = hdf5_files(brdf_dir, tile=tile, year_from=year_from)

        min_numpix_required = np.rint((pthresh / 100.0) * len(h5_info))

        # get counts of good pixel quality
        quality_count = get_qualityband_count(h5_info=h5_info, band_name=quality_band_name(band))
        quality_count = np.ma.masked_invalid(quality_count)

        # get the index where band_quality number is less the minimum number of valid pixels required
        bad_indices = (quality_count < min_numpix_required).filled(False)

        shape, attrs = get_band_info(h5_info, albedo_band_name(band))
        shape = shape[-2:]

        thresholds = calculate_thresholds(h5_info, albedo_band_name(band), shape, compute_chunks, nprocs=nprocs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # TODO outdir -> tmp_dir
        clean_data_file = pjoin(outdir, 'clean_data_{}_{}.h5'.format(band, tile))

        with h5py.File(clean_data_file, 'w') as clean_data, timing('apply threshold'):
            for key in h5_info:
                create_dataset(clean_data, key, (3, shape[0], shape[1]), {})

            args = []
            for x, y in generate_tiles(shape[0], shape[1], compute_chunks[0], compute_chunks[1]):
                window = (slice(*y), slice(*x))
                args.append([h5_info, albedo_band_name(band), window, filter_size, thresholds, bad_indices[window]])

            if nprocs:
                with ProcessPool(processes=nprocs) as pool:
                    clean_data_shards = pool.starmap(apply_threshold, args)
            else:
                clean_data_shards = [apply_threshold(*arg) for arg in args]

            for window, entry in clean_data_shards:
                for key, value in entry.items():
                    clean_data[key][(slice(None),) + window] = value

        with timing('post cleanup'):
            set_doys = sorted(set(folder_doy(item) for item in h5_info))

            for doy in set_doys:
                print('doy', doy, file=stdout)

                average_metadata = {key: h5_info[key] for key in h5_info if folder_doy(key) == doy}

                # TODO create dataset to store uuid for brdf_fallback provenance (use h5_info for threhold generation
                # and average_metadata for average brdf parameters generation. h5_info and average_metadata are
                # dictionary(eg : key = '2002.01.01', value = 'absolute path to a h5 file'

                outfile = pjoin(outdir, BRDF_AVG_FILE_FMT.format(tile, doy, band))
                LOCKS[outfile] = Lock()

                with h5py.File(outfile, 'w') as fid:
                    create_brdf_datasets(fid, band, shape, attrs, chunks=data_chunks, compression=compression)

            args = []
            for x, y in generate_tiles(shape[0], shape[1], compute_chunks[0], compute_chunks[1]):
                window = (slice(*y), slice(*x))

                args.append([window, set_doys, h5_info, outdir, tile,
                             clean_data_file, filter_size, band, bad_indices])

            if nprocs:
                with ProcessPool(processes=nprocs) as pool:
                    pool.starmap(post_cleanup_process, args)
            else:
                _ = [post_cleanup_process(*arg) for arg in args]


@click.command()
@click.option('--brdf-dir', default='/g/data/v10/eoancillarydata.reS/fetch/BRDF/MCD43A1.006/')
@click.option('--outdir', default='/g/data/u46/users/pd1813/BRDF_PARAM/test_v9')
@click.option('--tile', default='h29v12')
@click.option('--band', default='Band1')
@click.option('--year-from', default=2002)
@click.option('--filter-size', default=22)
@click.option('--nprocs', default=15)
@click.option('--compression', default=H5CompressionFilter.BLOSC_ZSTANDARD)
def main(brdf_dir, outdir, tile, band, year_from, filter_size, nprocs, compression):
    write_brdf_fallback_band(brdf_dir, tile, band, outdir, filter_size, year_from=year_from, nprocs=nprocs,
                             compression=compression)


if __name__ == "__main__":
    main()
