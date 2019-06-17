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
from typing import Optional, Dict
import uuid
from pathlib import Path

from multiprocessing import Pool as ProcessPool, Lock
import fnmatch
import h5py
import numpy as np
import click

from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import attach_image_attributes
from wagl.tiling import generate_tiles
from wagl.constants import BrdfModelParameters

from . import brdf_shape
from .h5utils import write_h5_md, YAML

BAND_LIST = ['Band{}'.format(band) for band in range(1, 8)]
FALLBACK_PRODUCT_HREF = 'https://collections.dea.ga.gov.au/ga_c_m_brdfalbedo_2'

BRDF_AVG_FILE_BAND_FMT = 'MCD43A1.JLAV.006.{}.DOY.{:03}.{}.h5'
BRDF_AVG_FILE_FMT = 'MCD43A1.JLAV.006.{}.DOY.{:03}.h5'
BRDF_MATCH_PATTERN = '*{}.DOY.{:03}*Band*.h5'

LOCKS = {}

NODATA = 32767

SCALE_FACTOR = 0.0001
INV_SCALE_FACTOR = 10000

# for clean data interim file
SCALE_FACTOR_2 = 0.001
INV_SCALE_FACTOR_2 = 1000


def get_datetime(dt: Optional[datetime.datetime] = None):
    """
    Returns a datetime object (defaults to utcnow) with tzinfo set to utc

    :param dt:
        (Optional) a datetime object to add utcnow to; defaults to utcnow()
    :return:
        A 'datetime' type with tzinfo set
    """
    if not dt:
        dt = datetime.datetime.utcnow()
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt


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


def shape_of_window(window):
    y_shape = window[0].stop - window[0].start
    x_shape = window[1].stop - window[1].start
    return (y_shape, x_shape)


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


def hdf5_files(brdf_dir, tile, year_from=None, year_to=None):
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
        elif year_to is not None and folder_year(item) > year_to:
            continue

        files = os.listdir(pjoin(brdf_dir, item))
        try:
            filename = fnmatch.filter(files, '*.{}*.h5'.format(tile))[0]
            h5_info[item] = pjoin(brdf_dir, item, filename)
        except IndexError:
            pass

    return h5_info


def read_brdf_quality_dataset(ds, window=None):
    """
    :param ds:
        A 'file object' type: hdf5 file object containing the BRDF quality data set.
    :param window:
        A 'slice object' type: contain the set of indices specified by
        range(start, stop, step). Default=None, results in reading whole
        data set.
    :return:
        A 'array' type: slice of data set specified by window if set or all
        the data set. The BRDF parameter are scale and offset factor
        corrected and filled with numpy 'nan' for no data values.
    """
    if window is None:
        window = slice(None)

    data = ds[window].astype('float32')

    nodata_mask = (data == float(ds.attrs['_FillValue']))
    data[nodata_mask] = np.nan
    
    return data


def read_brdf_dataset(ds, param,  window=None):
    """
    :param ds:
        A 'file object' type: hdf5 file object containing the BRDF data set
        as a numpy structured data.
    :param param:
        A 'str' type: name of brdf parameter 
    :param window:
        A 'slice object' type: contain the set of indices specified by
        range(start, stop, step). Default=None, results in reading whole
        data set.
    :return:
        A 'array' type: slice of data set specified by window if set or all
        the data set. The BRDF parameter are scale and offset factor
        corrected and filled with numpy 'nan' for no data values.
    """
    if window is None:
        window = slice(None)

    data = ds[window][param].astype('float32')
    
    nodata_mask = (data == float(ds.attrs['_FillValue']))
    data[nodata_mask] = np.nan

    scale_factor = ds.attrs['scale_factor']
    add_offset = ds.attrs['add_offset']

    return scale_factor * (data - add_offset)


def get_qualityband_count_window(h5_info, band_name, window):
    def read_quality_data(filename):
        with h5py.File(filename, 'r') as fid:
            return 1. - read_brdf_quality_dataset(fid[band_name], window)

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

    return window, data_sum


def get_qualityband_count(h5_info, band_name, shape, compute_chunks, nprocs):
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
    data_sum = np.zeros(shape=shape, dtype='int16')

    with Pool(processes=nprocs) as pool:
        results = pool.starmap(get_qualityband_count_window,
                               [(h5_info, band_name, window)
                                for window in generate_windows(shape, compute_chunks)])

    for window, data in results:
        data_sum[window] = data

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


def get_std_block(h5_info, band_name, param, window):
    """
    A function to compute a standard deviation for across a temporal axis.
    This function was written to facilitate parallel processing.
    """

    def __get_data(dat_filename, param, window):
        with h5py.File(dat_filename, 'r') as fid:
            dat = read_brdf_dataset(fid[band_name], param, window)
            return dat

    data = np.zeros((len(h5_info),) + shape_of_window(window), dtype='float32')
    for layer, filename in enumerate(h5_info.values()):
        data[layer] = __get_data(filename, param,  window)

    run_median = np.nanmedian(data, axis=0, keepdims=False)

    for item in range(data.shape[0]):
        idx = np.where(np.isnan(data[item]))
        data[item][idx] = run_median[idx]

    return np.nanstd(data, axis=0, ddof=1, keepdims=False)


def concatenate_files(infile_paths, outfile, avg_metadata, threshold_metadata):
    """
    A function to concatenate multiple h5 files
    """
    if os.path.exists(outfile):
        os.remove(outfile)

    assert len(infile_paths) == 7

    # TODO create dataset to store uuid for brdf_fallback provenance (use h5_info for threshold generation
    # and avg_metadata for average brdf parameters generation. h5_info and average_metadata are
    # dictionary(eg : key = '2002.01.01', value = 'absolute path to a h5 file'

    with h5py.File(outfile, 'w') as out_fid:
        for fp in infile_paths:
            with h5py.File(fp, 'r') as in_fid:
                for ds_band in in_fid:
                    in_fid.copy(source=ds_band, dest=out_fid)


def generate_windows(shape, compute_chunks):
    for x, y in generate_tiles(shape[0], shape[1], compute_chunks[0], compute_chunks[1]):
        yield (slice(*y), slice(*x))


class DummyPool:
    def __enter__(self):
        return self

    def starmap(self, func, args):
        return [func(*arg) for arg in args]


def Pool(processes):
    if not processes:
        return DummyPool()

    return ProcessPool(processes=processes)


def calculate_thresholds(h5_info, band_name, shape, compute_chunks, nprocs=None):
    """
    Computes threshold needed needed to clean temporal series
    """
    thresh_dict = {}
    for param in BrdfModelParameters:
        with Pool(nprocs) as pool:
            results = pool.starmap(get_std_block,
                                   [(h5_info, band_name, param.value, window)
                                    for window in generate_windows(shape, compute_chunks)])

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

    attrs = dict(scale_factor=SCALE_FACTOR, add_offset=0,
                 _FillValue=NODATA, bands="{}: 1, {}: 2:, {}: 3".format(BrdfModelParameters.ISO.value,
                                                                        BrdfModelParameters.VOL.value,
                                                                        BrdfModelParameters.GEO.value),
                 description=('BRDF albedo parameters (iso, vol and geo)'
                              ' derived from {}'
                              ' in lognormal space'.format(albedo_band_name(band_name))),
                 **common_attrs)
    create_dataset(group, 'BRDF_Albedo_Parameters_{}'.format(band_name),
                   (3,) + shape, attrs,
                   chunks=chunks, filter_opts=filter_opts, compression=compression)

    attrs = dict(scale_factor=SCALE_FACTOR, add_offset=0,
                 _FillValue=NODATA, bands="afx: 1, rms: 2",
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

    data_main = data_main * INV_SCALE_FACTOR
    data_main = np.rint(data_main).filled(fill_value=NODATA).astype(np.int16)

    fid['BRDF_Albedo_Parameters_{}'.format(band_name)][window] = data_main

    data_support = data_support * INV_SCALE_FACTOR
    data_support = np.rint(data_support).filled(fill_value=NODATA).astype(np.int16)
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
        data_param = np.array([data[param][key] for key in data[param].keys() if folder_doy(key) == doy])
        data_param = np.ma.masked_invalid(data_param)
        tmp[param] = dict(mean=np.ma.mean(data_param, axis=0),
                          std=np.ma.std(data_param, ddof=0, axis=0),
                          num=data_param.count(axis=0))
    return {doy: tmp}


def apply_threshold(clean_data_file, h5_info, band_name, window, filter_size, thresholds, bad_indices):
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

    def get_albedo_data(filename, param, window):
        with h5py.File(filename, 'r') as fid:
            return read_brdf_dataset(fid[band_name], param, window)

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
                data_dict[date] = np.ma.masked_invalid(np.array([get_albedo_data(h5_info[date], param.value, window) 
                                                                for param in BrdfModelParameters]))

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
            clean_data = clean_data * INV_SCALE_FACTOR_2
            clean_data = clean_data.filled(fill_value=NODATA).astype(np.int16)

            with LOCKS[clean_data_file]:
                with h5py.File(clean_data_file) as output:
                    output[key][(param_index,) + window] = clean_data


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
        data_clean = np.full(shape=(2 * filter_size + len(all_data_keys),) + shape_of_window(window),
                             fill_value=np.nan, dtype='float32')

        for layer, key in enumerate(all_data_keys):
            data_clean[filter_size + layer] = __get_clean_data(filename, key, (param_index,) + window)

        # set data that needs to be padded at the end and front to perform convolution
        data_clean[:filter_size] = np.array([data_clean[filter_size] for i in range(filter_size)])
        data_clean[len(all_data_keys) + filter_size:] = np.array([data_clean[filter_size + len(all_data_keys) - 1]
                                                                  for i in range(filter_size)])

        # mask where data are invalid
        invalid_data = data_clean == NODATA
        data_clean = np.where(~invalid_data, data_clean, np.nan)

        # get mean across temporal axis to fill the np.nan in data_clean array
        median_data = np.nanmedian(data_clean, axis=0)
        for index in range(data_clean.shape[0]):
            data_clean[index] = np.where(invalid_data[index], median_data, data_clean[index])

        data_clean *= SCALE_FACTOR_2

        # perform convolution using Gaussian filter defined above
        for i in range(data_clean.shape[1]): 
            for j in range(data_clean.shape[2]):
                data_clean[:, i, j] = np.convolve(data_clean[:, i, j], filt, mode='same')

        for index, key in enumerate(all_data_keys):
            temp[key] = data_clean[index + filter_size]

        data_convolved[param] = temp

    return data_convolved


def generate_fallback_metadata_doc(h5info: Dict, doy: str):
    """
    Generates a metadata doc for a brdf fallback dataset attributes are derived
    from the metadata of the source dataset; assume metadata reference is ga-stac-like
    :param h5info:
        A dictionary of day-of-year strings to h5 collection references
    :param doy:
        A string of the day of year being averaged for the tile

    :return:
        A metadata dictionary describing the dataset
    """
    def _reader(filepath: str, offset='/METADATA/CURRENT'):
        with h5py.File(filepath, 'r') as src:
            ds = YAML.load(src[offset][()].item())
        return ds

    metadata_doc = {
        'id': str(uuid.uuid4()),  # Not sure what the params would be for deterministic uuid
        'product': {'href': FALLBACK_PRODUCT_HREF},
        'crs': None,
        'geometry': None,  # Need to validate imagery intersection
        'grids': None,  # Need to verify quality mask
        'measurements': None,  # TODO
        'lineage': {
            'brdf_threshold': [],
            'doy_average': [],
        },
        'properties': {}
    }
    start_ds, *_, end_ds = sorted(h5info.keys())

    for datestr, fp in h5info.items():
        src_md = _reader(fp)
        curr_doy = datetime.datetime.strptime(datestr, '%Y.%m.%d').strftime('%j')
        metadata_doc['lineage']['brdf_threshold'].append(src_md['id'])
        if curr_doy == doy:
            metadata_doc['lineage']['doy_average'].append(src_md['id'])

        if not metadata_doc.get('grids', None):
            metadata_doc['crs'] = src_md['crs']
            metadata_doc['geometry'] = src_md['geometry']
            metadata_doc['grids'] = src_md['grids']
            metadata_doc['measurements'] = '???'  # Need to know the measurements
            metadata_doc['properties'] = {
                'dtr:start_datetime': get_datetime(
                    datetime.datetime.strptime(start_ds, '%Y.%m.%d')).isoformat(),
                'dtr:end_datetime': get_datetime(
                    datetime.datetime.strptime(end_ds, '%Y.%m.%d')).isoformat(),
                'eo:instrument': src_md['properties']['eo:instrument'],
                'eo:platform': src_md['properties']['eo:platform'],
                'eo:gsd': src_md['properties']['eo:gsd'],
                'eo:epsg': src_md['properties']['eo:epsg'],
                'item:providers': [{
                    'name': 'Geoscience Australia',
                    'roles': ['processor', 'host'],
                }],
                'odc:creation_datetime': get_datetime().isoformat(),
                'odc:file_format': 'HDF5',
                'odc:region_code': src_md['properties']['odc:region_code']
            }

    return metadata_doc


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

        outfile = pjoin(outdir, BRDF_AVG_FILE_BAND_FMT.format(tile, doy, band))
        with LOCKS[outfile]:
            with h5py.File(outfile) as fid:
                write_chunk(filtered_data, fid, band, window=(slice(None),) + window)
                # md = generate_fallback_metadata_doc(h5_info, doy)
                # write_h5_md(fid, md)


def write_brdf_fallback_band(h5_info, tile, band, outdir, filter_size, set_doys,
                             pthresh, data_chunks, compute_chunks, nprocs, compression):
    """
    Computes pre-MODIS BRDF for a single band for every unique day of the year (from set_doys)
    """

    min_numpix_required = np.rint((pthresh / 100.0) * len(h5_info))

    shape, attrs = get_band_info(h5_info, albedo_band_name(band))
    shape = shape[-2:]

    # get counts of good pixel quality
    quality_count = get_qualityband_count(h5_info=h5_info, band_name=quality_band_name(band),
                                          shape=shape, compute_chunks=compute_chunks, nprocs=nprocs)
    quality_count = np.ma.masked_invalid(quality_count)

    # get the index where band_quality number is less the minimum number of valid pixels required
    bad_indices = (quality_count < min_numpix_required).filled(False)

    thresholds = calculate_thresholds(h5_info, albedo_band_name(band), shape, compute_chunks, nprocs=nprocs)
    quality_count = None
    
    clean_data_file = pjoin(outdir, 'clean_data_{}_{}.h5'.format(band, tile))
    LOCKS[clean_data_file] = Lock()
  
    with h5py.File(clean_data_file, 'w') as clean_data:
        for key in h5_info:
            create_dataset(clean_data, key, (3, shape[0], shape[1]), {})

    with Pool(processes=nprocs) as pool:
        pool.starmap(apply_threshold,
                     [(clean_data_file, h5_info, albedo_band_name(band),
                       window, filter_size, thresholds, bad_indices[window])
                      for window in generate_windows(shape,
                                                     compute_chunks=compute_chunks)])

    for doy in set_doys:
        outfile = pjoin(outdir, BRDF_AVG_FILE_BAND_FMT.format(tile, doy, band))
        LOCKS[outfile] = Lock()

        with h5py.File(outfile, 'w') as fid:
            create_brdf_datasets(fid, band, shape, attrs, chunks=data_chunks, compression=compression)

    with Pool(processes=nprocs) as pool:
        pool.starmap(post_cleanup_process,
                     [(window, set_doys, h5_info, outdir, tile,
                       clean_data_file, filter_size, band, bad_indices)
                      for window in generate_windows(shape, compute_chunks)])

    os.remove(clean_data_file)


def write_brdf_fallback(brdf_dir, outdir, tile, year_from, year_to, filter_size, nprocs, compression):
    """
    Generates a single h5 files per day of the year containing first seven MODIS bands as sub-dataset.
    """
    h5_info = hdf5_files(brdf_dir, tile=tile, year_from=year_from, year_to=year_to)
    set_doys = sorted(set(folder_doy(item) for item in h5_info))

    with tempfile.TemporaryDirectory() as tmp_dir:
        for band in BAND_LIST:
            write_brdf_fallback_band(h5_info, tile, band, tmp_dir, filter_size, set_doys,
                                     pthresh=10.0, data_chunks=(1, 240, 240), compute_chunks=(240, 240),
                                     nprocs=nprocs, compression=compression)
        with Pool(processes=nprocs) as pool: 
            pool.starmap(concatenate_files, [([str(fp) for fp in Path(tmp_dir).rglob(BRDF_MATCH_PATTERN
                                                                                     .format(tile, doy))],
                                              os.path.join(outdir, BRDF_AVG_FILE_FMT.format(tile, doy)),
                                              {key: h5_info[key] for key in h5_info if folder_doy(key) == doy},
                                              h5_info)
                                             for doy in set_doys])


@click.command()
@click.option('--brdf-dir', default='/g/data/v10/eoancillarydata.reS/brdf.av/MCD43A1.006/')
@click.option('--outdir', default='/g/data/u46/users/pd1813/BRDF_PARAM/test_struct')
@click.option('--tile', default='h29v12')
@click.option('--year-from', default=2002)
@click.option('--year-to', default=2018)
@click.option('--filter-size', default=22)
@click.option('--nprocs', default=27)
@click.option('--compression', default=H5CompressionFilter.BLOSC_ZSTANDARD)
def main(brdf_dir, outdir, tile, year_from, year_to, filter_size, nprocs, compression):
    write_brdf_fallback(brdf_dir, outdir, tile, year_from, year_to, filter_size, nprocs, compression)


if __name__ == "__main__":
    main()
