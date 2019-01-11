#!/usr/bin/env python

"""
BRDF data extraction utilities from hdf5 files
"""

import os
import sys
import rasterio
import rasterio.plot as plot
from os.path import join as pjoin, basename
import timeit
import h5py
import fnmatch
import datetime
import numpy as np
from numpy import vstack
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

SDS_BAND_QUALITY_NAME = {}


class JsonEncoder(json.JSONEncoder):
    """
    A wrapper class to address the issue of json encoding error
    This class handles  the json serializing error for numpy
    datatype: 'float32' and numpy arrays
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


def get_h5_lists(brdf_dir=None, folder_fmt='%Y.%m.%d'):
    """
    A function to extract all the MODIS BRDF acquisition details
    from the root folder where BRDF data are stored.

    :param brdf_dir:
        A path name to where BRDF data are stored
        The BRDF directories are assumed to be yyyy.mm.dd naming convention.
    :return:
        a nested dict containing  tile numbers and dates and BRDF file path
        which can be accessed through a key param defined by a folder name

    """
    brdf_data = {}

    for item in os.listdir(brdf_dir):
        print(item)
        files = [f for f in os.listdir(pjoin(brdf_dir, item))]
        h5_names = {}
        try:
            filename = fnmatch.filter(files, '*.h5')[0]
            file_path = pjoin(brdf_dir, item, filename)
            h5_names['path'] = file_path
        except IndexError:
            h5_names['path'] = None

        h5_names['datetime'] = datetime.datetime.strptime(item, folder_fmt)
        brdf_data[item] = h5_names

    return brdf_data


def get_h5_data_attrs(h5file, sds_name):

    with h5py.File(h5file, 'r') as src:
        attrs = {k: v for k, v in src[sds_name].attrs.items()}

        if fnmatch.fnmatch(sds_name, '*Quality*'):
            nodata_val = 255
            scale = 1
            offset = 0

            attrs['scale'] = scale
            attrs['offset'] = offset
        else:
            nodata_val = 32767

        attrs['nodata'] = nodata_val

        data = src[sds_name][:]

    return data, attrs


def mask_and_scale(data, metadata, apply_scale=False):

    if not type(data) is np.ndarray:
        data = np.array(data)

    nodata = metadata['nodata']
    scale = metadata['scale']
    offset = metadata['offset']

    masked_data = np.ma.masked_where(data == nodata, data)

    if apply_scale:
        scaled_masked_data = masked_data * float(scale) + float(offset)
    else:
        scaled_masked_data = masked_data

    return scaled_masked_data


def sum_time_series(data):
    """expects data array to be masked array"""

    if not np.ma.is_masked(data):
        data = np.ma.array(data)

    return np.ma.sum(data, axis=0)


def get_sds_doy_dataset(dirs=None, dayofyear=None, sds_name=None, year=None, apply_scale=False):
    """
    returns masked numpy array of specific sds scaled and masked data
    for particular day of the year from all the hdf5 files in a
    given directory
    """

    h5_lists_dict = get_h5_lists(brdf_dir=dirs)
    keys = np.sort([k for k in h5_lists_dict])
    data_all = []

    for k in keys:
        dt = (h5_lists_dict[k]['datetime'])
        doy = dt.timetuple().tm_yday
        yr = dt.timetuple().tm_year

        if doy == dayofyear and yr > year:
            h5_file = (h5_lists_dict[k]['path'])
            data, metadata = get_sds_data(h5_file, sds_name)
            scaled_data = mask_and_scale(data, metadata, apply_scale=apply_scale)
            print(scaled_data[5000:6000, 9000:15000])

            data_all.append(scaled_data)

    return np.ma.array(data_all)


def main():

    brdf_dir = '/g/data/u46/users/pd1813/BRDF_PARAM/MCD43A1_C6_HDF5_TEST/'
    sds_name = SDS_BAND_NAME['QUAL_BAND1']
    apply_scale = False
    dataset = get_sds_doy_dataset(dirs=brdf_dir, dayofyear=185, sds_name=sds_name, year=2010, apply_scale=apply_scale)

    mean_data = sum_time_series(dataset)




    # print(mean_data.shape)
    plot.show(mean_data)
    # print(mean_data.shape)
    # print(mean_data[5000:6000, 9000:15000])
    #
    #print(timeit.default_timer() - start_time)


if __name__ == "__main__":
    main()