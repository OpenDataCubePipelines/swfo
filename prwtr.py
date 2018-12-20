#!/usr/bin/env python

"""
Convert PR_WTR NetCDF files to HDF5.
"""

import json
import osr
import rasterio
import h5py
import pandas

from wagl.hdf5 import write_h5_image, attach_attributes, write_dataframe


CRS = osr.SpatialReference()
CRS.ImportFromEPSG(4326)


def convert_file(fname, out_fname, compression, filter_opts):
    """
    Convert a PR_WTR NetCDF file into HDF5.

    :param fname:
        A str containing the PR_WTR filename.

    :param out_fname:
        A str containing the output filename for the HDF5 file.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :return:
        None. Content is written directly to disk.
    """
    with h5py.File(out_fname, 'w') as fid:
        with rasterio.open(fname) as ds:
            name_fmt = 'BAND-{}'

            # global attributes
            # TODO update the history attrs
            # TODO remove the NC_GLOBAL str and just have plain attr names
            g_attrs = ds.tags()

            # get timestamp info
            origin = g_attrs.pop('time#units').replace('hours since ', '')
            hours = json.loads(
                g_attrs.pop('NETCDF_DIM_time_VALUES').replace('{', '[').replace('}', ']')
            )
            df = pandas.DataFrame(
                {
                    'timestamp': pandas.to_datetime(hours, unit='h', origin=origin),
                    'band_name': [name_fmt.format(i+1) for i in range(ds.count)]
                }
            )

            # create a timestamp and band name index table dataset
            desc = "Timestamp and Band Name index information."
            attrs = {
                'description': desc
            }
            write_dataframe(df, 'INDEX', fid, compression, attrs=attrs)

            attach_attributes(fid, g_attrs)

            # process every band
            for i in range(1, ds.count + 1):
                ds_name = name_fmt.format(i)

                # create empty or copy the user supplied filter options
                if not filter_opts:
                    f_opts = dict()
                else:
                    f_opts = filter_opts.copy()


                # band attributes
                # TODO remove NETCDF tags
                # TODO add fillvalue attr
                attrs = ds.tags(i)
                attrs['timestamp'] = df.iloc[i-1]['timestamp']
                attrs['geotransform'] = ds.transform.to_gdal()
                attrs['crs_wkt'] = CRS.ExportToWkt()

                # use ds native chunks if none are provided
                if 'chunks' not in f_opts:
                    try:
                        f_opts['chunks'] = ds.block_shapes[i]
                    except IndexError:
                        print("Chunk error: {}".format(fname))
                        f_opts['chunks'] = (73, 144)

                # write to disk as an IMAGE Class Dataset
                write_h5_image(ds.read(i), ds_name, fid, attrs=attrs,
                               compression=compression, filter_opts=f_opts)
