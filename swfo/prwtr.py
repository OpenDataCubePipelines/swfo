#!/usr/bin/env python

"""
Convert PR_WTR NetCDF files to HDF5.
"""

from datetime import datetime
from pathlib import Path
import json
import numpy
import osr
import rasterio
import h5py
import pandas

from wagl.geobox import GriddedGeoBox
from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import read_h5_table, write_h5_image, write_dataframe
from wagl.hdf5 import attach_attributes


CRS = osr.SpatialReference()
CRS.ImportFromEPSG(4326)


def convert_file(fname, out_h5: h5py.Group, compression, filter_opts):
    """
    Convert a PR_WTR NetCDF file into HDF5.

    :param fname:
        A str containing the PR_WTR filename.

    :param out_fname:
        A h5py.Group to write output datasets to

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
        df['dataset_name'] = df.timestamp.dt.strftime('%Y/%B-%d/%H%M')
        df['dataset_name'] = df['dataset_name'].str.upper()

        # create a timestamp and band name index table dataset
        desc = "Timestamp and Band Name index information."
        attrs = {
            'description': desc
        }
        write_dataframe(df, 'INDEX', out_h5, compression, attrs=attrs)

        attach_attributes(out_h5, g_attrs)

        # process every band
        for i in range(1, ds.count + 1):
            ds_name = df.iloc[i-1].dataset_name

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
            attrs['band_name'] = df.iloc[i-1]['band_name']
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
            write_h5_image(ds.read(i), ds_name, out_h5, attrs=attrs,
                           compression=compression, filter_opts=f_opts)


def _build_index(indir):
    """
    Read the INDEX table for each file and build a full history
    index.
    The records are sorted in ascending time (earliest to most recent)
    """
    df = pandas.DataFrame(columns=['filename', 'band_name', 'timestamp'])
    for fname in Path(indir).glob("pr_wtr.eatm.[0-9]*.h5"):
        with h5py.File(str(fname), 'r') as fid:
            tmp_df = read_h5_table(fid, 'INDEX')
            tmp_df['filename'] = fid.filename
            df = df.append(tmp_df)

    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    return df


def _average(dataframe):
    """
    Given a dataframe with the columns:
        * filename
        * band_name

    Calculate the 3D/timeseries average from all input records.
    Each 2D dataset has dimensions (73y, 144x), and type float32.
    """
    dims = (dataframe.shape[0], 73, 144)
    data = numpy.zeros(dims, dtype="float32")

    # load all data into 3D array (dims are small so just read all)
    for i, rec in enumerate(dataframe.iterrows()):
        row = rec[1]
        with h5py.File(row.filename, "r") as fid:
            ds = fid[row.dataset_name]
            ds.read_direct(data[i])
            no_data = float(ds.attrs['missing_value'])

        # check for nodata and convert to nan
        # do this for each dataset in case the nodata value changes
        data[i][data[i] == no_data] = numpy.nan

    # get the geobox, chunks
    with h5py.File(row.filename, "r") as fid:
        ds = fid[row.dataset_name]
        geobox = GriddedGeoBox.from_dataset(ds)
        chunks = ds.chunks

    mean = numpy.nanmean(data, axis=0)

    return mean, geobox, chunks


def fallback(indir, outdir, compression=H5CompressionFilter.LZF,
             filter_opts=None):
    """
    Take the 4 hourly daily average from all files.
    """
    df = _build_index(indir)

    # grouping
    groups = df.groupby([df.index.month, df.index.day, df.index.hour])

    # create directories as needed
    out_fname = Path(outdir).joinpath("pr_wtr.eatm.average.h5")
    if not out_fname.parent.exists():
        out_fname.parent.mkdir(parents=True)

    # create output file
    with h5py.File(str(out_fname), 'w') as fid:

        # the data is ordered so we can safely use BAND-1 = Jan-1
        for band_index, item in enumerate(groups):
            grp_name, grp_df = item

            # synthesised leap year timestamp (use year 2000)
            fmt = "2000 {:02d} {:02d} {:02d}"
            dtime = datetime.strptime(fmt.format(*grp_name), "%Y %m %d %H")

            # mean
            mean, geobox, chunks = _average(grp_df)

            # dataset name format "%B-%d/%H%M" eg FEBRUARY-06/1800 for Feb 6th 1800 hrs
            dname = "AVERAGE/{}".format(dtime.strftime("%B-%d/%H%M").upper())

            # dataset description
            description = ("Average data for {year_month} {hour}00 hours, "
                           "over the timeperiod {dt_min} to {dt_max}")
            description = description.format(
                year_month=dtime.strftime("%B-%d"),
                hour=dtime.strftime("%H"),
                dt_min=grp_df.index.min(),
                dt_max=grp_df.index.max()
            )

            # dataset attributes
            attrs = {
                "description": description,
                "timestamp": dtime,
                "date_format": "2000 %B-%d/%H%M",
                "band_name": "BAND-{}".format(band_index + 1),
                "geotransform": geobox.transform.to_gdal(),
                "crs_wkt": geobox.crs.ExportToWkt()
            }

            # create empty or copy the user supplied filter options
            if not filter_opts:
                f_opts = dict()
            else:
                f_opts = filter_opts.copy()

            # use original chunks if none are provided
            if 'chunks' not in f_opts:
                f_opts['chunks'] = chunks

            # write
            write_h5_image(mean, dname, fid, attrs=attrs,
                           compression=compression, filter_opts=f_opts)
