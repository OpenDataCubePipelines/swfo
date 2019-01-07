#!/bin/bash

"""
Create timeseries averages for the NOAA water vapour data.
"""

from datetime import datetime
from pathlib import Path
import numpy
import h5py
import pandas
from wagl.geobox import GriddedGeoBox
from wagl.hdf5 import read_h5_table, write_h5_image


def build_index(indir):
    """
    Read the INDEX table for each file and build a full history
    index.
    The records are sorted in ascending time (earliest to most recent)
    """
    df = pandas.DataFrame(columns=['filename', 'band_name', 'timestamp'])
    for fname in Path(indir).glob("pr_wtr.eatm.*.h5"):
        with h5py.File(str(fname), 'r') as fid:
            tmp_df = read_h5_table(fid, 'INDEX')
            tmp_df['filename'] = fid.filename
            df = df.append(tmp_df)

    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    return df


def calculate_average(dataframe):
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
            ds = fid[row.band_name]
            ds.read_direct(data[i])

    # get the geobox
    with h5py.File(row.filename, "r") as fid:
        ds = fid[row.band_name]
        geobox = GriddedGeoBox.from_dataset(ds)
        chunks = ds.chunks

    mean = data.mean(axis=0)

    return mean, geobox, chunks


def prwtr_average(indir, outdir, compression, filter_opts):
    """
    Take the 4 hourly daily average from all files.
    """
    df = build_index(indir)

    # grouping
    groups = df.groupby([df.index.month, df.index.day, df.index.hour])

    # create directories as needed
    out_fname = outdir.joinpath("pr_wtr_average.h5")
    if not out_fname.parent.exists():
        out_fname.parent.mkdir(parents=True)

    # create output file
    with h5py.File(str(out_fname), 'w') as fid:

        # the data is ordered so we can safely use BAND-1 = Jan-1
        for band_index, item in enumerate(groups):
            grp_name, grp_df = item

            # synthesised leap year timestamp (use year 2000)
            fmt = "2000 {:02d} {:02d} {:02d}"
            dtime = datetime.strptime(fmt.format(*grp_name), "2000 %m %d %H")

            # mean
            mean, geobox, chunks = calculate_average(grp_df)

            # dataset name format "%B-%d-%H" eg FEBRUARY-06-18 for Feb 6th 1800 hrs
            dname = dtime.strftime("%B-%d-%H").upper()

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
                "date_format": "2000 %B-%d-%H",
                "band_name": "BAND-{}".format(band_index +1),
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
