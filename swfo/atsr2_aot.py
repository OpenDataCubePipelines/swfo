#!/usr/bin/env python

"""
Conversion utilities for CSIRO's custom *.pix and *.cmp file format
containing Aerosol Optical Thickness data.
"""

from datetime import datetime as dt
from pathlib import Path

from posixpath import join as ppjoin

import numpy
import h5py
import pandas
from shapely.geometry import Polygon
from shapely import wkt
from wagl.hdf5 import write_dataframe

from .h5utils import (
    generate_fallback_uuid, generate_md5sum
)


PRODUCT_HREF = 'https://collections.dea.ga.gov.au/ga_c_c_aerosol_2'


def read_pix(filename: Path):
    """
    The pix files are sparse 3D arrays.
    Will store as a Table and remove invalid data.
    """
    with filename.open('rb') as src:
        recs = numpy.fromfile(src, dtype='int32', count=3)
        xgrid = numpy.fromfile(src, dtype='float32', count=recs[0])
        ygrid = numpy.fromfile(src, dtype='float32', count=recs[1])
        idxlon = numpy.fromfile(src, dtype='int16', count=recs[2])
        idxlat = numpy.fromfile(src, dtype='int16', count=recs[2])
        date = numpy.fromfile(src, dtype='int16',
                              count=recs[2] * 3).reshape(3, recs[2])
        time = numpy.fromfile(src, dtype='int16',
                              count=recs[2] * 3).reshape(3, recs[2])
        # lat = numpy.fromfile(src, dtype='float32', count=recs[2])
        # lon = numpy.fromfile(src, dtype='float32', count=recs[2])
        aot = numpy.fromfile(src, dtype='float32', count=recs[2])

    obs_lon = xgrid[idxlon]
    obs_lat = ygrid[idxlat]
    timestamps = []
    for i in range(recs[2]):
        timestamps.append(dt(date[0, i], date[1, i], date[2, i],
                             time[0, i], time[1, i], time[2, i]))

    df = pandas.DataFrame({'timestamp': timestamps,
                           'lon': obs_lon,
                           'lat': obs_lat,
                           'aerosol': aot})

    # throw away bad data
    wh = (df['aerosol'] > 0.0) & (df['aerosol'] <= 1.0)
    df = df[wh]
    df.reset_index(inplace=True, drop=True)

    # get the minimum bounding box
    ul = (df['lon'].min(), df['lat'].max())
    ur = (df['lon'].max(), df['lat'].max())
    lr = (df['lon'].max(), df['lat'].min())
    ll = (df['lon'].min(), df['lat'].min())
    extents = Polygon([ul, ur, lr, ll])

    return df, extents


def read_cmp(filename: Path):
    """
    The cmp data is a 2D grid, but the pixel sizes are
    not constant (according to the lon and lat arrays.
    Will store as a Table and remove invalid data.
    """
    with filename.open('rb') as src:
        nx = numpy.fromfile(src, dtype='int32', count=1)[0]
        ny = numpy.fromfile(src, dtype='int32', count=1)[0]
        lon = numpy.fromfile(src, dtype='float32', count=nx)
        lat = numpy.fromfile(src, dtype='float32', count=ny)
        aot = numpy.fromfile(src, dtype='float32', count=nx*ny)

    obs_lon = []
    obs_lat = []
    for j in range(ny):
        for i in range(nx):
            obs_lon.append(lon[i])
            obs_lat.append(lat[j])

    df = pandas.DataFrame({'lon': obs_lon, 'lat': obs_lat, 'aerosol': aot})

    # throw away bad data
    wh = (df['aerosol'] > 0.0) & (df['aerosol'] <= 1.0)
    df = df[wh]
    df.reset_index(inplace=True, drop=True)

    # get the minimum bounding box
    ul = (df['lon'].min(), df['lat'].max())
    ur = (df['lon'].max(), df['lat'].max())
    lr = (df['lon'].max(), df['lat'].min())
    ll = (df['lon'].min(), df['lat'].min())
    extents = Polygon([ul, ur, lr, ll])

    return df, extents


def convert(aerosol_path, out_h5: h5py.Group, compression, filter_opts):
    """
    Converts all the .pix and .cmp files found in `aerosol_path`
    to a HDF5 file.
    """
    # define a case switch
    func = {'pix': read_pix, 'cmp': read_cmp}
    dataset_names = []
    metadata = []


    pattern = ['*.pix', '*.cmp']
    for p in pattern:
        for search_path in aerosol_path.glob(p):
            _path = search_path.resolve()
            fname, ext = _path.stem, _path.suffix[1:]  # exclude the period from ext
            out_path = ppjoin(ext, fname)
            df, extents = func[ext](_path)

            # read/write
            df, extents = func[ext](_path)

            # src checksum; used to help derive fallback uuid
            with _path.open('rb') as src:
                src_checksum = generate_md5sum(src).hexdigest()

            attrs = {'extents': wkt.dumps(extents),
                     'source filename': str(_path)}
            write_dataframe(df, out_path, out_h5, compression=compression,
                            attrs=attrs, filter_opts=filter_opts)
            dataset_names.append(out_path)
            metadata.append({
                'id': str(generate_fallback_uuid(
                    PRODUCT_HREF,
                    path=str(_path.stem),
                    md5=src_checksum)
                         )
            })

    return metadata, dataset_names
