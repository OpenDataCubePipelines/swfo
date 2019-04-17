#!/usr/bin/env python

"""
Utility for converting GA's monthly ozone TIFF's to HDF5.
"""

from pathlib import Path
import h5py
import rasterio

from wagl.hdf5 import write_h5_image


def convert(indir, out_fname, compression, filter_opts):
    """
    Convert GA's ozone TIFF's to HDF5.
    The TIFF's will be converted into HDF5 IMAGE Datasets,
    and contained within a single HDF5 file.
    """
    # convert to Path object
    indir = Path(indir)

    # create empty or copy the user supplied filter options
    if not filter_opts:
        filter_opts = dict()
    else:
        filter_opts = filter_opts.copy()

    with h5py.File(str(out_fname), 'w') as fid:
        for fname in indir.glob('*.tif'):
            with rasterio.open(str(fname)) as rds:
                # the files have small dimensions, so store as a single chunk
                if 'chunks' not in filter_opts:
                    filter_opts['chunks'] = (rds.height, rds.width)

                attrs = {
                    'description': 'Ozone data compiled by Geoscience Australia',
                    'geotransform': rds.transform.to_gdal(),
                    'crs_wkt': rds.crs.wkt
                }

                # output
                dname = fname.stem
                write_h5_image(rds.read(1), dname, fid, compression, attrs,
                               filter_opts)
