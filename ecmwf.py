#!/usr/bin/env python

"""
Conversion utilities for converting ECMWF
(European Centre for Medium Weather Forecast) GRIB files to HDF5.
"""

from datetime import datetime, timezone
from pathlib import Path
import rasterio
import h5py
import pandas

from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import attach_attributes, attach_image_attributes
from wagl.hdf5 import write_h5_image, write_dataframe
from wagl.tiling import generate_tiles


def metadata_dataframe(dataset):
    """
    Retrieves the metadata tags for a list of bands from a `GDAL`
    compliant dataset and structures it into a pandas.DataFrame.

    :param dataset:
        An opened rasterio dataset

    :return:
        A `pandas.DataFrame`.
    """
    # column names
    tag_data = {k: [] for k in dataset.tags(1).keys()}
    tag_data['timestamp'] = []
    tag_data['description'] = []

    for band in range(1, dataset.count + 1):
        tags = dataset.tags(band)
        for tag in tags:
            tag_data[tag].append(tags[tag])

        # create datetime object from seconds stored as str
        seconds = int(tags['GRIB_REF_TIME'].strip().split(' ')[0])
        tstamp = datetime.fromtimestamp(seconds, timezone.utc)
        tag_data['timestamp'].append(tstamp)

        # description
        tag_data['description'].append(dataset.descriptions[band -1])

    return pandas.DataFrame(tag_data)


def _convert_2d(rds, fid, dataset_name, compression, filter_opts):
    """
    Private routine for converting the 2D GRIB file to HDF5.
    """
    attrs = {
        'geotransform': rds.transform.to_gdal(),
        'crs_wkt': rds.crs.wkt,
        'history': 'Converted to HDF5'
    }
    data = rds.read(1)
    write_h5_image(data, dataset_name, fid, compression, attrs, filter_opts)

    # add dimension labels, but should we also include dimension scales?
    dataset = fid[dataset_name]
    dataset.dims[0].label = 'Y'
    dataset.dims[1].label = 'X'

    # metadata
    metadata = metadata_dataframe(rds)
    write_dataframe(metadata, 'METADATA', fid, compression)


def _convert_3d(rds, fid, dataset_name, compression, filter_opts):
    """
    Private routine for converting the 37 layer atmospheric data
    in the GRIB file to HDF5.
    """
    # basic metadata to attach to the dataset
    attrs = {
        'geotransform': rds.transform.to_gdal(),
        'crs_wkt': rds.crs.wkt,
        'history': 'Converted to HDF5'
    }

    # bands list, nrows to process (ytile)
    bands = list(range(1, rds.count + 1))
    ytile = filter_opts['chunks'][1]
    dims = (rds.count, rds.height, rds.width)

    # dataset creation options
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    kwargs['shape'] = dims
    kwargs['dtype'] = rds.dtypes[0]

    dataset = fid.create_dataset(dataset_name, **kwargs)
    attach_image_attributes(dataset, attrs)

    # add dimension labels, but should we also include dimension scales?
    dataset.dims[0].label = 'Atmospheric Level'
    dataset.dims[1].label = 'Y'
    dataset.dims[2].label = 'X'

    # process by tile
    for tile in generate_tiles(rds.width, rds.height, rds.width, ytile):
        idx = (
            slice(None),
            slice(tile[0][0], tile[0][1]),
            slice(tile[1][0], tile[1][1])
        )
        dataset[idx] = rds.read(bands, window=tile)

    # metadata
    metadata = metadata_dataframe(rds)
    write_dataframe(metadata, 'METADATA', fid, compression)


def _convert_4d(rds, fid, dataset_name, compression, filter_opts):
    """
    Private routine for converting the multiples of 37 layer
    atmospheric data in the GRIB file to HDF5.
    For a months worth of data, the dimensions become:
        * (day, atmospheric level, y, x)
    """
    attrs = {
        'geotransform': rds.transform.to_gdal(),
        'crs_wkt': rds.crs.wkt,
        'history': 'Converted to HDF5'
    }

    # band groups of 37, nrows to process (ytile)
    band_groups = range(1, rds.count + 1, 37)
    ytile = filter_opts['chunks'][2]
    dims = (len(band_groups), 37, rds.height, rds.width)
    tiles = generate_tiles(rds.width, rds.height, rds.width, ytile)

    # dataset creation options
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    kwargs['shape'] = dims
    kwargs['dtype'] = rds.dtypes[0]

    dataset = fid.create_dataset(dataset_name, **kwargs)
    attach_image_attributes(dataset, attrs)

    # add dimension labels, but should we also include dimension scales?
    dataset.dims[0].label = 'Day'
    dataset.dims[1].label = 'Atmospheric Level'
    dataset.dims[2].label = 'Y'
    dataset.dims[3].label = 'X'

    # process by spatial tile containing 37 atmospheric layers for 1 day
    for i, bg in enumerate(band_groups):
        bands = list(range(bg, bg + 37))
        for tile in tiles:
            idx = (
                slice(i, bg),
                slice(None),
                slice(tile[0][0], tile[0][1]),
                slice(tile[1][0], tile[1][1])
            )
            dataset[idx] = rds.read(bands, window=tile)

    # metadata
    metadata = metadata_dataframe(rds)
    write_dataframe(metadata, 'METADATA', fid, compression)


def convert(fname, base_outdir, compression=H5CompressionFilter.LZF,
            filter_opts=None):
    """
    Convert ECWMF GRIB files to HDF5.
    Each output file will contain 2 HDF5 Datasets:
        * METADATA (TABLE CLASS, containing metadata for each 2D slice)
        * {product_name} (IMAGE CLASS, could be 2D, 3D or 4D)
    """
    # create Path objects
    fname = Path(fname)
    base_outdir = Path(base_outdir)

    # 2D & 3D fname format "{product_name}_{yyyy-mm-dd}"
    product, date = fname.stem.split('_')

    # {base_dir}/{product_name}/{year}/{product_name}_{yyyy-mm-dd}.h5
    out_fname = base_outdir.joinpath(
        product,
        date.split('-')[0],
        fname.with_suffix('.h5').name
    )

    # create empty or copy the user supplied filter options
    if not filter_opts:
        filter_opts = dict()
    else:
        filter_opts = filter_opts.copy()

    # use sds native chunks if none are provided
    if 'chunks' not in filter_opts:
        filter_opts['chunks'] = (64, 64)

    # create directories as needed
    if not out_fname.parent.exists():
        out_fname.parent.mkdir(parents=True)

    with h5py.File(str(out_fname), 'w') as fid:
        with rasterio.open(fname) as rds:
            if rds.count == 1:
                # convert 2D
                _convert_2d(rds, fid, product, compression, filter_opts)
            elif rds.count == 37:
                # convert 3D (37 atmospheric layers)
                if len(filter_opts['chunks']) != 3:
                    chunks = list(filter_opts['chunks'])
                    chunks.insert(0, 37)
                    filter_opts['chunks'] = chunks
                _convert_3d(rds, fid, product, compression, filter_opts)
            elif not rds.count % 37:
                # convert 4D (eg months worth of 37 atmospheric layers)
                if len(filter_opts['chunks']) != 4:
                    chunks = list(filter_opts['chunks'])
                    chunks.insert(0, 37)
                    chunks.insert(0, 1)
                    filter_opts['chunks'] = chunks
                _convert_4d(rds, fid, product, compression, filter_opts)
            else:
                # don't have multiples of 37
                raise Exception("Number of bands is not a multiple of 37.")
