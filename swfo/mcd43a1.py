#!/usr/bin/env python

"""
Convert MCD43A1 HDF4 files to HDF5.
"""

from pathlib import Path
from subprocess import check_call
import tempfile
import rasterio
import h5py

from wagl.hdf5 import write_h5_image, attach_attributes, attach_image_attributes
from wagl.hdf5.compression import H5CompressionFilter
from wagl.tiling import generate_tiles


RASTERIO_PREFIX = 'tar:{}!'
GDAL_PREFIX = '/vsitar/{}'


def convert_tile(fname, out_fname, compression, filter_opts):
    """
    Convert a MCD43A1 HDF4 tile into HDF5.
    Global and datasetl level metadata are copied across.

    :param fname:
        A str containing the MCD43A1 filename.

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
            # global attributes
            attach_attributes(fid, ds.tags())

            # find and convert every subsdataset (sds)
            for sds_name in ds.subdatasets:
                with rasterio.open(sds_name) as sds:
                    ds_name = Path(sds_name.replace(':', '/')).name

                    # create empty or copy the user supplied filter options
                    if not filter_opts:
                        f_opts = dict()
                    else:
                        f_opts = filter_opts.copy()

                    # use sds native chunks if none are provided
                    if 'chunks' not in f_opts:
                        f_opts['chunks'] = list(sds.block_shapes[0])

                    # modify to have 3D chunks if we have a multiband sds
                    if sds.count == 3:
                        # something could go wrong if a user supplies
                        # a 3D chunk eg (2, 256, 340)
                        f_opts['chunks'].insert(0, 1)
                        f_opts['chunks'] = tuple(f_opts['chunks'])
                    else:
                        f_opts['chunks'] = tuple(f_opts['chunks'])

                    # subdataset attributes and spatial attributes
                    attrs = sds.tags()
                    attrs['geotransform'] = sds.transform.to_gdal()
                    attrs['crs_wkt'] = sds.crs.wkt

                    # ensure single band sds is read a 2D not 3D
                    data = sds.read() if sds.count == 3 else sds.read(1)

                    # write to disk as an IMAGE Class Dataset
                    write_h5_image(data, ds_name, fid, attrs=attrs,
                                   compression=compression,
                                   filter_opts=f_opts)


def buildvrt(indir, outdir):
    """
    Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.
    """
    indir = Path(indir)
    outdir = Path(outdir)

    # loop over each day directory
    for day in indir.iterdir():
        # expecting 20 subdatasets in each hdf4 file (hopefully the order gdal lists them in is consistent)
        subdataset_fnames = {i: [] for i in range(20)}

        # mosaic each MODIS tile for the current day directory
        for h4_fname in day.rglob('*.hdf'):
            with rasterio.open(str(h4_fname.absolute())) as h4_ds:

                # each subdataset will form a separate mosaic
                for i, sds_name in enumerate(h4_ds.subdatasets):
                    subdataset_fnames[i].append(sds_name)

        # loop over each subdataset and mosaic from all supporting MODIS tiles
        for _, file_list in subdataset_fnames.items():

            # temp file for the input file list
            with tempfile.NamedTemporaryFile('w') as tmpf:
                tmpf.writelines("\n".join(file_list))
                tmpf.flush()

                # mimic the 'day' directory partition
                base_name = Path(file_list[0].replace(':', '/')).name
                out_fname = outdir.joinpath(day.name, '{}.vrt'.format(base_name))

                if not out_fname.parent.exists():
                    out_fname.parent.mkdir(parents=True)

                # buildvrt
                cmd = [
                    'gdalbuildvrt',
                    '-input_file_list',
                    tmpf.name,
                    str(out_fname)
                ]

                check_call(cmd)


def convert_vrt(fname, out_fname, dataset_name='dataset',
                compression=H5CompressionFilter.LZF, filter_opts=None,
                attrs=None):
    """
    Convert the VRT mosaic to HDF5.
    The HDF5 file specified by out_fname will be opened
    in append mode.
    """
    with h5py.File(out_fname) as fid:
        with rasterio.open(fname) as rds:
            # set default chunks and set dimensions
            if rds.count == 3:
                chunks = (3, 256, 256)
                dims = (3, rds.height, rds.width)
            else:
                chunks = (256, 256)
                dims = (rds.height, rds.width)

            # create empty or copy the user supplied filter options
            if not filter_opts:
                filter_opts = dict()
                filter_opts['chunks'] = chunks
            else:
                filter_opts = filter_opts.copy()


            if 'chunks' not in filter_opts:
                filter_opts['chunks'] = chunks

            # modify to have 3D chunks if we have a multiband vrt
            if rds.count == 3 and len(filter_opts['chunks']) != 3:
                # copy the users original 2D chunk and insert the third
                chunks = list(filter_opts['chunks'])
                chunks.insert(0, 3)
                filter_opts['chunks'] = chunks

            # dataset attributes
            if attrs:
                attrs = attrs.copy()
            else:
                attrs = {}

            attrs['geotransform'] = rds.transform.to_gdal()
            attrs['crs_wkt'] = rds.crs.wkt

            # dataset creation options
            kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
            kwargs['shape'] = dims
            kwargs['dtype'] = rds.dtypes[0]

            dataset = fid.create_dataset(dataset_name, **kwargs)
            attach_image_attributes(dataset, attrs)

            # tiled processing (all cols by chunked rows)
            ytile = filter_opts['chunks'][1] if rds.count == 3 else filter_opts['chunks'][0]
            tiles = generate_tiles(rds.width, rds.height, rds.width, ytile)

            for tile in tiles:
                # numpy index
                if rds.count == 3:
                    idx = (
                        slice(None),
                        slice(tile[0][0], tile[0][1]),
                        slice(tile[1][0], tile[1][1])
                    )
                else:
                    idx = (
                        slice(tile[0][0], tile[0][1]),
                        slice(tile[1][0], tile[1][1])
                    )

                # ensure single band rds is read as 2D not 3D
                data = rds.read(window=tile) if rds.count == 3 else rds.read(1, window=tile)

                # write
                dataset[idx] = data
