#!/usr/bin/env python

"""
Convert MCD43A1 HDF4 files to HDF5.
"""

from pathlib import Path
from subprocess import check_call
import tempfile
import rasterio
import h5py

from wagl.hdf5 import write_h5_image, attach_attributes


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
