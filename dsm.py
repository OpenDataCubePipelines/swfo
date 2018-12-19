#!/usr/bin/env python

"""
Convert a generic single band image files to HDF5.
A quick (temporary) script for mosaicing the JAXA DSM for testing
purposes within wagl (which assumes a mosaic).
"""

from posixpath import join as ppjoin
from pathlib import Path
from subprocess import check_call
import tarfile
import tempfile
import rasterio
import h5py

from wagl.hdf5 import write_h5_image, attach_image_attributes
from wagl.hdf5.compression import H5CompressionFilter
from wagl.tiling import generate_tiles


RASTERIO_PREFIX = 'tar:{}!/{}'
GDAL_PREFIX = '/vsitar/{}'


def convert_file(fname, out_fname, group_name='/', dataset_name='dataset',
                 compression=H5CompressionFilter.LZF, filter_opts=None):
    """
    Convert generic single band image file to HDF5.
    Processes in a tiled fashion to minimise memory use.
    Will process all columns by n (default 256) rows at a time,
    where n can be specified via command line using:
    --filter-opts '{"chunks": (n, xsize)}'

    :param fname:
        A str containing the raster filename.

    :param out_fname:
        A str containing the output filename for the HDF5 file.

    :param dataset_name:
        A str containing the dataset name to use in the HDF5 file.

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
    # opening as `append` mode allows us to add additional datasets
    with h5py.File(out_fname) as fid:
        with rasterio.open(fname) as ds:

            # create empty or copy the user supplied filter options
            if not filter_opts:
                filter_opts = dict()
            else:
                filter_opts = filter_opts.copy()

            # use sds native chunks if none are provided
            if 'chunks' not in filter_opts:
                filter_opts['chunks'] = (256, 256)

            # read all cols for n rows (ytile), as the GA's DEM is BSQ interleaved
            ytile = filter_opts['chunks'][0]

            # dataset attributes
            attrs = {
                'description': ('1 second DSM derived from the SRTM; '
                                'Shuttle Radar Topography Mission'),
                'geotransform': ds.transform.to_gdal(),
                'crs_wkt': ds.crs.wkt
            }

            # dataset creation options
            kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
            kwargs['shape'] = (ds.height, ds.width)
            kwargs['dtype'] = ds.dtypes[0]

            dataset_name = ppjoin(group_name, dataset_name)
            dataset = fid.create_dataset(dataset_name, **kwargs)
            attach_image_attributes(dataset, attrs)

            # process each tile
            for tile in generate_tiles(ds.width, ds.height, ds.width, ytile):
                idx = (slice(tile[0][0], tile[0][1]), slice(tile[1][0], tile[1][1]))
                data = ds.read(1, window=tile)
                dataset[idx] = data



def jaxa_buildvrt(indir, outdir):
    """
    Mosaic all the JAXA DSM files via a VRT.
    """
    indir = Path(indir)
    outdir = Path(outdir)

    fnames = {
        'average_dsm_fnames': [],
        'median_dsm_fnames':  [],
        'average_msk_fnames':  [],
        'median_msk_fnames':  [],
        'average_stk_fnames':  [],
        'median_stk_fnames':  []
    }

    case = {
        'AVE_DSM': 'average_dsm_fnames',
        'MED_DSM': 'median_dsm_fnames',
        'AVE_MSK': 'average_msk_fnames',
        'MED_MSK': 'median_msk_fnames',
        'AVE_STK': 'average_stk_fnames',
        'MED_STK': 'median_stk_fnames'
    }

    # TODO specify no data values (src and dst) for vrt creation
    # no_data = {
    #     'AVE_DSM': -9999,
    #     'MED_DSM': -9999,
    #     'AVE_MSK': 'average_msk_fnames',
    #     'MED_MSK': 'median_msk_fnames',
    #     'AVE_STK': 'average_stk_fnames',
    #     'MED_STK': 'median_stk_fnames'
    # }

    for fname in indir.rglob('*.tar.gz'):
        with tarfile.open(str(fname), 'r') as targz:
            for member in targz.getmembers():
                if member.name.endswith('.tif'):
                    name = Path(member.name).name
                    key = case[name[name.find('_') +1:name.find('.')]]

                    pathname = GDAL_PREFIX.format(fname.absolute().joinpath(member.name))
                    fnames[key].append(pathname)

    for key, value in case.items():
        with tempfile.NamedTemporaryFile('w') as tmpf:
            tmpf.writelines("\n".join(fnames[value]))
            tmpf.flush()
            out_fname = outdir.joinpath('{}.vrt'.format(key))

            # buildvrt
            # TODO set src and dst no_data values
            cmd = [
                'gdalbuildvrt',
                '-input_file_list',
                tmpf.name,
                str(out_fname)
            ]

            check_call(cmd)


def jaxa_tile(fname, out_fname, compression=H5CompressionFilter.LZF,
              filter_opts=None):
    """
    Convert a JAXA DSM .tar.gz file into a HDF5 file.
    """
    with tarfile.open(fname) as targz:
        for member in targz.getmembers():
            # only process TIFF's
            if member.name.endswith('.tif'):

                # define HDF5 Dataset name
                name = Path(member.name)
                ds_name = name.parent.joinpath(name.stem)

                # rasterio tar filename format
                raster_fname = RASTERIO_PREFIX.format(fname, name)

                # convert
                convert_file(raster_fname, out_fname, dataset_name=ds_name,
                             compression=compression, filter_opts=filter_opts)
