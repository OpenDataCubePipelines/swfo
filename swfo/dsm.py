#!/usr/bin/env python

"""
Convert a generic single band image files to HDF5.
A quick (temporary) script for mosaicing the JAXA DSM for testing
purposes within wagl (which assumes a mosaic).
"""

from pathlib import Path
from subprocess import check_call
import tarfile
import tempfile
import rasterio
import h5py

from wagl.hdf5 import attach_image_attributes
from wagl.hdf5.compression import H5CompressionFilter
from wagl.tiling import generate_tiles

from .h5utils import generate_fallback_uuid


RASTERIO_PREFIX = "tar:{}!/{}"
GDAL_PREFIX = "/vsitar/{}"

PRODUCT_HREF = "https://collections.dea.ga.gov.au/ga_c_c_dsm_1"


def convert_file(
    fname: Path,
    out_h5: h5py.Group,
    out_dataset_path: str = "SWFO-DSM",
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    attrs=None,
):
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

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :param attrs:
        A dict containing any attribute information to be attached
        to the HDF5 Dataset.

    :return:
        None. Content is written directly to disk.
    """
    with rasterio.open(str(fname), "r") as ds:
        # create empty or copy the user supplied filter options
        if not filter_opts:
            filter_opts = dict()
        else:
            filter_opts = filter_opts.copy()

        # use sds native chunks if none are provided
        if "chunks" not in filter_opts:
            filter_opts["chunks"] = (min(256, ds.height), min(256, ds.width))

        # read all cols for n rows (ytile), as the GA's DEM is BSQ interleaved
        ytile = filter_opts["chunks"][0]

        # dataset attributes
        if attrs:
            attrs = attrs.copy()
        else:
            attrs = {}
        attrs["geotransform"] = ds.transform.to_gdal()
        attrs["crs_wkt"] = ds.crs.wkt

        # dataset creation options
        kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
        kwargs["shape"] = (ds.height, ds.width)
        kwargs["dtype"] = ds.dtypes[0]

        dataset = out_h5.create_dataset(out_dataset_path, **kwargs)
        attach_image_attributes(dataset, attrs)

        # process each tile
        for tile in generate_tiles(ds.width, ds.height, ds.width, ytile):
            idx = (slice(tile[0][0], tile[0][1]), slice(tile[1][0], tile[1][1]))
            data = ds.read(1, window=tile)
            dataset[idx] = data

        assert ds.count == 1  # checksum call assumes single band image
        metadata = {
            "id": str(
                generate_fallback_uuid(
                    PRODUCT_HREF, path=str(fname.stem), checksum=ds.checksum(1)
                )
            )
        }

    return [metadata], [out_dataset_path]


def jaxa_buildvrt(indir, outdir):
    """
    Mosaic all the JAXA DSM files via a VRT.
    """
    indir = Path(indir)
    outdir = Path(outdir)

    fnames = {
        "average_dsm_fnames": [],
        "median_dsm_fnames": [],
        "average_msk_fnames": [],
        "median_msk_fnames": [],
        "average_stk_fnames": [],
        "median_stk_fnames": [],
    }

    case = {
        "AVE_DSM": "average_dsm_fnames",
        "MED_DSM": "median_dsm_fnames",
        "AVE_MSK": "average_msk_fnames",
        "MED_MSK": "median_msk_fnames",
        "AVE_STK": "average_stk_fnames",
        "MED_STK": "median_stk_fnames",
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

    for fname in indir.rglob("*.tar.gz"):
        with tarfile.open(str(fname), "r") as targz:
            for member in targz.getmembers():
                if member.name.endswith(".tif"):
                    name = Path(member.name).name
                    key = case[name[name.find("_") + 1 : name.find(".")]]

                    pathname = GDAL_PREFIX.format(
                        fname.absolute().joinpath(member.name)
                    )
                    fnames[key].append(pathname)

    for key, value in case.items():
        with tempfile.NamedTemporaryFile("w") as tmpf:
            tmpf.writelines("\n".join(fnames[value]))
            tmpf.flush()
            out_fname = outdir.joinpath("{}.vrt".format(key))

            # buildvrt
            # TODO set src and dst no_data values
            cmd = ["gdalbuildvrt", "-input_file_list", tmpf.name, str(out_fname)]

            check_call(cmd)


def jaxa_tile(
    fname: Path,
    out_h5: h5py.Group,
    out_dataset_prefix: str = "/",
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """
    Convert a JAXA DSM .tar.gz file into a HDF5 file.
    """
    metadata = []
    dataset_names = []
    with tarfile.open(str(fname), "r") as targz:
        for member in targz.getmembers():
            # only process TIFF's
            if member.name.endswith(".tif"):
                tz_name = Path(member.name)

                # define HDF5 Dataset name
                name = Path(out_dataset_prefix).joinpath(tz_name.parent, tz_name.stem)

                # rasterio tar filename format
                raster_fname = RASTERIO_PREFIX.format(fname, tz_name)

                # convert
                _md, _ds = convert_file(
                    raster_fname,
                    out_h5,
                    out_dataset_path=name,
                    compression=compression,
                    filter_opts=filter_opts,
                )
                metadata.extend(_md)
                dataset_names.extend(_ds)
    return metadata, dataset_names
