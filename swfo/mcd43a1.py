#!/usr/bin/env python

"""
Convert MCD43A1 HDF4 files to HDF5.
"""

from pathlib import Path
from subprocess import check_call
import tempfile
import rasterio
import netCDF4
import numpy
import h5py

from wagl.hdf5 import write_h5_image, attach_attributes, attach_image_attributes
from wagl.hdf5.compression import H5CompressionFilter
from wagl.tiling import generate_tiles
from wagl.constants import BrdfModelParameters


OUT_DTYPE = numpy.dtype(
    [
        (BrdfModelParameters.ISO.value, "int16"),
        (BrdfModelParameters.VOL.value, "int16"),
        (BrdfModelParameters.GEO.value, "int16"),
    ]
)

RASTERIO_PREFIX = "tar:{}!"
GDAL_PREFIX = "/vsitar/{}"


def _brdf_netcdf_band_orderer(band_name: str):
    """
    sort key to return band names in set ordering;
        * Albedo band parameters layers
        * Quality layers


    :param band_name:
        netcdf4 subdataset name
    """
    return band_name.split("_")[-2:]


def convert_tile(fname, out_h5: h5py.Group, compression, filter_opts):
    """
    Convert a MCD43A1 HDF4 tile into HDF5.
    Global and datasetl level metadata are copied across.

    :param fname:
        A str containing the MCD43A1 filename.

    :param out_h5:
        A h5py.Group to write the output data to

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
    # read the geo-spatial information beforehand
    # relying on gdal to parse it
    geospatial = {}
    with rasterio.open(fname) as ds:
        for sds_name in ds.subdatasets:
            with rasterio.open(sds_name) as sds:
                band_name = sds_name.split(":")[-1]
                geospatial[band_name] = {
                    "geotransform": sds.transform.to_gdal(),
                    "crs_wkt": sds.crs.wkt,
                }

    # convert data
    with netCDF4.Dataset(fname) as ds:
        ds.set_auto_scale(False)

        # global attributes
        global_attrs = {key: ds.getncattr(key) for key in ds.ncattrs()}
        attach_attributes(out_h5, global_attrs)

        # find and convert every subsdataset (sds)
        for sds_name in sorted(ds.variables, key=_brdf_netcdf_band_orderer):
            sds = ds.variables[sds_name]

            # create empty or copy the user supplied filter options
            if not filter_opts:
                f_opts = dict()
            else:
                f_opts = filter_opts.copy()

            # Recreate datasets as 2-dimensional dataset
            dim1, dim2, *_ = sds.shape
            if "chunks" not in f_opts:
                assert dim1 == 2400 and dim2 == 2400
                f_opts["chunks"] = (240, 240)
            else:
                f_opts["chunks"] = (f_opts[0], f_opts[1])

            # subdataset attributes and spatial attributes
            attrs = {key: sds.getncattr(key) for key in sds.ncattrs()}
            # attrs['geotransform'] = sds.transform.to_gdal()
            # attrs['crs_wkt'] = sds.crs.wkt
            attrs.update(geospatial[sds_name])

            in_arr = sds[:]
            if len(in_arr.shape) == 3:
                data = numpy.ndarray(shape=(dim1, dim2), dtype=OUT_DTYPE)
                for idx, band_name in enumerate(OUT_DTYPE.names):
                    data[band_name] = in_arr[:, :, idx]
            else:
                data = in_arr

            # write to disk as an IMAGE Class Dataset
            write_h5_image(
                data,
                sds_name,
                out_h5,
                attrs=attrs,
                compression=compression,
                filter_opts=f_opts,
            )


def buildvrt(indir, outdir):
    """
    Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.
    """
    indir = Path(indir)
    outdir = Path(outdir)

    # loop over each day directory
    for day in indir.iterdir():
        # expecting 20 subdatasets in each hdf4 file (hopefully the order gdal lists them in is consistent)  # noqa: E501 # pylint: disable=line-too-long
        subdataset_fnames = {i: [] for i in range(20)}

        # mosaic each MODIS tile for the current day directory
        for h4_fname in day.rglob("*.hdf"):
            with rasterio.open(str(h4_fname.absolute())) as h4_ds:

                # each subdataset will form a separate mosaic
                for i, sds_name in enumerate(h4_ds.subdatasets):
                    subdataset_fnames[i].append(sds_name)

        # loop over each subdataset and mosaic from all supporting MODIS tiles
        for _, file_list in subdataset_fnames.items():

            # temp file for the input file list
            with tempfile.NamedTemporaryFile("w") as tmpf:
                tmpf.writelines("\n".join(file_list))
                tmpf.flush()

                # mimic the 'day' directory partition
                base_name = Path(file_list[0].replace(":", "/")).name
                out_fname = outdir.joinpath(day.name, "{}.vrt".format(base_name))

                if not out_fname.parent.exists():
                    out_fname.parent.mkdir(parents=True)

                # buildvrt
                cmd = ["gdalbuildvrt", "-input_file_list", tmpf.name, str(out_fname)]

                check_call(cmd)


def convert_vrt(
    fname,
    out_h5: h5py.Group,
    dataset_name="dataset",
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    attrs=None,
):
    """
    Convert the VRT mosaic to HDF5.
    """
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
            filter_opts["chunks"] = chunks
        else:
            filter_opts = filter_opts.copy()

        if "chunks" not in filter_opts:
            filter_opts["chunks"] = chunks

        # modify to have 3D chunks if we have a multiband vrt
        if rds.count == 3 and len(filter_opts["chunks"]) != 3:
            # copy the users original 2D chunk and insert the third
            chunks = list(filter_opts["chunks"])
            chunks.insert(0, 3)
            filter_opts["chunks"] = chunks

        # dataset attributes
        if attrs:
            attrs = attrs.copy()
        else:
            attrs = {}

        attrs["geotransform"] = rds.transform.to_gdal()
        attrs["crs_wkt"] = rds.crs.wkt
        attrs["nodata"] = rds.nodata

        # dataset creation options
        kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
        kwargs["shape"] = dims
        kwargs["dtype"] = rds.dtypes[0]

        dataset = out_h5.create_dataset(dataset_name, **kwargs)
        attach_image_attributes(dataset, attrs)

        # tiled processing (all cols by chunked rows)
        ytile = filter_opts["chunks"][1] if rds.count == 3 else filter_opts["chunks"][0]
        tiles = generate_tiles(rds.width, rds.height, rds.width, ytile)

        for tile in tiles:
            # numpy index
            if rds.count == 3:
                idx = (
                    slice(None),
                    slice(tile[0][0], tile[0][1]),
                    slice(tile[1][0], tile[1][1]),
                )
            else:
                idx = (slice(tile[0][0], tile[0][1]), slice(tile[1][0], tile[1][1]))

            # ensure single band rds is read as 2D not 3D
            data = rds.read(window=tile) if rds.count == 3 else rds.read(1, window=tile)

            # write
            dataset[idx] = data
