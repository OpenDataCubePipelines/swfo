#!/usr/bin/env python

"""
Utility for converting GA's monthly ozone TIFF's to HDF5.
"""

from pathlib import Path
import rasterio
import h5py

from wagl.hdf5 import write_h5_image

from .h5utils import (
    generate_fallback_uuid, generate_md5sum
)


PRODUCT_HREF = 'https://collections.dea.ga.gov.au/ga_c_c_ozone_1'


def _month_sort(ozone_path: Path):
    """
    Provides a chronological ordering of signifiers
    """
    months = [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    search = ozone_path.stem.lower()
    return (months.index(search), None) if search in months else (13, search)


def convert(indir, out_h5: h5py.Group, compression, filter_opts):
    """
    Convert GA's ozone TIFF's to HDF5.
    The TIFF's will be converted into HDF5 IMAGE Datasets,
    and contained within a single HDF5 file.
    """
    # convert to Path object
    indir = Path(indir)
    dataset_names = []
    metadata = []

    # create empty or copy the user supplied filter options
    if not filter_opts:
        filter_opts = dict()
    else:
        filter_opts = filter_opts.copy()

    for fname in sorted(indir.glob('*.tif'), key=_month_sort):
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
            write_h5_image(rds.read(1), dname, out_h5, compression, attrs,
                           filter_opts)

            # src checksum; used to help derive fallback uuid
            with fname.open('rb') as src:
                src_checksum = generate_md5sum(src).hexdigest()

            dataset_names.append(dname)
            metadata.append({
                'id': str(generate_fallback_uuid(
                    PRODUCT_HREF,
                    path=str(dname),
                    md5=src_checksum
                ))
            })

    return metadata, dataset_names
