#!/usr/bin/env python

"""
- Conversion utilities for GA's auxillary data.

- Create timeseries averages for the NOAA water vapour data.
"""

import sys
import os
import json
from pathlib import Path
import fnmatch
from contextlib import contextmanager
import tempfile

import click
import h5py

import dateutil.parser

from eodatasets.prepare import (
    noaa_c_c_prwtreatm_1_prepare as water_vapour,
    nasa_c_m_mcd43a1_6_prepare as modis_brdf
)

from wagl.hdf5.compression import H5CompressionFilter

from . import mcd43a1, prwtr, dsm, atsr2_aot, ozone, ecmwf
from .h5utils import write_h5_md


VLEN_STRING = h5py.special_dtype(vlen=str)


def _compression_options(f):
    """
    Decorator to add default compression options
    """
    f = click.option(
        "--compression",
        type=CompressionType(),
        envvar="SWFO_H5_COMPRESSION",
        default="LZF"
        )(f)
    f = click.option(
        "--filter-opts",
        type=JsonType(),
        envvar="SWFO_H5_FILTER_OPTS",
        default=None
        )(f)
    return f


def _io_dir_options(f):
    """
    Decorator to specify input/output directories
    """
    f = click.option(
        "--indir",
        type=click.Path(dir_okay=True, file_okay=False),
        help="A readable directory to containing the original files."
        )(f)
    f = click.option(
        "--outdir",
        type=click.Path(dir_okay=True, file_okay=False, writable=True),
        help="A writeable directory to contain the converted files."
        )(f)
    return f


@contextmanager
def _atomic_h5_write(fname: Path, mode='r', **kwargs):
    """
    Creates a temporary h5 file location before writing out datasets
    """
    os_fid, tpath = tempfile.mkstemp(
        dir=fname.parent,
        prefix='.tmp',
        suffix='.h5')
    fp = Path(tpath)
    try:
        with h5py.File(tpath, mode=mode, **kwargs) as h5_ref:
            yield h5_ref
        fp.rename(fname)
        fp = None
    finally:
        os.close(os_fid)
        if fp and fp.exists():
            fp.unlink()


class JsonType(click.ParamType):
    """
    Custom JSON type for handling interpretation from JSON
    """

    name = 'json-dict'

    def convert(self, value, param, ctx):
        return json.loads(value)


class CompressionType(click.ParamType):
    """
    Click wrapper for configuring hdf5 compression settings
    """

    name = 'compression-type'
    filters = [f.name for f in list(H5CompressionFilter)]

    def get_metavar(self, param):
        return '[{}]'.format('|'.join(self.filters))

    def get_missing_message(self, param):
        return 'Choose from:\n\t{}'.format(',\n\t'.join(self.filters))

    def convert(self, value, param, ctx):
        return H5CompressionFilter[value]


@click.group()
def cli():
    """
    Command line interface parent group
    """


@cli.group(name='mcd43a1', help='Convert MCD43A1 files.')
def mcd43a1_cli():
    """
    MODIS MCD43A1 dataset command group
    """


@cli.group(name='prwtr', help='Convert Water Vapour files.')
def prwtr_cli():
    """
    NOAA NCEP/NCAR Reanalysis 1 precipital water command group
    """


@cli.group(name='srtm-dsm', help='Convert SRTM DSM files.')
def srtm_dsm_cli():
    pass


@cli.group(name='jaxa-dsm', help='Convert JAXA DSM files.')
def jaxa_dsm_cli():
    pass


@cli.group(name='aot', help='Convert Aerosol Optical Thickness files.')
def aot_cli():
    """
    Aerosol optical thickness command group
    """


@cli.group(name='ozone', help='Convert Ozone files.')
def ozone_cli():
    """
    Ozone command group
    """


@cli.group(name='ecmwf', help='Convert ECMWF GRIB files.')
def ecmwf_cli():
    """
    ECMWF command group
    """


@mcd43a1_cli.command('h5', help='Convert MCD43A1 HDF4 tiles to HDF5 tiles.')
@_io_dir_options
@_compression_options
def mcd43a1_tiles(indir, outdir, compression, filter_opts):
    """
    Convert MCD43A1 HDF4 tiles to HDF5 tiles.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # find every HDF4 tile and convert to HDF5
    for fname in indir.rglob('*.hdf'):
        # out_fname includes the immediate parent directory
        out_fname = outdir.joinpath(*fname.parts[-2:]).with_suffix('.h5')
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        with _atomic_h5_write(out_fname, 'w') as out_h5:
            mcd43a1.convert_tile(str(fname), out_h5, compression,
                                 filter_opts)


@mcd43a1_cli.command('h5-md', help='Convert MCD43A1 hdf4 tile to HDF5 tile with ODC metadata file.')
@click.option('--fname', type=click.Path(dir_okay=False, file_okay=True),
              help='Path to hdf4 modis brdf tile')
@click.option('--outdir', type=click.Path(dir_okay=True, file_okay=False, writable=True),
              help='directory to output hdf5 file')
@click.option('--md-file', type=click.Path(dir_okay=False, file_okay=True), default=None,
              help='Path to the hdf4 xml metadatafile to generate an ODC metadata file for')
@_compression_options
def mcd43a1_tile_with_md(fname, outdir, md_file, compression, filter_opts):
    """
    Convert MCD43A1 HDF4 tile to HDF5 tile with metadata.
    """
    infile = Path(fname)
    if md_file:
        md_file = Path(md_file)
    else:
        md_file = Path(fname + '.xml')
    outdir = Path(outdir)
    if not md_file.exists():
        click.echo("Unable to find metadata file at: %s" % str(md_file))
        sys.exit(1)

    out_fname = outdir.joinpath(*infile.parts[-1:]).with_suffix('.h5')
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    md = modis_brdf.process_datasets(infile, md_file)[0]

    md['properties']['odc:file_format'] = 'HDF5'

    # rewrite file paths:
    for band in md['measurements']:
        md['measurements'][band]['layer'] = '//{}'.format(
            md['measurements'][band]['layer'].split(':')[-1]
        )
        md['measurements'][band]['path'] = ''

    md['properties']['item:providers'].append({
        'name': 'GeoscienceAustralia',
        'roles': ['host'],
    })

    with _atomic_h5_write(out_fname, 'w') as out_h5:
        mcd43a1.convert_tile(str(infile), out_h5, compression,
                             filter_opts)
        write_h5_md(out_h5, md)


@mcd43a1_cli.command('vrt', help='Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.')
@_io_dir_options
def mcd43a1_vrt(indir, outdir):
    """
    Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.
    """
    mcd43a1.buildvrt(indir, outdir)


@mcd43a1_cli.command('vrt-to-h5', help='Convert MCD43A1 VRT files to a HDF5 file.')
@_io_dir_options
@_compression_options
def mcd43a1_h5(indir, outdir, compression, filter_opts):
    """
    Convert MCD43A1 VRT files to HDF5 files.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # find vrt files
    for day in indir.iterdir():

        out_fname = outdir.joinpath('MCD43A1_{}_.h5'.format(day.name))
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        attrs = {
            'description': 'MCD43A1 product, mosaiced and converted to HDF5.'
        }

        with _atomic_h5_write(out_fname, 'a') as out_h5:
            for vrt_fname in day.rglob('*.vrt'):

                # attributes for h5 (scale and offset has been hardcoded to the values
                # for MCD43A1 from (https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mcd43a1_v006)

                if fnmatch.fnmatch(str(vrt_fname), '*Quality*'):
                    attrs['scales'] = 1
                    attrs['offsets'] = 0
                else:
                    attrs['scales'] = 0.001
                    attrs['offsets'] = 0

                mcd43a1.convert_vrt(str(vrt_fname), out_h5,
                                    dataset_name=vrt_fname.stem,
                                    compression=compression,
                                    filter_opts=filter_opts, attrs=attrs)


@prwtr_cli.command('h5', help='Convert PR_WTR NetCDF files into HDF5 files.')
@_io_dir_options
@_compression_options
def pr_wtr_cmd(indir, outdir, compression, filter_opts):
    """
    Convert PR_WTR NetCDF files into HDF5 files.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # create empty or copy the user supplied filter options
    if not filter_opts:
        filter_opts = dict()
    else:
        filter_opts = filter_opts.copy()

    # find every pr_wtr.eatm.{year}.nc file
    for fname in indir.rglob('pr_wtr*.nc'):
        out_fname = outdir.joinpath(fname.name).with_suffix('.h5')
        out_fname.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_h5_write(out_fname, 'w') as out_h5:
            prwtr.convert_file(str(fname), out_h5, compression,
                               filter_opts)


@prwtr_cli.command('h5-md', help='Convert PR_WTR NetCDF files into HDF5 files with metadata entries')
@click.option('--fname', type=click.Path(dir_okay=False, file_okay=True),
              help='Path to precipital water NetCDF stack.')
@click.option('--outdir', type=click.Path(dir_okay=True, file_okay=False, writable=True),
              help='directory to output hdf5 file')
@_compression_options
def pr_wtr_md_cmd(fname, outdir, compression, filter_opts):
    """
    Convert PR_WTR NetCDF files into HDF5 files with metadata.
    """
    # create empty or copy the user supplied filter options
    if not filter_opts:
        filter_opts = dict()
    else:
        filter_opts = filter_opts.copy()

    outdir = Path(outdir)
    fname = Path(fname)
    out_fname = outdir.joinpath(fname.name).with_suffix('.h5')

    # create directories as needed
    if not out_fname.parent.exists():
        out_fname.parent.mkdir(parents=True)

    md = water_vapour.process_datasets(fname)

    dataset_names = []
    for _md in md:
        _md['properties']['odc:file_format'] = 'HDF5'
        for measurement in _md['measurements']:
            layer_name = (
                dateutil.parser.parse(_md['datetime'])
                .strftime('//%Y/%B-%d/%H%M')
                .upper()
            )
            _md['measurements'][measurement]['layer'] = layer_name
            _md['measurements'][measurement]['path'] = ''
            del _md['measurements'][measurement]['band']
            dataset_names.append(layer_name)

        _md['properties']['item:providers'].append({
            'name': 'GeoscienceAustralia',
            'roles': ['host'],
        })

    with _atomic_h5_write(out_fname, 'w') as out_h5:
        prwtr.convert_file(str(fname), out_h5, compression,
                           filter_opts)
        write_h5_md(out_h5, md, dataset_names)


@prwtr_cli.command('fallback', help='Create a PR_WTR fallback dataset based on averages.')
@_io_dir_options
@_compression_options
def pr_wtr_fallback(indir, outdir, compression, filter_opts):
    """
    Create a PR_WTR fallback dataset based on averages.
    """
    prwtr.fallback(indir, outdir, compression, filter_opts)


@srtm_dsm_cli.command('h5', help="Convert GA's SRTM ENVI file into HDF5.")
@click.option("--fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A readable srtm envi file.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@_compression_options
def ga_dsm(fname, out_fname, compression, filter_opts):
    """
    Convert GA's SRTM ENVI file into HDF5.
    """
    # convert to a Path object
    fname = Path(fname)
    out_fname = Path(out_fname)
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    attrs = {
        'description': ('1 second DSM derived from the SRTM; '
                        'Shuttle Radar Topography Mission')
    }
    with _atomic_h5_write(out_fname, 'w') as out_h5:
        dsm.convert_file(fname, out_h5, 'SRTM', 'GA-DSM', compression,
                         filter_opts, attrs)


@srtm_dsm_cli.command('h5-md', help="Convert GA's SRTM ENVI file into HDF5 with md.")
@click.option("--fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A readable srtm envi file.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@_compression_options
def ga_dsm_md(fname, out_fname, compression, filter_opts):
    """
    Convert GA's SRTM ENVI file into HDF5 with metadata.
    """
    # convert to a Path object
    fname = Path(fname)
    out_fname = Path(out_fname)
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    attrs = {
        'description': ('1 second DSM derived from the SRTM; '
                        'Shuttle Radar Topography Mission')
    }
    with _atomic_h5_write(out_fname, 'a') as out_h5:
        md, dataset_names = dsm.convert_file(
            fname, out_h5, 'SRTM', 'GA-DSM', compression,
            filter_opts, attrs
        )
        write_h5_md(out_h5, md, dataset_names)


@jaxa_dsm_cli.command('vrt', help='Mosaic all the JAXA DSM files via a VRT.')
@_io_dir_options
def jaxa_vrt(indir, outdir):
    """
    Mosaic all the JAXA DSM files via a VRT.
    """
    dsm.jaxa_buildvrt(indir, outdir)


@jaxa_dsm_cli.command('vrt-to-h5', help='Convert JAXA VRT files to a HDF5 file.')
@_io_dir_options
@_compression_options
def jaxa_h5(indir, out_fname, compression, filter_opts):
    """
    Convert JAXA VRT files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    # find vrt files
    with _atomic_h5_write(out_fname, 'a') as out_h5:
        for vrt_fname in indir.rglob('*.vrt'):
            dsm.convert_file(str(vrt_fname), out_h5, 'JAXA-ALOS',
                             vrt_fname.stem, compression, filter_opts)


@jaxa_dsm_cli.command('h5', help='Convert a JAXA DSM .tar.gz file into a HDF5 file.')
@_io_dir_options
@_compression_options
def jaxa_tiles(indir, outdir, compression, filter_opts):
    """
    Convert JAXA tar.gz files into a HDF5 files.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # find vrt files
    for fname in indir.rglob('*.tar.gz'):
        out_fname = outdir.joinpath(Path(fname.stem).with_suffix('.h5'))
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        with _atomic_h5_write(out_fname, 'a') as out_h5:
            dsm.jaxa_tile(fname, out_h5, compression, filter_opts)


@jaxa_dsm_cli.command('h5-md', help='Convert a JAXA DSM .tar.gz file into a HDF5 file with metadata.')
@_io_dir_options
@_compression_options
def jaxa_tiles(indir, outdir, compression, filter_opts):
    """
    Convert JAXA tar.gz files into a HDF5 files with metadata.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # find vrt files
    for fname in indir.rglob('*.tar.gz'):
        out_fname = outdir.joinpath(Path(fname.stem).with_suffix('.h5'))
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        with _atomic_h5_write(out_fname, 'a') as out_h5:
            md, dataset_names = dsm.jaxa_tile(
                fname, out_h5, compression, filter_opts)
            write_h5_md(out_h5, md, dataset_names)


@aot_cli.command('h5', help='Converts .pix & .cmp files to a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable file location to contain the converted collection.")
@_compression_options
def atsr2_files(indir, out_fname, compression, filter_opts):
    """
    Converts .pix & .cmp files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # Create parent directories if missing
    out_fname.parent.mkdir(exist_ok=True, parents=True)

    # convert the data
    with _atomic_h5_write(out_fname, 'w') as out_h5:
        atsr2_aot.convert(indir, out_h5, compression, filter_opts)


@aot_cli.command('h5-md', help='Converts .pix & .cmp files to a HDF5 file with metadata.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable file location to contain the converted collection.")
@_compression_options
def atsr2_files_md(indir, out_fname, compression, filter_opts):
    """
    Converts .pix & .cmp files to a HDF5 file with metadata.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # Create parent directories if missing
    out_fname.parent.mkdir(exist_ok=True, parents=True)

    with _atomic_h5_write(out_fname, 'w') as out_h5:
        md, dataset_names = atsr2_aot.convert(
            indir, out_h5, compression, filter_opts)
        write_h5_md(out_h5, md, dataset_names)


@ozone_cli.command('h5', help='Converts ozone .tif files to a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable file location to contain the converted files.")
@_compression_options
def ozone_files(indir, out_fname, compression, filter_opts):
    """
    Converts ozone .tif files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # create directories as needed
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    # convert the data
    with _atomic_h5_write(out_fname, 'w') as out_h5:
        ozone.convert(indir, out_h5, compression, filter_opts)


@ozone_cli.command('h5-md', help='Converts ozone .tif directory to a HDF5 collection with metadata.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable file location to contain the converted collection.")
@_compression_options
def ozone_files_md(indir, out_fname, compression, filter_opts):
    """
    Converts ozone .tif files to a HDF5 file with associated metadata.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # create directories as needed
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    with _atomic_h5_write(out_fname, 'w', track_order=True) as out_h5:
        md, dataset_names = ozone.convert(
            indir, out_h5, compression, filter_opts)
        write_h5_md(out_h5, md, dataset_names)


@ecmwf_cli.command('h5', help='Convert ECMWF GRIB files into a HDF5 files.')
@_io_dir_options
@_compression_options
def ecmwf_files(indir, outdir, compression, filter_opts):
    """
    Convert ECMWF GRIB files into a HDF5 files.
    """
    # convert to Path objects
    indir = Path(indir)
    outdir = Path(outdir)

    # find vrt files
    for fname in indir.rglob('*.grib'):
        ecmwf.convert(fname, outdir, compression, filter_opts)


if __name__ == '__main__':
    cli()
