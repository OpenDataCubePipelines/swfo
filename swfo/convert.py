#!/usr/bin/env python

"""
Conversion utilities for GA's auxillary data.
Create timeseries averages for the NOAA water vapour data.
"""

import sys
import json
from pathlib import Path
import click
import h5py
import yaml

import dateutil.parser

from eodatasets.prepare import (
    ncep_reanalysis_surface_pr_wtr as water_vapour,
    modis_usgs_mcd43a1 as modis_brdf
)

from wagl.hdf5.compression import H5CompressionFilter

from swfo import mcd43a1, prwtr, dsm, atsr2_aot, ozone, ecmwf
from swfo.h5utils import write_h5_md


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


class JsonType(click.ParamType):

    name = 'json-dict'

    def convert(self, value, param, ctx):
        return json.loads(value)


class CompressionType(click.ParamType):

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
    pass


@cli.group(name='mcd43a1', help='Convert MCD43A1 files.')
def mcd43a1_cli():
    pass


@cli.group(name='prwtr', help='Convert water vapour filess.')
def prwtr_cli():
    pass


@cli.group(name='dsm', help='Convert DSM files.')
def dsm_cli():
    pass


@cli.group(name='aot', help='Convert Aerosol Optical Thickness files.')
def aot_cli():
    pass


@cli.group(name='ozone', help='Convert Ozone files.')
def ozone_cli():
    pass


@cli.group(name='ecmwf', help='Convert ECMWF GRIB files.')
def ecmwf_cli():
    pass


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

        # create directories as needed
        if not out_fname.parent.exists():
            out_fname.parent.mkdir(parents=True)

        mcd43a1.convert_tile(str(fname), str(out_fname), compression,
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

    # create directories as needed
    if not out_fname.parent.exists():
        out_fname.parent.mkdir(parents=True)

    md = modis_brdf.process_datasets(infile, md_file)[0]

    mcd43a1.convert_tile(str(infile), str(out_fname), compression,
                         filter_opts)

    md['format'] = {'name': 'HDF5'}

    # rewrite file paths:
    for band in md['image']['bands']:
        md['image']['bands'][band]['layer'] = '//{}'.format(
            md['image']['bands'][band]['layer'].split(':')[-1]
        )
        md['image']['bands'][band]['path'] = ''

    with h5py.File(str(out_fname), 'a') as fid:
        write_h5_md(fid, md)


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
        # create directories as needed
        if not out_fname.absolute().parent.exists():
            out_fname.parent.mkdir(parents=True)

        attrs = {
            'description': 'MCD43A1 product, mosaiced and converted to HDF5.'
        }

        for vrt_fname in day.rglob('*.vrt'):
            mcd43a1.convert_vrt(str(vrt_fname), out_fname,
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

        # create directories as needed
        if not out_fname.parent.exists():
            out_fname.parent.mkdir(parents=True)

        prwtr.convert_file(str(fname), str(out_fname), compression,
                           filter_opts)


@prwtr_cli.command('h5-md', help='Convert PR_WTR NetCDF files into HDF5 files with metadata entries')
@click.option('--fname', type=click.Path(dir_okay=False, file_okay=True),
              help='Path to hdf4 modis brdf tile')
@click.option('--outdir', type=click.Path(dir_okay=True, file_okay=False, writable=True),
              help='directory to output hdf5 file')
@_compression_options
def pr_wtr_md_cmd(fname, outdir, compression, filter_opts):
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

    prwtr.convert_file(str(fname), str(out_fname), compression,
                       filter_opts)

    dataset_names = []
    for _md in md:
        _md['format'] = {'name': 'HDF5'}
        for band in _md['image']['bands']:
            layer_name = (
                dateutil.parser.parse(_md['extent']['center_dt'])
                .strftime('//%Y/%B-%d/%H%M')
                .upper()
            )
            _md['image']['bands'][band]['layer'] = layer_name
            _md['image']['bands'][band]['path'] = ''
            del _md['image']['bands'][band]['band']
            dataset_names.append(layer_name)

    with h5py.File(str(out_fname), 'a') as fid:
        write_h5_md(fid, md, dataset_names)


@prwtr_cli.command('fallback', help='Create a PR_WTR vallback dataset based on averages.')
@_io_dir_options
@_compression_options
def pr_wtr_fallback(indir, outdir, compression, filter_opts):
    """
    Create a PR_WTR fallback dataset based on averages.
    """
    prwtr.fallback(indir, outdir, compression, filter_opts)


@dsm_cli.command('h5', help="Convert GA's SRTM ENVI file into HDF5.")
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
    out_fname = Path(out_fname)

    # create directories as needed
    if not out_fname.absolute().parent.exists():
        out_fname.parent.mkdir(parents=True)

    attrs = {
        'description': ('1 second DSM derived from the SRTM; '
                        'Shuttle Radar Topography Mission')
    }
    dsm.convert_file(fname, str(out_fname), 'SRTM', 'GA-DSM', compression,
                     filter_opts, attrs)


@dsm_cli.command('vrt', help='Mosaic all the JAXA DSM files via a VRT.')
@_io_dir_options
def jaxa_vrt(indir, outdir):
    """
    Mosaic all the JAXA DSM files via a VRT.
    """
    dsm.jaxa_buildvrt(indir, outdir)


@dsm_cli.command('vrt-to-h5', help='Convert JAXA VRT files to a HDF5 file.')
@_io_dir_options
@_compression_options
def jaxa_h5(indir, out_fname, compression, filter_opts):
    """
    Convert JAXA VRT files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # create directories as needed
    if not out_fname.absolute().parent.exists():
        out_fname.parent.mkdir(parents=True)

    # find vrt files
    for vrt_fname in indir.rglob('*.vrt'):
        dsm.convert_file(str(vrt_fname), str(out_fname), 'JAXA-ALOS',
                         vrt_fname.stem, compression, filter_opts)


@dsm_cli.command('h5', help='Convert a JAXA DSM .tar.gz file into a HDF5 file.')
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

        # create directories as needed
        if not out_fname.parent.exists():
            out_fname.parent.mkdir(parents=True)

        dsm.jaxa_tile(str(fname), str(out_fname), compression, filter_opts)


@aot_cli.command('h5', help='Converts .pix & .cmp files to a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@_compression_options
def atsr2_files(indir, out_fname, compression, filter_opts):
    """
    Converts .pix & .cmp files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # create directories as needed
    if not out_fname.absolute().parent.exists():
        out_fname.parent.mkdir(parents=True)

    # convert the data
    atsr2_aot.convert(indir, out_fname, compression, filter_opts)


@ozone_cli.command('h5', help='Converts ozone .tif files to a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@_compression_options
def ozone_files(indir, out_fname, compression, filter_opts):
    """
    Converts ozone .tif files to a HDF5 file.
    """
    # convert to Path objects
    indir = Path(indir)
    out_fname = Path(out_fname)

    # create directories as needed
    if not out_fname.absolute().parent.exists():
        out_fname.parent.mkdir(parents=True)

    # convert the data
    ozone.convert(indir, out_fname, compression, filter_opts)


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
