#!/usr/bin/env python

"""
Conversion utilities for GA's auxillary data.
"""

from pathlib import Path
import json
import click
import rasterio
import h5py

from wagl.hdf5.compression import H5CompressionFilter

import mcd43a1
import prwtr
import dsm


class JsonType(click.ParamType):

    name = 'json-dict'

    def convert(self, value, param, ctx):
        return json.loads(value)


class CompressionType(click.ParamType):

    name = 'compression-type'
    filters = [f.name for f in list(H5CompressionFilter)]

    def get_metavar(self, param):
        return '[{}]'.format('|'.join(self.filters))

    def get_missing_messge(self, param):
        return 'Choose from:\n\t{}'.format(',\n\t'.join(self.filters))

    def convert(self, value, param, ctx):
        return H5CompressionFilter[value]


@click.group(name='mcd43a1', help='Convert MCD43A1 files.')
def mcd43a1_cli():
    pass


@click.group(name='prwtr', help='Convert water vapour filess.')
def prwtr_cli():
    pass


@click.group(name='dsm', help='Convert DSM files.')
def dsm_cli():
    pass


@mcd43a1_cli.command(help='Convert MCD43A1 HDF4 tiles to HDF5 tiles.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--outdir", type=click.Path(dir_okay=True, file_okay=False),
              help="A writeable directory to contain the converted files.")
@click.option("--compression", type=CompressionType(), default="LZF")
@click.option("--filter-opts", type=JsonType(), default=None)
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


@mcd43a1_cli.command(help='Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--outdir", type=click.Path(dir_okay=True, file_okay=False),
              help="A writeable directory to contain the converted files.")
def mcd43a1_vrt(indir, outdir):
    """
    Build VRT mosaic of each for each MCD43A1 HDF4 subdataset.
    """
    mcd43a1.buildvrt(indir, outdir)


@prwtr_cli.command(help='Convert PR_WTR NetCDF files into HDF5 files.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--outdir", type=click.Path(dir_okay=True, file_okay=False),
              help="A writeable directory to contain the converted files.")
@click.option("--compression", type=CompressionType(), default="LZF")
@click.option("--filter-opts", type=JsonType(), default=None)
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


@dsm_cli.command(help="Convert GA's SRTM ENVI file into HDF5.")
@click.option("--fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A readable directory to containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@click.option("--compression", type=CompressionType(), default="LZF")
@click.option("--filter-opts", type=JsonType(), default=None)
def ga_dsm(fname, out_fname, compression, filter_opts):
    """
    Convert GA's SRTM ENVI file into HDF5.
    """
    # convert to a Path object
    out_fname = Path(out_fname)

    # create directories as needed
    if not out_fname.absolute().parent.exists():
        out_fname.parent.mkdir(parents=True)

    dsm.convert_file(fname, str(out_fname), 'SRTM', 'GA-DSM', compression,
                     filter_opts)


@dsm_cli.command(help='Mosaic all the JAXA DSM files via a VRT.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--outdir", type=click.Path(dir_okay=True, file_okay=False),
              help="A writeable directory to contain the converted files.")
def jaxa_vrt(indir, outdir):
    """
    Mosaic all the JAXA DSM files via a VRT.
    """
    dsm.jaxa_buildvrt(indir, outdir)


@dsm_cli.command(help='Convert JAXA VRT files to a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--out-fname", type=click.Path(dir_okay=False, file_okay=True),
              help="A writeable directory to contain the converted files.")
@click.option("--compression", type=CompressionType(), default="LZF")
@click.option("--filter-opts", type=JsonType(), default=None)
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


@dsm_cli.command(help='Convert a JAXA DSM .tar.gz file into a HDF5 file.')
@click.option("--indir", type=click.Path(dir_okay=True, file_okay=False),
              help="A readable directory to containing the original files.")
@click.option("--outdir", type=click.Path(dir_okay=True, file_okay=False),
              help="A writeable directory to contain the converted files.")
@click.option("--compression", type=CompressionType(), default="LZF")
@click.option("--filter-opts", type=JsonType(), default=None)
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


cli = click.CommandCollection(sources=[mcd43a1_cli, prwtr_cli, dsm_cli])


if __name__ == '__main__':
    cli()
