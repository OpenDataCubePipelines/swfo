#!/usr/bin/env python

"""
Convert PR_WTR NetCDF files to HDF5.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
from typing import Dict, List, Tuple, Optional
import urllib.parse

import numpy
import osr
import rasterio
import pandas
import h5py

from eodatasets.prepare.utils import ItemProvider
from wagl.geobox import GriddedGeoBox
from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import read_h5_table, write_h5_image, write_dataframe
from wagl.hdf5 import attach_attributes

from . import h5utils


CRS = osr.SpatialReference()
CRS.ImportFromEPSG(4326)
UUID_NAMESPACE = uuid.UUID('48682821-4061-4635-83aa-6a6ee8e10ceb')
PRODUCT_HREF = 'https://collections.dea.ga.gov.au/ga_c_c_prwtrfallback_1'


def convert_file(fname, out_h5: h5py.Group, compression,
                 filter_opts: Optional[Dict] = None):
    """
    Convert a PR_WTR NetCDF file into HDF5.

    :param fname:
        A str containing the PR_WTR filename.

    :param out_fname:
        A h5py.Group to write output datasets to

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
    with rasterio.open(fname) as ds:
        name_fmt = 'BAND-{}'

        # global attributes
        # TODO update the history attrs
        # TODO remove the NC_GLOBAL str and just have plain attr names
        g_attrs = ds.tags()

        # get timestamp info
        origin = g_attrs.pop('time#units').replace('hours since ', '')
        hours = json.loads(
            g_attrs.pop('NETCDF_DIM_time_VALUES').replace('{', '[').replace('}', ']')
        )
        df = pandas.DataFrame(
            {
                'timestamp': pandas.to_datetime(hours, unit='h', origin=origin),
                'band_name': [name_fmt.format(i+1) for i in range(ds.count)]
            }
        )
        df['dataset_name'] = df.timestamp.dt.strftime('%Y/%B-%d/%H%M')
        df['dataset_name'] = df['dataset_name'].str.upper()

        # create a timestamp and band name index table dataset
        desc = "Timestamp and Band Name index information."
        attrs = {
            'description': desc
        }
        write_dataframe(df, 'INDEX', out_h5, compression, attrs=attrs)

        attach_attributes(out_h5, g_attrs)

        # process every band
        for i in range(1, ds.count + 1):
            ds_name = df.iloc[i-1].dataset_name

            # create empty or copy the user supplied filter options
            if not filter_opts:
                f_opts = dict()
            else:
                f_opts = filter_opts.copy()

            # band attributes
            # TODO remove NETCDF tags
            # TODO add fillvalue attr
            attrs = ds.tags(i)
            attrs['timestamp'] = df.iloc[i-1]['timestamp'].replace(tzinfo=timezone.utc)
            attrs['band_name'] = df.iloc[i-1]['band_name']
            attrs['geotransform'] = ds.transform.to_gdal()
            attrs['crs_wkt'] = CRS.ExportToWkt()

            # use ds native chunks if none are provided
            if 'chunks' not in f_opts:
                try:
                    f_opts['chunks'] = ds.block_shapes[i]
                except IndexError:
                    print("Chunk error: {}".format(fname))
                    f_opts['chunks'] = (73, 144)

            # write to disk as an IMAGE Class Dataset
            write_h5_image(ds.read(i), ds_name, out_h5, attrs=attrs,
                           compression=compression, filter_opts=f_opts)


def _build_index(indir):
    """
    Read the INDEX table for each file and build a full history
    index.
    The records are sorted in ascending time (earliest to most recent)
    """
    df = pandas.DataFrame(columns=['filename', 'band_name', 'timestamp'])
    for fname in Path(indir).glob("pr_wtr.eatm.[0-9]*.h5"):
        with h5py.File(str(fname), 'r') as fid:
            tmp_df = read_h5_table(fid, 'INDEX')
            tmp_df['filename'] = fid.filename
            df = df.append(tmp_df, sort=False)

    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    return df


def _average(dataframe):
    """
    Given a dataframe with the columns:
        * filename
        * band_name

    Calculate the 3D/timeseries average from all input records.
    Each 2D dataset has dimensions (73y, 144x), and type float32.
    """
    dims = (dataframe.shape[0], 73, 144)
    data = numpy.zeros(dims, dtype="float32")

    # load all data into 3D array (dims are small so just read all)
    for i, rec in enumerate(dataframe.iterrows()):
        row = rec[1]
        with h5py.File(row.filename, "r") as fid:
            ds = fid[row.dataset_name]
            ds.read_direct(data[i])
            no_data = float(ds.attrs['missing_value'])

        # check for nodata and convert to nan
        # do this for each dataset in case the nodata value changes
        data[i][data[i] == no_data] = numpy.nan

    # get the geobox, chunks
    with h5py.File(row.filename, "r") as fid:
        ds = fid[row.dataset_name]
        geobox = GriddedGeoBox.from_dataset(ds)
        chunks = ds.chunks

    mean = numpy.nanmean(data, axis=0)

    return mean, geobox, chunks


def generate_fallback_metadata(
        dataset_path: str,
        lineage_ids: List[str],
        obs_dt: datetime,
        start_dt: datetime,
        end_dt: datetime,
        creation_dt: datetime,
        src_md: Dict) -> Dict:
    """
    Generates metadata associated with a precipitable water fallback
    dataset.

    :param dataset_path:
        A path offset in the output h5py.Group

    :param lineage_ids:
        A dictionary of uuids of source datasets

    :param obs_dt:
        Observation time associated for indexing

    :param start_dt:
        Start date for a date range for the fallback dataset

    :param end_dt:
        End date for a date range for the fallback dataset

    :param src_md:
        A metadata document associated with the source datasets
        for transposing associated metadata

    :return:
        A dictionary with exepected metadata attributes
    """
    md = {}
    md['id'] = str(uuid.uuid5(
        UUID_NAMESPACE,
        PRODUCT_HREF + '&' + urllib.parse.urlencode({
            'noaa_c_c_prwtreatm_1': lineage_ids
        })
    ))
    md['product'] = {
        'href': PRODUCT_HREF
    }
    md['crs'] = src_md['crs']
    md['datetime'] = obs_dt.isoformat()
    md['geometry'] = src_md['geometry']
    md['grids'] = src_md['grids']
    md['lineage'] = {
        'noaa_c_c_prwtreatm_1': lineage_ids
    }
    md['measurements'] = {
        'water_vapour': {
            'path': '',
            'layer': dataset_path,
        }
    }
    md['properties'] = {
        'dtr:start_datetime': start_dt.isoformat(),
        'dtr:end_datetime': end_dt.isoformat(),
        'item:providers': {
            'name': 'Geoscience Australia',
            'roles': [
                ItemProvider.PRODUCER.value,
                ItemProvider.HOST.value
            ]
        },
        'odc:creation_datetime': creation_dt.isoformat(),
        'odc:file_format': 'HDF5'
    }

    return md


def _get_lineage_md(group_df) -> Tuple[List[str], Dict]:
    """
    returns a list of uuids sampled for the creation
    of the fallback water vapour dataset and a sample
    source metadata doc to propagate fields from

    :param group_df:
        Dataframe subset representing the source files

    :return:
        A tuple containing a list of uuids and a sample metadata doc

    """
    uuids = []
    for _, row in group_df.iterrows():
        with h5py.File(row[0]) as fid:
            doc = h5utils.YAML.load(
                fid['{}/{}/{}'.format(
                    h5utils.PUBLIC_NAMESPACE,
                    row[2],
                    h5utils.METADATA_PTR
                )][()].item())
            uuids.append(doc['id'])

    return uuids, doc


def fallback(indir, outdir, compression=H5CompressionFilter.LZF,
             filter_opts: Optional[Dict] = None):
    """
    Take the 4 hourly daily average from all files.
    """
    df = _build_index(indir)

    # grouping
    groups = df.groupby([df.index.month, df.index.day, df.index.hour])

    # create directories as needed
    out_fname = Path(outdir).joinpath("pr_wtr.eatm.average.h5")
    out_fname.parent.mkdir(exist_ok=True, parents=True)

    # Set one creation datetime for all datasets
    creation_dt = (
        datetime.utcnow()
        .replace(tzinfo=timezone.utc).isoformat(),
    )

    # create output file
    with h5utils.atomic_h5_write(out_fname, 'w', track_order=True) as fid:

        # the data is ordered so we can safely use BAND-1 = Jan-1
        for band_index, item in enumerate(groups):
            grp_name, grp_df = item

            # synthesised leap year timestamp (use year 2000)
            fmt = "2000 {:02d} {:02d} {:02d} +0000"
            dtime = datetime.strptime(fmt.format(*grp_name), "%Y %m %d %H %z")

            # mean
            mean, geobox, chunks = _average(grp_df)

            # dataset name format "%B-%d/%H%M" eg FEBRUARY-06/1800 for Feb 6th 1800 hrs
            dname = "AVERAGE/{}".format(dtime.strftime("%B-%d/%H%M").upper())

            # dataset description
            description = ("Average data for {year_month} {hour} hours, "
                           "over the time period {dt_min} to {dt_max}")
            description = description.format(
                year_month=dtime.strftime("%B-%d"),
                hour=dtime.strftime("%H%M"),
                dt_min=grp_df.index.min().date(),
                dt_max=grp_df.index.max().date()
            )

            # dataset attributes
            attrs = {
                "description": description,
                "timestamp": dtime,
                "date_format": "2000 %B-%d/%H%M",
                "band_name": "BAND-{}".format(band_index + 1),
                "geotransform": geobox.transform.to_gdal(),
                "crs_wkt": geobox.crs.ExportToWkt()
            }

            # create empty or copy the user supplied filter options
            if not filter_opts:
                f_opts = dict()
            else:
                f_opts = filter_opts.copy()

            # use original chunks if none are provided
            if 'chunks' not in f_opts:
                f_opts['chunks'] = chunks

            # write
            h5utils.create_groups(fid, dname.rsplit('/', 1)[0], track_order=True)
            write_h5_image(mean, dname, fid, attrs=attrs,
                           compression=compression, filter_opts=f_opts)
            # Generate metadata
            lineage_ids, src_md = _get_lineage_md(grp_df)
            md = generate_fallback_metadata(
                dname,
                lineage_ids=lineage_ids,
                obs_dt=dtime,
                start_dt=grp_df.index.min().replace(tzinfo=timezone.utc),
                end_dt=grp_df.index.max().replace(tzinfo=timezone.utc),
                creation_dt=creation_dt,
                src_md=src_md
            )
            h5utils.write_h5_md(fid, [md], [dname], track_order=True)
