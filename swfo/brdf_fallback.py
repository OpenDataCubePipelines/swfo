#!/usr/bin/env python

"""
Pre-MODIS BRDF average implementation as described in
MODIS_BRDF_MCD43A1_Processing_v2.docx and
BRDF_shape_parameters_and_indices.docx
"""

import os
from os.path import join as pjoin
from pathlib import Path

import tempfile
from datetime import datetime, timezone, date
from typing import Optional, Dict, Iterable, Set
import uuid
import urllib.parse
from multiprocessing import Pool as ProcessPool, Lock

import fnmatch
import h5py
import numpy as np
import click
from affine import Affine
import shapely.affinity
import shapely.geometry
import shapely.ops
from scipy import ndimage
import rasterio.features

from wagl.hdf5.compression import H5CompressionFilter
from wagl.hdf5 import attach_image_attributes
from wagl.tiling import generate_tiles
from wagl.constants import BrdfModelParameters

from . import brdf_shape
from .convert import _compression_options
from .h5utils import write_h5_md, YAML, atomic_h5_write

BAND_LIST = ["Band{}".format(band) for band in range(1, 8)]
FALLBACK_PRODUCT_HREF = "https://collections.dea.ga.gov.au/ga_c_m_brdfalbedo_2"
FALLBACK_NAMESPACE = uuid.UUID("5acf51a2-8129-4318-944e-9eaed6a56786")

BRDF_AVG_FILE_BAND_FMT = "MCD43A1.JLAV.061.{}.DOY.{:03}.{}.h5"
BRDF_AVG_FILE_FMT = "MCD43A1.JLAV.061.{}.DOY.{:03}.h5"
BRDF_MATCH_PATTERN = "*{}.DOY.{:03}*Band*.h5"

LOCKS = {}

NODATA = 32767

SCALE_FACTOR = 0.0001
INV_SCALE_FACTOR = 10000

# for clean data interim file
SCALE_FACTOR_2 = 0.001
INV_SCALE_FACTOR_2 = 1000

"""
Below we define the dtypes of 'structured' numpy arrays that store the following respectively:
'MAIN' - The temporal average of the BRDF parameters.
'QUALITY_COUNT' - The number of days included in the temporal average with some quality flag(s).
'SUPPORT' - The BRDF 'shape' parameters, which are used to filter out values outside a valid range.
'QUALITY' - Indicates whether a pixel has been masked (due to unfeasible BRDF shape parameters and other filters).
"""

DTYPE_MAIN = np.dtype(
    [
        (BrdfModelParameters.ISO.value, "int16"),
        (BrdfModelParameters.VOL.value, "int16"),
        (BrdfModelParameters.GEO.value, "int16"),
    ]
)
DTYPE_QUALITY_COUNT = np.dtype(
    [("Q0COUNT", "int16"), ("Q1COUNT", "int16"), ("NOTQ0Q1COUNT", "int16")]
)
DTYPE_SUPPORT = np.dtype([("AFX", "int16"), ("RMS", "int16")])
DTYPE_QUALITY = np.dtype([("MASK", "int16"), ("NUM", "int16")])

METADATA_OFFSET = "/METADATA/CURRENT"


def get_datetime(dt: Optional[datetime] = None) -> datetime:
    """
    Returns a datetime object (defaults to utcnow) with tzinfo set to utc

    :param dt:
        (Optional) a datetime object to add utcnow to; defaults to utcnow()
    :return:
        A 'datetime' type with tzinfo set
    """
    if not dt:
        dt = datetime.utcnow()
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def albedo_band_name(band):
    """
    :param band:
        A 'str' type: A MODIS band name.
    :return:
        A 'str' type: A subdataset (brdf parameter band name) for the specified band.
    """
    return "BRDF_Albedo_Parameters_{}".format(band)


def quality_band_name(band):
    """
    :param band:
        A 'str' type: A MODIS band name.
    :return:
        A 'str' type: A subdataset (brdf quality band) name for the specified band.
    """
    return "BRDF_Albedo_Band_Mandatory_Quality_{}".format(band)


def folder_datetime(folder: str) -> datetime:
    """
    :param folder:
        A 'str' type: A Folder name in format ('%Y'.%m.%d').
    :return:
        A 'datetime' object parsed from folder format.
    """
    return datetime.strptime(folder, "%Y.%m.%d")


def folder_doy(folder: str) -> int:
    """
    :param folder:
        A 'str' type: A Folder name in format ('%Y'.%m.%d').
    :return:
        A 'int' type: Day of the year that folder corresponds to.
    """
    return folder_datetime(folder).timetuple().tm_yday


def filter_h5_info(files_dict: dict, doy: int) -> dict:
    """
    :param files_dict:
        A nested dict containing dates and BRDF file paths, as returned by hdf5_files()
    :return filtered_files_dict:
        A filtered file dictionary containing dates and BRDF file paths for a specific day of year (doy).
    """
    filtered_files_dict = {
        k: v for (k, v) in files_dict.items() if folder_doy(k) == doy
    }
    return filtered_files_dict


def shape_of_window(window):
    """returns a shape from a window."""

    y_shape = window[0].stop - window[0].start
    x_shape = window[1].stop - window[1].start
    return (y_shape, x_shape)


def gauss_filter(filter_size):
    """
    A Guassian filter where the weights are a normal exponential but value 1 at centre.
    Filter_rad is half the total filter length. The value of the normal distribution at
    a distance filter_size defines the filter. If the value of the exponential at
    filter_rad is 0.01 then sig can be defined as in the function.
    It is a backwards way to get a filter that is long enough to do the job.
    In this case you define the filter_size and sig gets worked out. It is roughly where
    the area between -filter_size and +filter_size is 99.9% of the total area.
    It just makes sure you have a normal distribution that is basically zero outside
    the length filter_rad to filter_rad.
    :param filter_size:
        A 'int' type: A length of a filter
    :return:
        A numpy array: A filter kernels
    """
    sig = 0.329505 * filter_size

    return np.array(
        [
            np.exp(-0.5 * ((j - filter_size) / sig) ** 2)
            for j in range(2 * filter_size + 1)
        ]
    )


def hdf5_files(
    brdf_dir: Path,
    tile: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """
    A function to extract relevant MODIS BRDF acquisition details
    from the root folder where BRDF data are stored.
    :param brdf_dir:
        A 'str' type: path name to where hdf5 formatted BRDF data are stored
        The BRDF directories are assumed to be yyyy.mm.dd naming convention.
    :param tile:
        A 'str' type: The name of a MODIS tile.
    :param start_date:
        A datetime.date type: the date from where processing should begin from.
        Default = None, will not filter by start_date.
    :param end_date:
        A datetime.date type: the date from where processing should end.
        Default = None, will not filter by end_date.
    :return:
        A 'dict' type: a nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name.
    """
    h5_info = {}
    for item in os.listdir(brdf_dir):
        if start_date and folder_datetime(item).date() < start_date:
            continue
        elif end_date and folder_datetime(item).date() > end_date:
            continue

        files = os.listdir(pjoin(brdf_dir, item))
        try:
            filename = fnmatch.filter(files, "*.{}*.h5".format(tile))[0]
            h5_info[item] = pjoin(brdf_dir, item, filename)
        except IndexError:
            pass

    return h5_info


def read_brdf_quality_dataset(ds, window=None):
    """
    :param ds:
        A 'file object' type: hdf5 file object containing the BRDF quality data set.
    :param window:
        A 'slice object' type: contain the set of indices specified by
        range(start, stop, step). Default=None, results in reading whole
        data set.
    :return:
        A 'array' type: slice of data set specified by window if set or all
        the data set. The BRDF parameter are scale and offset factor
        corrected and filled with numpy 'nan' for no data values.
    """
    if window is None:
        window = slice(None)

    data = ds[window].astype("float32")

    nodata_mask = data == float(ds.attrs["_FillValue"])
    data[nodata_mask] = np.nan

    return data


def read_brdf_dataset(ds, param, window=None):
    """
    :param ds:
        A 'file object' type: hdf5 file object containing the BRDF data set
        as a numpy structured data.
    :param param:
        A 'str' type: name of brdf parameter
    :param window:
        A 'slice object' type: contain the set of indices specified by
        range(start, stop, step). Default=None, results in reading whole
        data set.
    :return:
        A 'array' type: slice of data set specified by window if set or all
        the data set. The BRDF parameter are scale and offset factor
        corrected and filled with numpy 'nan' for no data values.
    """
    if window is None:
        window = slice(None)

    data = ds[window][param].astype("float32")
    nodata_mask = data == float(ds.attrs["_FillValue"])
    data[nodata_mask] = np.nan

    scale_factor = ds.attrs["scale_factor"]
    add_offset = ds.attrs["add_offset"]

    return scale_factor * (data - add_offset)


def get_qualityband_count_window(
    h5_info: Dict,
    band_name: str,
    window: Iterable[Iterable[int]],
    include_quality_values: Iterable[int] = None,
    exclude_quality_values: Iterable[int] = None,
):
    """returns sum of good quality data count per pixels for a window."""

    error_msg = "Please provide include_quality_values OR exclude_quality_values."

    # Throw error when both include_quality_values and exclude_quality_values are None
    assert (include_quality_values is not None) or (
        exclude_quality_values is not None
    ), error_msg

    # Throw error when both include_quality_values and exclude_quality_values are not None
    assert (include_quality_values is None) or (
        exclude_quality_values is None
    ), error_msg

    def read_quality_data(filename: str):
        """
        This function maps the value from read_brdf_quality_dataset
        to get a good quality data count. A 'good quality' pixel is
        assumed to be where the quality bit index is 0 i.e.
        "Processed, good quality (full BRDF inversions)"
        """
        with h5py.File(filename, "r") as fid:
            # Define quality array
            qual_array = read_brdf_quality_dataset(fid[band_name], window)

            # Setup new array where quality values to be counted are 1.0
            if include_quality_values is not None:
                quality_count_array = np.isin(qual_array, include_quality_values)
            else:
                quality_count_array = ~np.isin(qual_array, exclude_quality_values)

            # Preserve nans from original array
            return np.where(np.isnan(qual_array), np.nan, quality_count_array * 1.0)

    first, *rest = list(h5_info)

    data_sum = read_quality_data(h5_info[first])

    for key in rest:
        new_data = read_quality_data(h5_info[key])
        # https://stackoverflow.com/questions/42209838/treat-nan-as-zero-in-numpy-array-summation-except-for-nan-in-all-arrays
        data_sum_nan = np.isnan(data_sum)
        new_data_nan = np.isnan(new_data)
        data_sum = np.where(
            data_sum_nan & new_data_nan,
            np.nan,
            np.where(data_sum_nan, 0.0, data_sum)
            + np.where(new_data_nan, 0.0, new_data),
        )

    return window, data_sum


def get_qualityband_count(
    h5_info: Dict,
    band_name: str,
    shape: Iterable[int],
    compute_chunks: Iterable[int],
    nprocs: int,
    include_quality_values: Iterable[int] = None,
    exclude_quality_values: Iterable[int] = None,
):
    """
    This function computes and returns the number of valid pixels
    in a time series stack.
    :param h5_info:
        A 'dict' type: A nested dict containing  dates and BRDF file path
        which can be accessed through a key param defined by a folder name.
    :param band_name:
        A 'str' type: A name of quality band.
    :return:
        A numpy array with total quality band counts across all the datasets
        in h5_info.
    """
    data_sum = np.zeros(shape=shape, dtype="int16")

    with Pool(processes=nprocs) as pool:
        results = pool.starmap(
            get_qualityband_count_window,
            [
                (
                    h5_info,
                    band_name,
                    window,
                    include_quality_values,
                    exclude_quality_values,
                )
                for window in generate_windows(shape, compute_chunks)
            ],
        )

    for window, data in results:
        data_sum[window] = data

    return data_sum


def calculate_combined_mask(afx, rms):
    """
    A function to generate mask based on BRDF shape indicies using unfeasible afx
    and rms values derived from respective min and max values
    The max and min for rms and afx are sourced from David's brdf document.
    :param afx:
        A 'numpy array' type: The Root Mean Square (RMS) statistics.
    :param afx:
        A 'numpy array' type: The Anisotropic Flat Index (AFX) statistics.
    :return:
        A 'numpy array' type: A mask on the array which are considered unfeasible
        based on conditions defined.
    """

    rms_min_mask = np.ma.masked_where(rms < brdf_shape.CONSTANTS["rmsmin"], rms).mask
    rms_max_mask = np.ma.masked_where(rms > brdf_shape.CONSTANTS["rmsmax"], rms).mask
    afx_min_mask = np.ma.masked_where(afx < brdf_shape.CONSTANTS["afxmin"], afx).mask
    afx_max_mask = np.ma.masked_where(afx > brdf_shape.CONSTANTS["afxmax"], afx).mask
    rms_mask = np.ma.mask_or(rms_min_mask, rms_max_mask, shrink=False)
    afx_mask = np.ma.mask_or(afx_min_mask, afx_max_mask, shrink=False)
    rms_afx_mask = np.ma.mask_or(rms_mask, afx_mask, shrink=False)

    return rms_afx_mask


def brdf_indices_quality_check(
    quality_count_q0, quality_count_q1, quality_count_notq0q1, avg_data=None
):
    """
    This function performs the quality check on the temporal averages data.
    The quality check is performed in following steps:
        1. data are masked if any of the iso, vol and geo data are not valid.
        2. data are further masked if data are greater or less than valid range
           of brdf shape indices rms and afx (max and min values are sourced
           from MODIS_BRDF_MCD43A_Processing_v2' by David Jupp 2018).
        3. Additional mask based on the requirement (b2 >= ac) is applied
           (parameters definition are based on BRDF_shape_parameters_and_indices'
           by David Jupp 2018.
    :param avg_data:
         A 'dict' type data set that contains the numpy array type dataset with
         temporal average of clean brdf parameters (iso, vol, and geo) and
         its associated standard deviation and number of observations used
         to generate the temporal average.

    :return filtered_data:
         A 'dict' type data that contains the filtered data of brdf
         shape function (alpha1 and alpha2) in lognormal space. Additional
         data, mean brdf iso parameter, number of observations used in
         generating the mean brdf iso parameters, shape indices (rms and afx),
         mask and number of observations used in generating shape function are
         also included.
    """
    filtered_data = {}
    for key in avg_data:
        # set the mean brdf data from the avg_data dict for each keys
        iso_mean = avg_data[key][BrdfModelParameters.ISO]["mean"]
        vol_mean = avg_data[key][BrdfModelParameters.VOL]["mean"]
        geo_mean = avg_data[key][BrdfModelParameters.GEO]["mean"]

        # generate new mask where all the iso, vol and geo brdf parameters are valid
        mask_param = np.ma.mask_or(
            np.ma.mask_or(iso_mean.mask, vol_mean.mask, shrink=False),
            geo_mean.mask,
            shrink=False,
        )

        min_num = np.min(
            np.array(
                [
                    avg_data[key][BrdfModelParameters.ISO]["num"],
                    avg_data[key][BrdfModelParameters.VOL]["num"],
                    avg_data[key][BrdfModelParameters.GEO]["num"],
                ]
            ),
            axis=0,
        )

        # mask the brdf param with new mask that is generated from union of masks from
        # individual brdf parameters (iso, vol, and geo)
        iso_mean = np.ma.masked_array(iso_mean, mask=mask_param)
        vol_mean = np.ma.masked_array(vol_mean, mask=mask_param)
        geo_mean = np.ma.masked_array(geo_mean, mask=mask_param)

        iso_std = np.ma.masked_array(
            avg_data[key][BrdfModelParameters.ISO]["std"], mask=mask_param
        )

        # set coefficients of variation
        cov_iso = iso_std / iso_mean

        # set alpha1 and alpha2 in lognormal space
        alpha1, alpha2 = brdf_shape.get_mean_shape_param(
            iso_mean, vol_mean, geo_mean, cov_iso
        )

        # set afx and rms indices
        afx = brdf_shape.get_afx_indices(alpha1, alpha2)
        rms = brdf_shape.get_rms_indices(alpha1, alpha2)

        combined_mask = calculate_combined_mask(afx, rms)

        temp = {}
        temp[BrdfModelParameters.ISO.value] = np.ma.masked_array(
            iso_mean, mask=combined_mask
        )
        temp[BrdfModelParameters.VOL.value] = np.ma.masked_array(
            alpha1 * iso_mean, mask=combined_mask
        )
        temp[BrdfModelParameters.GEO.value] = np.ma.masked_array(
            alpha2 * iso_mean, mask=combined_mask
        )

        # Define number of observations used for the parameter averages
        temp["Q0COUNT"] = quality_count_q0
        temp["Q1COUNT"] = quality_count_q1
        temp["NOTQ0Q1COUNT"] = quality_count_notq0q1

        temp["AFX"] = np.ma.masked_array(afx, mask=combined_mask)
        temp["RMS"] = np.ma.masked_array(rms, mask=combined_mask)
        temp["MASK"] = np.array(combined_mask)
        temp["NUM"] = np.array(min_num)
        filtered_data[key] = temp

    return filtered_data


def get_std_block(h5_info, band_name, param, window):
    """
    A function to compute a standard deviation for across a temporal axis.
    This function was written to facilitate parallel processing.
    """

    def __get_data(dat_filename, param, window):
        with h5py.File(dat_filename, "r") as fid:
            dat = read_brdf_dataset(fid[band_name], param, window)
            return dat

    data = np.zeros((len(h5_info),) + shape_of_window(window), dtype="float32")
    for layer, filename in enumerate(h5_info.values()):
        data[layer] = __get_data(filename, param, window)

    run_median = np.nanmedian(data, axis=0, keepdims=False)

    for item in range(data.shape[0]):
        idx = np.where(np.isnan(data[item]))
        data[item][idx] = run_median[idx]

    return np.nanstd(data, axis=0, ddof=1, keepdims=False)


def _to_lists(x):
    """
    Returns lists of lists when given tuples of tuples
    """
    if isinstance(x, tuple):
        return [_to_lists(el) for el in x]

    return x


def _calculate_valid_bounds(mask: np.ndarray, transform: Affine) -> Dict:
    """
    Implements same functionality as eodatasets.metadata.valid_region
    except mask and transform as supplied by user
    """
    mask = ndimage.binary_fill_holes(mask)
    shapes = rasterio.features.shapes(mask.astype("uint8"), mask=None)
    shape = shapely.ops.unary_union(
        [shapely.geometry.shape(shape) for shape, val in shapes if val == 1]
    )

    # convex hull
    geom = shape.convex_hull

    # buffer by 1 pixel
    geom = geom.buffer(1, join_style=3, cap_style=3)

    # simplify with 1 pixel radius
    geom = geom.simplify(1)

    # intersect with image bounding box
    geom = geom.intersection(shapely.geometry.box(0, 0, mask.shape[1], mask.shape[0]))

    # transform from pixel space into CRS space
    geom = shapely.affinity.affine_transform(
        geom,
        (
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.xoff,
            transform.yoff,
        ),
    )

    output = shapely.geometry.mapping(geom)
    output["coordinates"] = _to_lists(output["coordinates"])
    return output


def concatenate_files(
    infile_paths: Iterable[str],
    outfile: str,
    h5_info: Dict,
    doy: str,
    tile_metadata: Dict,
) -> None:
    """
    A function to concatenate multiple h5 files and append metadata information.
    """

    assert len(infile_paths) == 7

    # set the uuids used in processing average for given day of year
    tile_metadata["lineage"]["doy_average"] = [
        munge_metadata(h5_info[key]) for key in h5_info if folder_doy(key) == doy
    ]

    # Set deterministic UUID from lineage
    tile_metadata["id"] = str(
        uuid.uuid5(
            FALLBACK_NAMESPACE,
            FALLBACK_PRODUCT_HREF
            + "&"
            + urllib.parse.urlencode(tile_metadata["lineage"]),
        )
    )

    geom_mask = None
    transform = None

    with atomic_h5_write(Path(outfile), "w") as out_fid:
        # Sorting works since No. of bands < 10
        # note that 3 hdf5 datasets will show consecutively per band
        for fp in sorted(infile_paths):
            with h5py.File(fp, "r") as in_fid:
                for ds_band in in_fid:
                    if "BRDF_Albedo_Parameters_" in ds_band:
                        _band = in_fid[ds_band]
                        if not transform:
                            transform = Affine.from_gdal(*_band.attrs["geotransform"])
                        nodata_value = _band.attrs["_FillValue"]
                        albedo_params = _band[()]
                        if geom_mask is not None:
                            geom_mask |= np.logical_or(
                                *(
                                    (albedo_params[layer] != nodata_value)
                                    for layer in DTYPE_MAIN.names
                                )
                            )
                        else:
                            geom_mask = np.logical_or(
                                *(
                                    (albedo_params[layer] != nodata_value)
                                    for layer in DTYPE_MAIN.names
                                )
                            )
                    in_fid.copy(source=ds_band, dest=out_fid)
        # Calculate valid bounds from source ISO, VOL, GEO params
        tile_metadata["geometry"] = _calculate_valid_bounds(geom_mask, transform)
        # Write out metadata
        write_h5_md(out_fid, [tile_metadata], ["/"])


def generate_windows(
    shape: Iterable[int], compute_chunks: Iterable[int]
) -> Iterable[Iterable[int]]:
    """
    Generates a window of height and width equivalent to compute_chunk's shape.

    :param shape:
        Total size to iterate over

    :param compute_chunk:
        Size of shape subset at a time

    :return:
        a subset window of ((y, x), (y, x))
    """
    for x, y in generate_tiles(
        shape[0], shape[1], compute_chunks[0], compute_chunks[1]
    ):
        yield (slice(*y), slice(*x))


class DummyPool:
    def __enter__(self):
        return self

    def starmap(self, func, args):
        return [func(*arg) for arg in args]


def Pool(processes):
    if not processes:
        return DummyPool()

    return ProcessPool(processes=processes)


def calculate_thresholds(h5_info, band_name, shape, compute_chunks, nprocs=None):
    """
    Computes threshold needed needed to clean temporal series
    """
    thresh_dict = {}
    for param in BrdfModelParameters:
        with Pool(nprocs) as pool:
            results = pool.starmap(
                get_std_block,
                [
                    (h5_info, band_name, param.value, window)
                    for window in generate_windows(shape, compute_chunks)
                ],
            )

        thresh_dict[param] = np.nanmean(results)

    return thresh_dict


def create_dataset(
    group: h5py.Group,
    band_name: str,
    shape: Iterable[int],
    attrs: Dict,
    dtype=np.int16,
    chunks: Iterable[int] = (240, 240),
    compression: H5CompressionFilter = H5CompressionFilter.LZF,
    filter_opts: Optional[Dict] = None,
):
    """creates dataset and attaches attributes for h5 object."""

    if filter_opts is None:
        filter_opts = {}
    else:
        filter_opts = filter_opts.copy()

    if "chunks" not in filter_opts:
        filter_opts["chunks"] = chunks

    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    ds = group.create_dataset(band_name, shape=shape, dtype=dtype, **kwargs)

    attach_image_attributes(ds, attrs)

    return ds


def create_brdf_datasets(
    group: h5py.Group,
    band_name: str,
    shape: Iterable[int],
    common_attrs: Dict,
    chunks: Iterable[int] = (240, 240),
    compression: H5CompressionFilter = H5CompressionFilter.LZF,
    filter_opts: Optional[Dict] = None,
):
    """creates brdf dataset for brdf parameters."""

    attrs = dict(
        scale_factor=SCALE_FACTOR,
        add_offset=0,
        _FillValue=NODATA,
        description=(
            "BRDF albedo parameters (ISO, VOL and GEO)"
            " derived from {}"
            " in lognormal space".format(albedo_band_name(band_name))
        ),
        **common_attrs,
    )
    create_dataset(
        group,
        "BRDF_Albedo_Parameters_{}".format(band_name),
        shape,
        attrs,
        chunks=chunks,
        filter_opts=filter_opts,
        compression=compression,
        dtype=DTYPE_MAIN,
    )

    attrs = dict(
        scale_factor=SCALE_FACTOR,
        add_offset=0,
        _FillValue=NODATA,
        description=(
            "BRDF shape indices (AFX and RMS)"
            " generated to support future validation work"
        ),
        **common_attrs,
    )
    create_dataset(
        group,
        "BRDF_Albedo_Shape_Indices_{}".format(band_name),
        shape,
        attrs,
        chunks=chunks,
        filter_opts=filter_opts,
        compression=compression,
        dtype=DTYPE_SUPPORT,
    )

    attrs = dict(
        description=(
            "MASK and NUM(BER) of valid data used"
            " in generating BRDF Albedo shape parameters"
        ),
        **common_attrs,
    )
    create_dataset(
        group,
        "BRDF_Albedo_Shape_Parameters_Quality_{}".format(band_name),
        shape,
        attrs,
        chunks=chunks,
        filter_opts=filter_opts,
        compression=compression,
        dtype=DTYPE_QUALITY,
    )

    attrs = dict(
        description=(
            "Number of observations used" " in generating mean BRDF parameters"
        ),
        **common_attrs,
    )
    create_dataset(
        group,
        "BRDF_Albedo_Quality_Count_{}".format(band_name),
        shape,
        attrs,
        chunks=chunks,
        filter_opts=filter_opts,
        compression=compression,
        dtype=DTYPE_QUALITY_COUNT,
    )


def write_chunk(data_dict, fid, band, window):
    """
    write numpy array to to h5 files with user supplied attributes
    and compression.
    """
    assert len(data_dict) == 1

    key = list(data_dict.keys())[0]
    shape = shape_of_window(window)

    data_main = np.ndarray(shape, dtype=DTYPE_MAIN)
    for band_name in DTYPE_MAIN.names:
        data_main[band_name] = (
            np.rint(data_dict[key][band_name] * INV_SCALE_FACTOR)
            .filled(fill_value=NODATA)
            .astype("int16")
        )

    data_support = np.ndarray(shape, dtype=DTYPE_SUPPORT)
    for band_name in DTYPE_SUPPORT.names:
        data_support[band_name] = (
            np.rint(data_dict[key][band_name] * INV_SCALE_FACTOR)
            .filled(fill_value=NODATA)
            .astype("int16")
        )

    data_quality = np.ndarray(shape, dtype=DTYPE_QUALITY)
    for band_name in DTYPE_QUALITY.names:
        data_quality[band_name] = data_dict[key][band_name].astype("int16")

    data_count = np.ndarray(shape, dtype=DTYPE_QUALITY_COUNT)
    for band_name in DTYPE_QUALITY_COUNT.names:
        data_count[band_name] = data_dict[key][band_name].astype("int16")

    fid["BRDF_Albedo_Parameters_{}".format(band)][window] = data_main
    fid["BRDF_Albedo_Shape_Indices_{}".format(band)][window] = data_support
    fid["BRDF_Albedo_Shape_Parameters_Quality_{}".format(band)][window] = data_quality
    fid["BRDF_Albedo_Quality_Count_{}".format(band)][window] = data_count


def get_band_info(h5_info: Dict, band_name: str):
    """
    Returns the crs in wkt format and geotransform for one of the source files.
    """
    first_doy = next(iter(h5_info))
    with h5py.File(h5_info[first_doy], "r") as fid:
        ds = fid[band_name]
        return ds.shape, {key: ds.attrs[key] for key in ["crs_wkt", "geotransform"]}


def temporal_average(data, doy):
    """
    This function computes temporal average.

    returns the stats on the average using the mean, standard deviation and the number
    of good quality data used in deriving the stats

    In David document, Mean, Stdv, Num and Masks are returned for each temporal average,
    here, we did not output mask because it can be inferred from the number of good
    quality data.

    """
    tmp = {}
    for param in BrdfModelParameters:
        data_param = np.array(
            [data[param][key] for key in data[param].keys() if folder_doy(key) == doy]
        )
        data_param = np.ma.masked_invalid(data_param)
        tmp[param] = dict(
            mean=np.ma.mean(data_param, axis=0),
            std=np.ma.std(data_param, ddof=0, axis=0),
            num=data_param.count(axis=0),
        )
    return {doy: tmp}


def apply_threshold(
    clean_data_file, h5_info, band_name, window, filter_size, thresholds, bad_indices
):
    """
    This function applies median filter on the dataset, median filter with size of
    (2 * filter_size + 1) is applied as a running median filter with time steps centered
    the data value. For example, if time step included day of year [001, 002,...009] then
    median filer would consists of all dataset if filter_size is 4 and center time step
    would be day 005. For the edge cases, the filter size would only encompass the all
    available time step within the defined filter size. For example, if data were to be
    filtered for day 008 for [001, 002,...009] then the median filter would only encompass
    data from day [004, 005, 006, 007, 008 and 009] since day 010, 011, 012 are missing.

    """
    all_data_keys = sorted(list(h5_info.keys()))

    def get_albedo_data(filename, param, window):
        with h5py.File(filename, "r") as fid:
            return read_brdf_dataset(fid[band_name], param, window)

    # dictionary mapping date to data
    data_dict = {}

    for index, key in enumerate(all_data_keys):
        # set the index's for the median filters
        start_idx = index - filter_size
        end_idx = index + filter_size + 1

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(all_data_keys) - 1:
            end_idx = len(all_data_keys)

        # read the data relevant to this range
        for _date in all_data_keys[start_idx:end_idx]:
            if _date not in data_dict:
                data_dict[_date] = np.ma.masked_invalid(
                    np.array(
                        [
                            get_albedo_data(h5_info[_date], param.value, window)
                            for param in BrdfModelParameters
                        ]
                    )
                )

        # clean up the data we don't need anymore
        for _date in list(data_dict):
            if _date not in all_data_keys[start_idx:end_idx]:
                del data_dict[_date]

        data_all_params = np.ma.array(
            [data_dict[_date] for _date in all_data_keys[start_idx:end_idx]]
        )

        for param_index, param in enumerate(BrdfModelParameters):
            # get the data iso, vol or geo from data which is a dict for all the keys and convert to numpy array  # noqa: E501 # pylint: disable=line-too-long
            data_param = data_all_params[:, param_index, :, :]

            # extract the value for a key and mask invalid data
            clean_data = data_param[index - start_idx]

            # replace bad index data with nan
            clean_data[bad_indices] = np.nan

            clean_data = np.ma.masked_invalid(clean_data)

            # get temporal local median value as set by filter size
            local_median = np.ma.median(data_param, axis=0)

            # apply threshold test to clean the data set
            threshold_idx = np.ma.where(
                np.abs(local_median - clean_data) > thresholds[param]
            )

            # replace the data which did not pass threshold test with temporal local median value  # noqa: E501 # pylint: disable=line-too-long
            clean_data[threshold_idx] = local_median[threshold_idx]

            # convert to int16
            clean_data = clean_data * INV_SCALE_FACTOR_2
            clean_data = clean_data.filled(fill_value=NODATA).astype(np.int16)

            with LOCKS[clean_data_file]:
                with h5py.File(clean_data_file) as output:
                    output[key][(param_index,) + window] = clean_data


def apply_convolution(filename, h5_info, window, filter_size, mask_indices):
    """
    This function applies convolution on the clean dataset from applied threshold
    method.
    """
    all_data_keys = sorted(list(h5_info.keys()))

    # define a filter to be used in convolution and normalize to sum 1.0
    filt = gauss_filter(filter_size)
    filt = filt / np.sum(filt)

    def __get_clean_data(data_filename, data_key, data_window):
        with h5py.File(data_filename, "r") as fid:
            d = fid[data_key][data_window]
            d = d.astype("float32")
            d[mask_indices] = np.nan
            return d

    data_convolved = {}

    for param_index, param in enumerate(BrdfModelParameters):
        temp = {}

        # get clean dataset for all available dates for given window
        data_clean = np.full(
            shape=(2 * filter_size + len(all_data_keys),) + shape_of_window(window),
            fill_value=np.nan,
            dtype="float32",
        )

        for layer, key in enumerate(all_data_keys):
            data_clean[filter_size + layer] = __get_clean_data(
                filename, key, (param_index,) + window
            )

        # set data that needs to be padded at the end and front to perform convolution
        data_clean[:filter_size] = np.array(
            [data_clean[filter_size] for i in range(filter_size)]
        )
        data_clean[len(all_data_keys) + filter_size :] = np.array(
            [
                data_clean[filter_size + len(all_data_keys) - 1]
                for i in range(filter_size)
            ]
        )

        # mask where data are invalid
        invalid_data = data_clean == NODATA
        data_clean = np.where(~invalid_data, data_clean, np.nan)

        # get mean across temporal axis to fill the np.nan in data_clean array
        median_data = np.nanmedian(data_clean, axis=0)
        for index in range(data_clean.shape[0]):
            data_clean[index] = np.where(
                invalid_data[index], median_data, data_clean[index]
            )

        data_clean *= SCALE_FACTOR_2

        # perform convolution using Gaussian filter defined above
        for i in range(data_clean.shape[1]):
            for j in range(data_clean.shape[2]):
                data_clean[:, i, j] = np.convolve(
                    data_clean[:, i, j], filt, mode="same"
                )

        for index, key in enumerate(all_data_keys):
            temp[key] = data_clean[index + filter_size]

        data_convolved[param] = temp

    return data_convolved


def _get_measurement_info() -> Dict:
    """
    Returns band specifications for the BRDF fallback datasets
    """
    datatype_bandmap = {
        "Parameters": DTYPE_MAIN.names,
        "Shape_Indices": DTYPE_SUPPORT.names,
        "Shape_Parameters_Quality": DTYPE_QUALITY.names,
        "Parameters_Quality": DTYPE_QUALITY_COUNT.names,
    }

    measurements = {}

    for dt_key, dtype_names in datatype_bandmap.items():
        for band_name in BAND_LIST:
            for dt_name in dtype_names:
                h5_layer = "_".join(("BRDF_Albedo", dt_key, band_name))
                measurements[h5_layer + "_" + dt_name.lower()] = {
                    "path": "",
                    "layer": h5_layer,
                    "band": dt_name,
                }
    return measurements


def munge_metadata(fp: str, only_id: bool = True) -> Dict:
    """
    extracts metadata from MCD43A1 .h5 files. Returns only uuid of the
    h5 files if only_id is set to True. Else general metadata attributes
    of the h5 file is returned.
    """
    with h5py.File(fp, "r") as src:
        src_md = YAML.load(src[METADATA_OFFSET][()].item())

        if only_id:
            return src_md["id"]

    metadata_doc = {
        "id": None,
        "product": {"href": FALLBACK_PRODUCT_HREF},
        "crs": src_md["crs"],
        "geometry": src_md["geometry"],
        "grids": src_md["grids"],
        "measurements": _get_measurement_info(),
        "lineage": {"doy_average": [], "brdf_threshold": []},
        "properties": {
            "eo:instrument": src_md["properties"]["eo:instrument"],
            "eo:platform": src_md["properties"]["eo:platform"],
            "eo:gsd": src_md["properties"]["eo:gsd"],
            "eo:epsg": src_md["properties"]["eo:epsg"],
            "item:providers": [
                {"name": "Geoscience Australia", "roles": ["processor", "host"]}
            ],
            "odc:creation_datetime": get_datetime().isoformat(),
            "odc:file_format": "HDF5",
            "odc:region_code": src_md["properties"].get("odc:region_code", None),
        },
    }

    return metadata_doc


def post_cleanup_process(
    window,
    day_numbers,
    h5_info,
    outdir,
    tile,
    clean_data_file,
    filter_size,
    band,
    bad_indices,
):
    """
    This function implements gaussian smoothing of the cleaned dataset,
    temporal averaging of the gaussian smooth dataset,
    quality check based on brdf_shape indices and writes the final
    brdf averaged parameters to a h5 file.
    """

    data_convolved = apply_convolution(
        clean_data_file, h5_info, window, filter_size, bad_indices[window]
    )

    for doy in day_numbers:
        avg_data = temporal_average(data_convolved, doy)

        # Filter dictionary containing h5 files to only include files with DOY of interest
        filtered_h5_info = filter_h5_info(h5_info, doy)

        band_name = quality_band_name(band)

        # Calculate quality counts for window, 1st extension returns array only without window
        quality_count_q0_window = get_qualityband_count_window(
            filtered_h5_info, band_name, window, include_quality_values=[0]
        )[1]
        quality_count_q1_window = get_qualityband_count_window(
            filtered_h5_info, band_name, window, include_quality_values=[1]
        )[1]
        quality_count_notq0q1_window = get_qualityband_count_window(
            filtered_h5_info, band_name, window, exclude_quality_values=[0, 1]
        )[1]

        filtered_data = brdf_indices_quality_check(
            quality_count_q0_window,
            quality_count_q1_window,
            quality_count_notq0q1_window,
            avg_data,
        )

        outfile = pjoin(outdir, BRDF_AVG_FILE_BAND_FMT.format(tile, doy, band))
        with LOCKS[outfile]:
            with h5py.File(outfile) as fid:
                write_chunk(filtered_data, fid, band, window=window)


def write_brdf_fallback_band(
    h5_info: Dict,
    tile: str,
    band: str,
    outdir: Path,
    filter_size: int,
    day_numbers: Set[int],
    pthresh: float,
    data_chunks: Iterable[int],
    compute_chunks: Iterable[int],
    nprocs: int,
    compression: H5CompressionFilter,
    filter_opts: Optional[Dict],
):
    """
    Computes pre-MODIS BRDF for a single band for every unique day of the year (from day_numbers)  # noqa: E501 # pylint: disable=line-too-long
    """

    min_numpix_required = np.rint((pthresh / 100.0) * len(h5_info))

    shape, attrs = get_band_info(h5_info, albedo_band_name(band))
    shape = shape[-2:]

    # get counts of good pixel quality
    quality_count = get_qualityband_count(
        h5_info=h5_info,
        band_name=quality_band_name(band),
        shape=shape,
        compute_chunks=compute_chunks,
        nprocs=nprocs,
        include_quality_values=[0],
    )

    quality_count = np.ma.masked_invalid(quality_count)

    # get the index where band_quality number is less the minimum number of valid pixels required  # noqa: E501 # pylint: disable=line-too-long
    bad_indices = (quality_count < min_numpix_required).filled(False)

    thresholds = calculate_thresholds(
        h5_info, albedo_band_name(band), shape, compute_chunks, nprocs=nprocs
    )
    quality_count = None

    clean_data_file = pjoin(outdir, "clean_data_{}_{}.h5".format(band, tile))

    LOCKS[clean_data_file] = Lock()

    with h5py.File(clean_data_file, "w") as clean_data:
        for key in h5_info:
            create_dataset(
                clean_data,
                key,
                (4, shape[0], shape[1]),
                {},
                chunks=(1,) + data_chunks,
                compression=H5CompressionFilter.LZF,
                filter_opts={},
            )

    with Pool(processes=nprocs) as pool:
        pool.starmap(
            apply_threshold,
            [
                (
                    clean_data_file,
                    h5_info,
                    albedo_band_name(band),
                    window,
                    filter_size,
                    thresholds,
                    bad_indices[window],
                )
                for window in generate_windows(shape, compute_chunks=compute_chunks)
            ],
        )
    for doy in day_numbers:
        outfile = pjoin(outdir, BRDF_AVG_FILE_BAND_FMT.format(tile, doy, band))
        LOCKS[outfile] = Lock()

        with h5py.File(outfile, "w") as fid:
            create_brdf_datasets(
                fid,
                band,
                shape,
                attrs,
                chunks=data_chunks,
                compression=compression,
                filter_opts=filter_opts,
            )

    with Pool(processes=nprocs) as pool:
        pool.starmap(
            post_cleanup_process,
            [
                (
                    window,
                    day_numbers,
                    h5_info,
                    outdir,
                    tile,
                    clean_data_file,
                    filter_size,
                    band,
                    bad_indices,
                )
                for window in generate_windows(shape, compute_chunks)
            ],
        )

    os.remove(clean_data_file)


def write_brdf_fallback(
    brdf_dir: Path,
    outdir: Path,
    tile: str,
    filter_size: int,
    nprocs: int,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    compression: H5CompressionFilter = H5CompressionFilter.LZF,
    filter_opts: Optional[Dict] = None,
):
    """
    Generates a single h5 files per day of the year containing first seven MODIS bands as sub-dataset.  # noqa: E501 # pylint: disable=line-too-long
    """
    h5_info = hdf5_files(
        brdf_dir,
        tile=tile,
        start_date=start_dt.date() if start_dt else None,
        end_date=end_dt.date() if end_dt else None,
    )

    day_numbers = sorted(set(folder_doy(item) for item in h5_info))

    # Day 365 will be used to represent the leap day
    if 366 in day_numbers:
        day_numbers.remove(366)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for band in BAND_LIST:
            write_brdf_fallback_band(
                h5_info,
                tile,
                band,
                tmp_dir,
                filter_size,
                day_numbers,
                pthresh=10.0,
                data_chunks=(150, 150),
                compute_chunks=(150, 150),
                nprocs=nprocs,
                compression=compression,
                filter_opts=filter_opts,
            )
        # get metadata for a tile
        start_ds, *_ = sorted(h5_info.keys())
        tile_metadata = munge_metadata(h5_info[start_ds], only_id=False)

        with Pool(processes=nprocs) as pool:
            ids_brdf = pool.starmap(
                munge_metadata, [(fp,) for key, fp in h5_info.items()]
            )

        tile_metadata["lineage"]["brdf_threshold"] = ids_brdf

        with Pool(processes=nprocs) as pool:
            pool.starmap(
                concatenate_files,
                [
                    (
                        [
                            str(fp)
                            for fp in Path(tmp_dir).rglob(
                                BRDF_MATCH_PATTERN.format(tile, doy)
                            )
                        ],
                        os.path.join(outdir, BRDF_AVG_FILE_FMT.format(tile, doy)),
                        h5_info,
                        doy,
                        tile_metadata,
                    )
                    for doy in day_numbers
                ],
            )

    # symlink doy 366 to the final results of doy 365 results
    os.symlink(
        pjoin(outdir, BRDF_AVG_FILE_FMT.format(tile, 365)),
        pjoin(outdir, BRDF_AVG_FILE_FMT.format(tile, 366)),
    )


@click.command()
@click.option(
    "--brdf-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="BRDF dataset directory root.",
)
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output BRDF dataset directory root.",
)
@click.option("--tile", help="Modis tile reference. Example: h29v12", default="h29v12")
@click.option("--start-dt", type=click.DateTime(), default="2002-1-1")
@click.option("--end-dt", type=click.DateTime(), default="2018-12-31")
@click.option("--filter-size", type=int, default=22)
@click.option(
    "--nprocs", required=True, type=int, help="Number of processors used in parrallel"
)
@_compression_options
def main(
    brdf_dir: click.Path,
    outdir: click.Path,
    tile: str,
    start_dt: click.DateTime,
    end_dt: click.DateTime,
    filter_size: int,
    nprocs: int,
    compression: H5CompressionFilter,
    filter_opts: Optional[Dict] = None,
):
    """main function to execute brdf fallback computation for a MODIS tile."""

    write_brdf_fallback(
        brdf_dir=Path(brdf_dir),
        outdir=Path(outdir),
        tile=tile,
        filter_size=filter_size,
        nprocs=nprocs,
        start_dt=start_dt,
        end_dt=end_dt,
        compression=compression,
        filter_opts=filter_opts,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
