#!/usr/bin/env python

"""
BRDF shape parameters and indices computation functions
"""

import numpy as np

# these constants required for computation BRDF shape function and Indices
# which are provided by David Jupp's (2018) document 'Properties of BRDF
# shape parameters and indices'
CONSTANTS = {
    "c11": 0.015683596,
    "c12": 0.055165295,
    "c22": 0.371423479,
    "gvol": 0.189184,
    "ggeo": -1.377622,
    "afxmin": 0.01,
    "afxmax": 3.2,
    "rmsmin": 0.0,
    "rmsmax": 1.4,
}
CONSTANTS["ra"] = (
    CONSTANTS["c11"]
    - (2 * CONSTANTS["c12"]) * (CONSTANTS["gvol"] / CONSTANTS["ggeo"])
    + (CONSTANTS["c22"] * (CONSTANTS["gvol"] / CONSTANTS["ggeo"]) ** 2)
)
CONSTANTS["rb"] = (1.0 / CONSTANTS["ggeo"]) * (
    CONSTANTS["c12"] - CONSTANTS["c22"] * (CONSTANTS["gvol"] / CONSTANTS["ggeo"])
)
CONSTANTS["rc"] = CONSTANTS["c22"] / CONSTANTS["ggeo"] ** 2


def get_shape_function(fiso, fvol, fgeo):
    """
    This functions computes the brdf shape function alpha1 and alpha2
    from brdf shape parameters fiso, fvol and fgeo.

    :param fiso: 'array' isotrophic contribution (fiso)
    :param fvol: 'array' weights for volume-scattering (fvol) contribution
    :param fgeo: 'array' weights for geometric-optical contribution

    :return:
        alpha1: 'array' ratio of brdf shape parameter fvol/fiso
        alpha2: 'array' ratio of brdf shape paramter fgeo/fiso
    """
    alpha1 = fvol / fiso
    alpha2 = fgeo / fiso

    return alpha1, alpha2


def get_rms_indices(alpha1, alpha2):
    """
    This function computes the Root Mean Square (RMS) statistics
    for the brdf shape function.

    :param alpha1: 'array' ratio of brdf shape parameter fvol/fiso
    :param alpha2: 'array' ratio of brdf shape parameter fgeo/fiso

    :return:
        rms: 'array' RMS statistics for brdf shape function
    """
    alpha1_square = alpha1 * alpha1
    alpha2_square = alpha2 * alpha2
    alpha12 = alpha1 * alpha2

    return np.sqrt(
        (CONSTANTS["c11"] * alpha1_square)
        + (2 * CONSTANTS["c12"] * alpha12)
        + (CONSTANTS["c22"] * alpha2_square)
    )


def get_afx_indices(alpha1, alpha2):
    """
    This function computes the Anisotropic Flat Index (AFX) statistics
    for the brdf shape function.

    :param alpha1: 'array' ratio of brdf shape parameter fvol/fiso
    :param alpha2: 'array' ratio of brdf shape parameter fgeo/fiso

    :return:
        afx: 'array' AFX statistics for brdf shape function
    """

    return 1 + (CONSTANTS["gvol"] * alpha1) + (CONSTANTS["ggeo"] * alpha2)


def get_mean_shape_param(fiso_mean, fvol_mean, fgeo_mean, cov_iso):
    """
    This function computes the mean brdf shape function (alpha1 and alpha2)
    using lognormal model.

    :param fiso_mean: 'array' of mean brdf parameter fiso
    :param fvol_mean: 'array' of mean brdf parameter fvol
    :param fgeo_mean: 'array' of mean brdf paramter fgeo
    :param cov_iso: 'array' of coefficient of variation for fiso parameter

    :return:
        alpha1_mean: 'array' of mean brdf shape parameter fvol/fiso
        alpha2_mean: 'array' of mean brdf shape parameter fgeo/fiso
    """

    alpha1_mean = (fvol_mean / fiso_mean) * (1 + cov_iso**2)
    alpha2_mean = (fgeo_mean / fiso_mean) * (1 + cov_iso**2)

    return alpha1_mean, alpha2_mean


def get_cov_shape_param(cov_iso, cov_vol, cov_geo):
    """
    This function computes the coefficient of variation for brdf_ shape function
    (alpha1 and alpha2) using lognormal model.

    :param cov_iso: 'array' of coefficient of variation for fiso parameter
    :param cov_vol: 'array' of coefficient of variation for fvol parameter
    :param cov_geo: 'array' of coefficient of variation for fgeo paramter

    :return:
        alpha1_cov = 'array' of coefficient of variation for alpha1
        alpha2_cov = 'array' of coefficient of variation for alpha2
    """

    alpha1_cov = np.sqrt((1 + cov_vol**2) * (1 + cov_iso**2) - 1)
    alpha2_cov = np.sqrt((1 + cov_geo**2) * (1 + cov_iso**2) - 1)

    return alpha1_cov, alpha2_cov


def get_std_afx_indices(alpha1_mean, alpha2_mean, alpha1_cov, alpha2_cov):
    """
    This function computes the Anisotropic Flat Index (AFX) standard deviation
    for the brdf shape function using lognormal model.

    :param alpha1_mean: 'array' of mean brdf shape parameter fvol/fiso computed using
        lognormal model
    :param alpha2_mean: 'array' of mean brdf shape parameter fgeo/fiso computed using
        lognormal model
    :param alpha1_cov: 'array' of coefficient of variation for alpha1 computed using
        lognormal model
    :param alpha2_cov: 'array' of coefficient of variation for alpha2 computed in
        lognormal model
    :return:
        afx_std: 'array' of afx standard deviation computed using lognormal model
    """
    return np.sqrt(
        (CONSTANTS["gvol"] * alpha1_mean * alpha1_cov) ** 2
        + (CONSTANTS["ggeo"] * alpha2_mean * alpha2_cov) ** 2
    )


def get_std_rms_indices(alpha1_mean, alpha2_mean, alpha1_cov, alpha2_cov, rms):
    """
    This function computes the Root Mean Square standard deviation for brdf shape function
    using lognormal model

    :param alpha1_mean: 'array' of mean brdf shape parameter fvol/fiso computed
        using lognormal model
    :param alpha2_mean: 'array' of mean brdf shape parameter fgeo/fiso computed
        using lognormal model
    :param alpha1_cov: 'array' of coefficient of variation for alpha1 computed
        using lognormal model
    :param alpha2_cov: 'array' of coefficient of variation for alpha2 computed
        using lognormal model
    :param rms: 'array' RMS statistics computed using brdf shape function

    :return:
        rms_std: 'array' of rms standard deviation in lognormal space
    """

    t1 = (
        (
            (CONSTANTS["c11"] * alpha1_mean**2)
            + (CONSTANTS["c12"] * alpha1_mean * alpha2_mean)
        )
        * alpha1_cov
    ) ** 2
    t2 = (
        (
            (CONSTANTS["c12"] * alpha1_mean * alpha2_mean)
            + (CONSTANTS["c22"] * alpha2_mean**2)
        )
        * alpha2_cov
    ) ** 2

    return np.sqrt((t1 + t2)) / rms


def get_unfeasible_mask(rms, afx):
    """
    This function returns the infeasibility index mask generated from
    brdf shape indicies.

    :param rms: 'array' of Root Mean Square (RMS) statistics
    :param afx: 'array' of Anisotropic Flat Index (AFX) statistics

    :return:
        mask: mask set by unfeasible values (b2 < ac)
    """
    a = CONSTANTS["ra"]
    b = CONSTANTS["rb"] * (afx - 1)
    c = CONSTANTS["rc"] * (afx - 1) ** 2 - rms**2
    return np.ma.masked_where(b**2 < a * c, rms).mask
