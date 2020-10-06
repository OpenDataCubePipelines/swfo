#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="swfo",
    use_scm_version=True,
    url="https://github.com/OpenDataCubePipelines/swfo",
    description=("A collection of data transformation scripts"),
    packages=find_packages(exclude=("tests",)),
    setup_requires=["setuptools_scm"],
    install_requires=[
        "affine",
        "click",
        "eodatasets",
        "GDAL",
        "netCDF4",
        "h5py",
        "numpy",
        "pandas",
        "python-dateutil",
        "rasterio",
        "ruamel.yaml",
        "scipy",
        "shapely",
        "wagl",
        "importlib-metadata;python_version<'3.8'",
    ],
    dependency_links=[
        "git+https://github.com/GeoscienceAustralia/wagl@develop#egg=wagl",
        "git+https://github.com/GeoscienceAustralia/eo-datasets@eodatasets-0.12#egg=eodatasets-0.12",  # noqa: E501
    ],
    extras_require=dict(
        test=["pytest", "pytest-flake8", "deepdiff", "flake8", "pep8-naming"]
    ),
    entry_points=dict(console_scripts=["swfo-convert=swfo.convert:cli"]),
    include_package_data=True,
)
