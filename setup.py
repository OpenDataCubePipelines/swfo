#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="swfo",
    use_scm_version=True,
    url="https://github.com/OpenDataCubePipelines/swfo",
    description="A collection of data transformation scripts",
    packages=find_packages(exclude=("tests",)),
    setup_requires=["setuptools_scm"],
    install_requires=[
        'ard-pipeline',
        "affine",
        "click",
        "eodatasets3",
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
        "importlib-metadata;python_version<'3.8'",
    ],
    dependency_links=[
        "git+https://github.com/OpenDataCubePipelines/ard-pipeline.git@main#egg=ard-pipeline",
    ],
    extras_require=dict(
        test=["pytest", "pytest-flake8", "deepdiff", "flake8", "pep8-naming"]
    ),
    entry_points=dict(console_scripts=["swfo-convert=swfo.convert:cli"]),
    include_package_data=True,
)
