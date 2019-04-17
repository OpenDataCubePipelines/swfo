#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer


setup(name='swfo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      url='https://github.com/OpenDataCubePipelines/swfo',
      description=('A collection of data transformation scripts'),
      packages=find_packages(exclude=("tests", )),
      install_requires=[
          'click',
          'GDAL',
          'h5py',
          'numpy',
          'pandas',
          'python-dateutil',
          'rasterio',
          'shapely',
          'wagl',
      ],
      extras_require=dict(
          test=[
              'pytest',
              'pytest-flake8',
              'deepdiff',
              'flake8',
              'pep8-naming',
          ],
      ),
      entry_points=dict(
          console_scripts=[
              'swfo-convert=swfo.convert:cli'
          ]
      ),
      include_package_data=True)
