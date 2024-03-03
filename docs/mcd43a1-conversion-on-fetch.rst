MCD43A1 Fetch Integration
=========================

SWFO's mcd43a1 h5-md command line tool was issued in conjunction with `Fetch <https://github.com/GeoscienceAustralia/fetch>`_. The scope of fetch is planned to increase in the future; but until then the MCD43A1 collection can be continually pulled and converted by adding the following options to the *Modis BRDF* example.

::

  Modis BRDF:
    ...
    process: !shell
      command: '([[ -f {parent_dir}/"$(basename {filename} .xml)" && -f {parent_dir}/"$(basename {filename} .xml)".xml ]] && swfo-convert mcd43a1 h5-md --fname {parent_dir}/"$(basename {filename} .xml)" --outdir {parent_dir} ) || true'
      expect_file: "{parent_dir}/{filename}"

For the Australian continental region the following name_pattern was used to determine which tiles to retrieve:

::

    MCD43A1\.A[0-9]{7}\.h(2[7-9]|3[0-2])v(09|1[0-3])\.006\.[0-9]{13}\.hdf

Limitation:

* expect_file checks existance of the hdf file; not the h5 file due to requiring both the dataset and associated metadata document prior to conversion.

  * This avoided additional complexity/extension of the http-fetch directive.
