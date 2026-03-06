Quickstart
==========

Overview
--------

MapLines is designed to model emission-line spectra using flexible
line profiles and Bayesian parameter estimation.

It supports both:

- single-spectrum fitting
- IFU cube fitting

Example: fitting a single spectrum
----------------------------------

The simplest workflow uses the function
``line_fit_single`` from ``MapLines.tools.line_fit``.

.. code-block:: python

   from MapLines.tools.line_fit import line_fit_single

   line_fit_single(
       file1="spectrum.fits",
       file_out="fit_result",
       file_out2="fit_parameters",
       name_out2="example_fit",
       config_lines="line_prop.yml",
       z=0.01,
       lA1=6500,
       lA2=6600
   )

Inputs
------

The main inputs are:

- a FITS spectrum
- a configuration file describing the emission lines
- the redshift of the source
- the wavelength range to fit

Outputs
-------

The fitting routine produces:

- model spectra
- component-separated spectra
- parameter tables
- diagnostic plots

Example: fitting an IFU cube
----------------------------

For spatially resolved spectroscopy:

.. code-block:: python

   from MapLines.tools.line_fit import line_fit

   line_fit(
       file1="cube.fits",
       file2="auxiliary.fits",
       file3="mask.fits",
       file_out="cube_model",
       file_out2="cube_parameters",
       name_out2="cube_example",
       config_lines="line_prop.yml",
       z=0.01
   )

This produces parameter maps and diagnostic outputs for every spatial pixel.