Example Gallery
===============

This page collects representative MapLines workflows.

Single-spectrum fit
-------------------

Fit an emission-line complex in a one-dimensional spectrum.

.. code-block:: python

   from MapLines.tools.line_fit import line_fit_single

   line_fit_single(
       file1="spectrum.fits",
       file_out="fit_result",
       file_out2="fit_parameters",
       name_out2="single_example",
       config_lines="line_prop.yml",
       z=0.01,
       lA1=6500.0,
       lA2=6600.0
   )

IFU cube analysis
-----------------

Fit a spectral cube spaxel by spaxel.

.. code-block:: python

   from MapLines.tools.line_fit import line_fit

   line_fit(
       file1="cube.fits",
       file2="aux.fits",
       file3="mask.fits",
       file_out="cube_model",
       file_out2="cube_params",
       name_out2="ifu_example",
       config_lines="line_prop.yml",
       z=0.01
   )

BPT diagnostics
---------------

Build and visualize a BPT classification map from fitted products.

.. code-block:: python

   import MapLines.tools.plot_tools as pt

   pt.plot_bpt_map2(
       fileR="red_fit.fits.gz",
       fileB="blue_fit.fits.gz",
       name="galaxy_name",
       path="outputs/"
   )

Region extraction
-----------------

Extract spectra or values from DS9-defined apertures and paths.

.. code-block:: python

   import MapLines.tools.tools as tools

   wave, spec = tools.extract_spec(
       filename="cube.fits",
       ra="10:27:00.0",
       dec="+17:49:00.0",
       rad=1.5
   )

Map visualization
-----------------

Display parameter maps and derived products.

.. code-block:: python

   import MapLines.tools.plot_tools as pt

   pt.plot_single_map(
       file="Ha_map.fits.gz",
       valmax=100,
       valmin=1,
       tit="Halpha Flux",
       lab="[10^{-16} erg/s/cm^2/arcsec^2]"
   )