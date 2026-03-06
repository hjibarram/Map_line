Examples
========

The MapLines repository contains several example workflows that illustrate
typical usage.

Single-spectrum examples
------------------------

Examples for fitting individual spectra are available in:

.. code-block:: text

   examples/single_spectra/

These include configuration files and execution scripts.

IFU cube examples
-----------------

Examples for fitting data cubes can be found in:

.. code-block:: text

   examples/ifu_spectra/

These workflows demonstrate spatially resolved emission-line analysis.

Notebook examples
-----------------

Interactive examples are provided as Jupyter notebooks in:

.. code-block:: text

   examples/notebooks/

These notebooks demonstrate:

- spectral fitting
- parameter exploration
- diagnostic plotting

Suggested workflow
------------------

A typical analysis using MapLines follows these steps:

1. prepare a configuration file
2. define the wavelength range of interest
3. run the fitting routine
4. inspect diagnostic plots
5. analyze the resulting parameter maps