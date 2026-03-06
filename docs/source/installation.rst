Installation
============

Overview
--------

MapLines is a Python package designed to perform emission-line fitting
in astronomical spectra and integral-field spectroscopy (IFS) data.
It supports parametric modeling of emission lines, Bayesian inference
with MCMC, and spatially resolved analysis of data cubes.

Requirements
------------

MapLines relies on the scientific Python ecosystem and requires:

- Python >= 3.9
- numpy
- scipy
- matplotlib
- astropy
- emcee
- pyyaml
- corner

Optional packages used in visualization include:

- cmasher
- astropy.wcs
- astropy.io.fits

Install from source
-------------------

Clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/USERNAME/MapLines.git
   cd MapLines
   pip install -e .

Editable installation allows modifications to the source code without
reinstalling the package.

Install dependencies manually
-----------------------------

If needed, dependencies can be installed individually:

.. code-block:: bash

   pip install numpy scipy matplotlib astropy emcee pyyaml corner

Building the documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx myst-parser sphinx-rtd-theme

Then run:

.. code-block:: bash

   cd docs
   make html

The generated documentation will appear in:

.. code-block:: text

   docs/build/html/index.html