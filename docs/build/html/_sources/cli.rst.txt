Command Line Interface
======================

Overview
--------

MapLine provides a command-line interface (CLI) to perform emission-line
fitting directly from the terminal. The CLI allows users to run the full
analysis workflow without writing Python scripts.

The main entry point is:

.. code-block:: bash

   run_mapline

Two main commands are available:

- ``run`` — perform emission-line fitting on IFU cubes
- ``runoned`` — fit emission lines in a single spectrum

.. code-block:: bash

   run_mapline --help
   run_mapline run --help
   run_mapline runoned --help

These commands internally call the fitting routines implemented in
``MapLines.tools.line_fit``.

CLI reference
-------------

.. click:: _cli_wrapper:cli
   :prog: run_mapline
   :nested: full

..    ---------------------------------------------------------------------

    run_mapline run
    ===============

    Run the emission-line mapper on an IFU data cube.

    Basic usage
    -----------

    .. code-block:: bash

       run_mapline run -g config.yml

    The configuration file defines the cube name, redshift, fitting setup,
    and output directories.

    Options
    -------

    -g, --config_file
        YAML configuration file describing the fitting setup.

    -n, --name
        Name of the IFS cube.

    -o, --name_out
        Base name for the output files.

    -m, --mask
        Mask cube used during fitting.

    -p, --path
        Path to the input data cubes.

    -y, --path_out
        Directory where output products will be written.

    -c, --ncpus
        Number of CPUs used during the fitting process.

    -z, --zt
        Redshift of the object.

    -f, --fluxf
        Multiplicative factor applied to the flux scale.

    -q, --line_config
        Line-model configuration file.

    -w, --line_config_path
        Directory containing the line configuration file.

    Model options
    -------------

    -k, --kskew
        Enable skewed Gaussian line profiles.

    -x, --outflow
        Enable outflow components in the emission-line model.

    -l, --lorentz
        Use Lorentzian profiles.

    -d, --powlaw
        Include a power-law continuum component.

    -r, --feii
        Enable FeII template fitting.

    Analysis options
    ----------------

    -t, --test
        Run analysis for a single spaxel for testing purposes.

    -i, --it
        Spaxel index (x) used during test runs.

    -j, --jt
        Spaxel index (y) used during test runs.

    -e, --error
        Automatically compute the error vector.

    -b, --bcont
        Disable automatic continuum subtraction.

    -s, --sprogressd
        Disable the progress bar.

    Outputs
    -------

    The command generates:

    - model FITS cubes
    - parameter FITS cubes
    - diagnostic plots

    Output files are written to the directory specified by ``--path_out``.

    Example
    -------

    .. code-block:: bash

       run_mapline run \
           -g config.yml \
           -p data/ \
           -y outputs/ \
           -z 0.015 \
           -c 16

    ---------------------------------------------------------------------

    run_mapline runoned
    ===================

    Fit emission lines in a one-dimensional spectrum.

    Basic usage
    -----------

    .. code-block:: bash

       run_mapline runoned -g config.yml

    Options
    -------

    -g, --config_file
        YAML configuration file describing the fitting setup.

    -n, --name
        Name of the input spectrum.

    -p, --path
        Directory containing the spectrum.

    -y, --path_out
        Output directory.

    -c, --ncpus
        Number of CPUs used for the fitting process.

    -z, --zt
        Redshift of the source.

    -f, --fluxf
        Flux scaling factor.

    -q, --line_config
        Line-model configuration file.

    -w, --line_config_path
        Directory containing the line configuration file.

    Model options
    -------------

    -k, --kskew
        Enable skewed line profiles.

    -u, --outflow
        Enable outflow components.

    -l, --lorentz
        Use Lorentzian line profiles.

    Input options
    -------------

    -i, --input_format
        Format of the input spectrum (default: CSV).

    -m, --smoth
        Apply smoothing to the spectrum.

    -h, --ker
        Width of the smoothing kernel in pixels.

    Outputs
    -------

    The command generates:

    - best-fit model spectra
    - component-separated spectra
    - diagnostic plots

    Example
    -------

    .. code-block:: bash

       run_mapline runoned \
           -g config.yml \
           -p spectra/ \
           -y outputs/