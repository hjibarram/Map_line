Configuration Files
===================

Overview
--------

MapLines uses YAML configuration files to define the emission-line model
and the parameter constraints used during fitting.

These files allow flexible modeling of complex emission-line systems.

Typical configuration files
---------------------------

Two configuration files are typically used:

- ``config.yml``
- ``line_prop.yml``

The first defines the general fitting setup, while the second describes
the emission-line components.

Line properties
---------------

The line configuration file defines:

- line names
- rest wavelengths
- amplitude relations
- velocity constraints
- width constraints

Example structure:

.. code-block:: yaml

   Halpha:
       lambda: 6562.8
       amplitude: free
       velocity: tied
       width: tied

Parameter linking
-----------------

MapLines allows parameters to be tied between components.

For example:

- velocity of [NII] tied to Hα
- width of multiple lines tied together

This ensures physically consistent models.

Priors
------

Parameter priors define the allowed ranges for each parameter.

These include:

- amplitude limits
- velocity limits
- width limits

The priors are evaluated by the module:

``MapLines.tools.priors``

which defines the likelihood and posterior functions used by the MCMC sampler.