Methodology
===========

Overview
--------

MapLines models emission-line spectra using parametric line profiles
combined with Bayesian parameter estimation.

The approach is designed to analyze both integrated spectra and
spatially resolved IFU observations.

Spectral models
---------------

The spectral model can include several components:

- Gaussian emission lines
- skewed Gaussian profiles
- Lorentzian profiles
- Voigt profiles
- outflow components
- power-law continuum
- FeII templates

Each component is defined in the configuration file and combined
to produce the total model spectrum.

Parameter inference
-------------------

Parameter estimation is performed using Markov Chain Monte Carlo (MCMC)
sampling through the ``emcee`` ensemble sampler.

The posterior probability is defined as:

.. math::

    P(\theta | D) \propto L(D | \theta) P(\theta)

where:

- :math:`L` is the likelihood
- :math:`P` is the prior
- :math:`D` is the observed spectrum

Likelihood function
-------------------

The likelihood assumes Gaussian uncertainties in the observed spectrum.

It is implemented in:

``MapLines.tools.priors``

Posterior sampling
------------------

Posterior sampling is performed using the routines in:

``MapLines.tools.mcmc``

These routines generate chains of model parameters that sample the
posterior distribution.

Outputs
-------

The fitting procedure produces several products:

- best-fit spectra
- posterior parameter distributions
- parameter maps (for IFU data)
- diagnostic plots

These products can then be used to study the physical properties
of ionized gas in galaxies and active galactic nuclei.