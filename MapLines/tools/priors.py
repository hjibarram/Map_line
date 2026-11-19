#!/usr/bin/env python

"""
MapLines.tools.priors
=====================

Likelihood and prior functions used by MapLines spectral models.

This module defines the statistical functions required for Bayesian
inference in MapLines. In particular, it provides routines to evaluate:

- the log-likelihood of a spectral model given an observed spectrum
- the log-prior for the model parameters
- the log-posterior probability used by the MCMC sampler
- the posterior probability for multiple parameter sets
- optional GPU-accelerated likelihood and posterior evaluations

These functions are used together with the spectral models defined in
``MapLines.tools.models`` and the MCMC driver implemented in
``MapLines.tools.mcmc``.

The module supports posterior probability calculations for spectral
models with different emission-line profiles and optional components,
including asymmetric profiles, outflows, power-law continua, and FeII
emission templates.

In addition to standard CPU calculations, the module provides
interfaces for evaluating multiple MCMC parameter vectors using
vectorized spectral models. CUDA acceleration is supported through
CuPy, allowing spectral models and their likelihoods to be evaluated
simultaneously for batches of walkers.

Notes
-----
The likelihood assumes Gaussian uncertainties in the observed spectrum,
while the prior enforces physically and empirically motivated parameter
ranges defined in the configuration files.

The original CPU likelihood, prior, and posterior functions are
preserved to maintain compatibility with existing MapLine fitting
routines and scientific results.

The batched CPU posterior implementation provides a reference for
validating numerical consistency with GPU-accelerated calculations.

In the CUDA implementation, parameter priors are evaluated using
the original CPU functions, while spectral models and likelihoods
are evaluated on the GPU. Only parameter sets satisfying the prior
constraints are submitted for GPU evaluation.

The GPU-computed log-posterior values are returned as NumPy arrays
to maintain compatibility with the ``emcee`` ensemble sampler.

CUDA acceleration is optional and requires CuPy, a compatible NVIDIA
GPU, and a vectorized spectral model implementation supporting the
selected components. Unsupported CUDA configurations must be handled
by the calling code before starting the MCMC sampling process.

The underlying Bayesian formulation, prior constraints, and MCMC
sampling algorithm remain unchanged.
"""
import MapLines.tools.models as mod
import numpy as np
'''This module contains the functions to calculate the likelihood and prior of the models'''


def lnlike_gauss_Lin(theta, spec, specE , x, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii, data):
    """
    Compute the log-likelihood of a spectral line model.

    The likelihood is calculated assuming Gaussian uncertainties
    in the observed spectrum.

    Parameters
    ----------
    theta : array_like
        Model parameters.
    spec : ndarray
        Observed spectrum.
    specE : ndarray
        Uncertainty of the spectrum.
    x : ndarray
        Spectral axis.
    waves0 : array_like
        Rest wavelengths of spectral lines.
    fac0, facN0 : array_like
        Flux normalization parameters.
    velfac0, velfacN0 : array_like
        Velocity scaling factors.
    fwhfac0, fwhfacN0 : array_like
        Line width scaling factors.
    names0 : list
        Names of spectral components.
    n_lines : int
        Number of emission lines.
    vals : dict
        Additional configuration parameters.

    Returns
    -------
    float
        Log-likelihood value.
    """
    '''This function calculates the likelihood of a double model for the spectrum'''
    model=mod.line_model(theta, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=x, skew=skew, voigt=voigt, lorentz=lorentz, outflow=outflow, powlaw=powlaw, feii=feii, data=data)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike


def lnlike_gauss_Lin_gpu(thetas, spec, specE, x, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0,
    names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii, data):
    """
    Compute vectorized log-likelihoods of spectral line models on CUDA.

    This function extends the original ``lnlike_gauss_Lin`` to
    evaluate multiple parameter sets simultaneously using CuPy.

    The likelihood assumes Gaussian uncertainties in the observed
    spectrum and preserves the original chi-squared formulation:

        ln(L) = -0.5 * nansum(((spec - model) / specE)**2)

    Spectral models are evaluated using the vectorized interface
    of ``MapLines.tools.models.line_model``.

    Parameters
    ----------
    thetas : array_like
        Model parameter vectors with shape (nwalkers_batch, ndim).

    spec : array_like
        Observed spectrum with shape (nwavelength,).

    specE : array_like
        Uncertainty of the observed spectrum.

    x : array_like
        Spectral axis used for model evaluation.

    waves0 : array_like
        Rest wavelengths of spectral lines.

    fac0, facN0 : array_like
        Flux normalization parameters.

    velfac0, velfacN0 : array_like
        Velocity scaling factors.

    fwhfac0, fwhfacN0 : array_like
        Line-width scaling factors.

    names0 : list
        Names of spectral components.

    n_lines : int
        Number of emission lines.

    vals : dict
        Additional model configuration parameters.

    skew, voigt, lorentz, outflow, powlaw, feii : bool
        Flags controlling the spectral model components.

    data : optional
        Additional information passed to the spectral model.

    Returns
    -------
    log_likelihood : cupy.ndarray
        Log-likelihood values with shape (nwalkers_batch,).

    Raises
    ------
    ValueError
        If the input dimensions are inconsistent.

    NotImplementedError
        If the selected spectral model is not supported by
        the CUDA vectorized implementation.

    Notes
    -----
    The numerical formulation is the same as the original
    CPU likelihood, including the use of ``nansum``.

    Spectral model evaluation and likelihood calculations
    are performed on the GPU.

    This function does not evaluate parameter priors.
    Posterior calculations are handled separately by
    ``lnprob_gauss_Lin_gpu``.

    The function requires CuPy and a functional CUDA runtime.
    The original CPU likelihood remains unchanged.
    """
    import cupy as cp

    thetas_gpu = cp.asarray(thetas)

    if thetas_gpu.ndim != 2:
        raise ValueError(
            "thetas must have shape (nwalkers_batch, ndim)"
        )

    spec_gpu = cp.asarray(spec)
    specE_gpu = cp.asarray(specE)
    x_gpu = cp.asarray(x)

    if spec_gpu.ndim != 1:
        raise ValueError(
            "spec must be a one-dimensional array"
        )

    if specE_gpu.shape != spec_gpu.shape:
        raise ValueError(
            "specE must have the same shape as spec"
        )

    if x_gpu.shape != spec_gpu.shape:
        raise ValueError(
            "x must have the same shape as spec"
        )

    models = mod.line_model(
        thetas_gpu,
        waves0, fac0, facN0,
        velfac0, velfacN0,
        fwhfac0, fwhfacN0,
        names0, n_lines, vals,
        x=x_gpu,
        skew=skew,
        voigt=voigt,
        lorentz=lorentz,
        outflow=outflow,
        powlaw=powlaw,
        feii=feii,
        data=data,
        vectorized=True,
        backend="cuda",
    )

    if models.shape != (
        thetas_gpu.shape[0], spec_gpu.size
    ):
        raise ValueError(
            "Vectorized model returned an invalid shape"
        )

    residuals = (
        spec_gpu[None, :] - models
    ) / specE_gpu[None, :]

    log_likelihood = -0.5 * cp.nansum(
        residuals**2.0,
        axis=1,
    )

    return log_likelihood



def lnprior_gauss_Lin(theta, Infvalues, Supvalues, valsp, skew=False, outflow=False, powlaw=False, feii=False):
    """
    Evaluate the prior probability for model parameters.

    The prior enforces parameter boundaries and additional
    physical constraints depending on the selected model
    components.

    Parameters
    ----------
    theta : array_like
        Model parameters.
    Infvalues : array_like
        Lower bounds of parameters.
    Supvalues : array_like
        Upper bounds of parameters.
    valsp : dict
        Dictionary containing prior limits for special components.
    skew, outflow, powlaw, feii : bool
        Flags enabling additional model components.

    Returns
    -------
    float
        Log-prior value (0 or -inf).
    """
    '''This function calculates the prior of a line model spectra'''  
    boolf=True 
    if skew:
        *f_parm,alp1,alpb=theta
        boolf=((alp1 >= -10) & (alp1 <= 10)) & ((alpb >= -10) & (alpb <= 10)) & boolf
    else:
        if outflow:
            *f_parm,F1o,dvO,fwhmO,alpo=theta
            boolf=((F1o >= valsp['f1i']) & (F1o <= valsp['f1s'])) & ((fwhmO >= valsp['fwhmOi']) & (fwhmO <= valsp['fwhmOs'])) & ((dvO >= valsp['dvOi']) & (dvO <= valsp['dvOs'])) & ((alpo >= valsp['alpOi']) & (alpo <= valsp['alpOs'])) & boolf
        else:
            f_parm=theta
    if powlaw:
        boolf=True 
        if feii:
            *f_parm,P1o,P2o,Fso,Fdo,Fao=theta
            boolf=((P1o >= valsp['P1i']) & (P1o <= valsp['P1s'])) & ((P2o >= valsp['P2i']) & (P2o <= valsp['P2s'])) & ((Fso >= valsp['Fsi']) & (Fso <= valsp['Fss'])) & ((Fdo >= valsp['Fdi']) & (Fdo <= valsp['Fds'])) & ((Fao >= valsp['Fai']) & (Fao <= valsp['Fas'])) & boolf
        else:
            *f_parm,P1o,P2o=theta
            boolf=((P1o >= valsp['P1i']) & (P1o <= valsp['P1s'])) & ((P2o >= valsp['P2i']) & (P2o <= valsp['P2s'])) & boolf
    elif feii:
        boolf=True
        *f_parm,Fso,Fdo,Fao=theta
        boolf=((Fso >= valsp['Fsi']) & (Fso <= valsp['Fss'])) & ((Fdo >= valsp['Fdi']) & (Fdo <= valsp['Fds'])) & ((Fao >= valsp['Fai']) & (Fao <= valsp['Fas'])) & boolf
    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf    

    if boolf:
        return 0.0
    else:
        return -np.inf            
                

def lnprob_gauss_Lin(theta, spec, specE, x, Infvalues, Supvalues, valsp, 
    waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, 
    n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii, data):
    """
    Compute the posterior probability of the spectral model.

    The posterior is defined as the sum of the prior and the likelihood.

    Parameters
    ----------
    theta : array_like
        Model parameters.

    Returns
    -------
    float
        Log-posterior probability.
    """
    '''This function calculates the posterior of the double model for the spectrum'''
    lp = lnprior_gauss_Lin(theta, Infvalues, Supvalues, valsp, skew=skew, outflow=outflow, powlaw=powlaw, feii=feii)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin(theta, spec, specE, x, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii, data) 



def lnprob_gauss_Lin_gpu(thetas, spec, specE, x, Infvalues, Supvalues, valsp, waves0, fac0, facN0,
    velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow,
    powlaw, feii, data):
    """
    Compute vectorized posterior probabilities using CUDA.

    This function extends the original ``lnprob_gauss_Lin``
    to evaluate multiple MCMC parameter vectors simultaneously.

    The posterior is defined as the sum of the prior and the
    Gaussian log-likelihood.

    The original parameter constraints are preserved by
    evaluating ``lnprior_gauss_Lin`` independently for each
    walker on the CPU.

    Parameter sets satisfying the priors are evaluated together
    on the GPU using ``lnlike_gauss_Lin_gpu``. Invalid parameter
    sets are assigned negative infinity without evaluating
    their spectral models.

    Parameters
    ----------
    thetas : array_like
        Model parameter vectors with shape (nwalkers_batch, ndim).

    spec : array_like
        Observed spectrum.

    specE : array_like
        Uncertainty of the observed spectrum.

    x : array_like
        Spectral axis.

    Infvalues, Supvalues : array_like
        Lower and upper boundaries of the model parameters.

    valsp : dict
        Dictionary containing prior limits for optional
        spectral components.

    waves0 : array_like
        Rest wavelengths of spectral lines.

    fac0, facN0 : array_like
        Flux normalization parameters.

    velfac0, velfacN0 : array_like
        Velocity scaling factors.

    fwhfac0, fwhfacN0 : array_like
        Line-width scaling factors.

    names0 : list
        Names of spectral components.

    n_lines : int
        Number of emission lines.

    vals : dict
        Additional model configuration parameters.

    skew, voigt, lorentz, outflow, powlaw, feii : bool
        Flags controlling the spectral model components
        and associated prior constraints.

    data : optional
        Additional information passed to the spectral model.

    Returns
    -------
    log_prob : numpy.ndarray
        Log-posterior probability values with shape
        (nwalkers_batch,).

        Parameter sets violating the original priors return
        negative infinity.

    Notes
    -----
    This function preserves the original prior constraints
    and Gaussian likelihood definition.

    Priors are evaluated on the CPU using the existing
    ``lnprior_gauss_Lin`` function.

    Only parameter sets satisfying the prior constraints
    are transferred to the GPU for spectral model evaluation.

    The resulting log-likelihood values are transferred back
    to NumPy for compatibility with the ``emcee`` sampler.

    No changes are introduced to the underlying Bayesian
    model or MCMC sampling algorithm.

    CUDA acceleration is optional and the original CPU
    posterior evaluation remains available.

    Examples
    --------
    >>> logp = lnprob_gauss_Lin_gpu(thetas, *data)
    >>> logp.shape
    (nwalkers_batch,)
    """
    import cupy as cp

    thetas = np.asarray(thetas)

    if thetas.ndim != 2:
        raise ValueError(
            "thetas must have shape (nwalkers_batch, ndim)"
        )

    nwalkers_batch = thetas.shape[0]

    # Evaluate the original priors on CPU.
    log_prior = np.asarray(
        [
            lnprior_gauss_Lin(
                theta,
                Infvalues,
                Supvalues,
                valsp,
                skew=skew,
                outflow=outflow,
                powlaw=powlaw,
                feii=feii,
            )
            for theta in thetas
        ],
        dtype=np.float64,
    )

    # Preserve the original rejection criterion.
    valid = np.isfinite(log_prior)

    log_prob = np.full(
        nwalkers_batch,
        -np.inf,
        dtype=np.float64,
    )

    # Avoid GPU execution if every walker is rejected.
    if not np.any(valid):
        return log_prob

    # Transfer only valid parameter vectors.
    valid_thetas = cp.asarray(thetas[valid])

    # Evaluate their likelihoods simultaneously on CUDA.
    log_likelihood = lnlike_gauss_Lin_gpu(
        valid_thetas,
        spec, specE, x,
        waves0, fac0, facN0,
        velfac0, velfacN0,
        fwhfac0, fwhfacN0,
        names0, n_lines, vals,
        skew, voigt, lorentz,
        outflow, powlaw, feii, data,
    )

    # Transfer the likelihood values back to CPU.
    log_prob[valid] = (
        log_prior[valid]
        + cp.asnumpy(log_likelihood)
    )
    return log_prob



def lnprob_gauss_Lin_batch(thetas, *args):
    """
    Evaluate the log-posterior probability for multiple parameter sets.

    This function extends the original spectral model posterior
    evaluation to a batch of parameter vectors. Each parameter
    set is evaluated independently using the existing
    ``lnprob_gauss_Lin`` function.

    The original prior constraints, Gaussian likelihood, and
    spectral model evaluation are preserved.

    This implementation provides a CPU reference for validating
    future vectorized CUDA posterior evaluations.

    Parameters
    ----------
    thetas : array_like
        Model parameter vectors with shape (nwalkers_batch, ndim).

        Each row contains the parameters associated with one
        MCMC walker.

    *args : tuple
        Additional arguments passed directly to
        ``lnprob_gauss_Lin``.

        These include the observed spectrum, uncertainties,
        spectral axis, parameter boundaries, model configuration,
        and optional spectral components.

    Returns
    -------
    log_prob : ndarray
        Log-posterior probability for each parameter vector.

        The returned array has shape (nwalkers_batch,).

        Parameter sets violating the original prior constraints
        return negative infinity.

    Notes
    -----
    The posterior probability is evaluated using the original
    MapLine prior and likelihood functions.

    This function does not modify the Bayesian model, prior
    distributions, or MCMC sampling algorithm.

    Evaluations are currently performed sequentially on CPU.
    GPU acceleration will require a vectorized implementation
    of the corresponding spectral models.

    This function can be used as a numerical reference for
    future CUDA implementations.

    Examples
    --------
    >>> logp = lnprob_gauss_Lin_batch(thetas, *data)
    >>> logp.shape
    (nwalkers_batch,)
    """
    thetas = np.asarray(thetas)

    if thetas.ndim != 2:
        raise ValueError(
            "thetas must have shape (nwalkers_batch, ndim)"
        )

    return np.asarray(
        [
            lnprob_gauss_Lin(theta, *args)
            for theta in thetas
        ],
        dtype=np.float64,
    )
