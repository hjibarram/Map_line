#!/usr/bin/env python
"""
MapLines.tools.mcmc
===================

MCMC utilities for spectral fitting in MapLines.

This module provides helper routines to run Markov Chain Monte Carlo
sampling and to evaluate posterior model realizations from the resulting
chains.

The main functionality includes:

- execution of the ``emcee`` ensemble sampler
- optional multiprocessing support
- burn-in and production runs
- extraction of representative model realizations from posterior samples

These routines are used by the fitting functions in
``MapLines.tools.line_fit`` and rely on the spectral models defined in
``MapLines.tools.models``.

Notes
-----
The sampler is designed for flexible use with the posterior probability
functions defined in ``MapLines.tools.priors`` and supports both serial
and multiprocessing execution modes.
"""
import numpy as np
import emcee
import MapLines.tools.models as mod

def mcmc(p0,nwalkers,niter,ndim,lnprob,data,verbose=False,multi=True,tim=False,ncpu=10):
    """
    Run an MCMC sampler using the ``emcee`` ensemble sampler.

    This function performs parameter estimation for spectral models using
    a Markov Chain Monte Carlo (MCMC) approach. It supports both serial
    and multiprocessing execution.

    Parameters
    ----------
    p0 : array_like
        Initial positions of the walkers in parameter space.
    nwalkers : int
        Number of walkers used by the sampler.
    niter : int
        Number of iterations in the production run.
    ndim : int
        Number of model parameters.
    lnprob : callable
        Log-probability function used by the sampler.
    data : tuple
        Additional arguments passed to ``lnprob``.
    verbose : bool, optional
        Print progress messages.
    multi : bool, optional
        Enable multiprocessing.
    tim : bool, optional
        Measure execution time.
    ncpu : int, optional
        Number of CPUs used when multiprocessing.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        MCMC sampler object.
    pos : ndarray
        Final walker positions.
    prob : ndarray
        Log-probability values.
    state : object
        Internal sampler state.

    Notes
    -----
    This function performs a burn-in phase before the production run.
    """
    if tim:
        import time
    if multi:
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        ncput=cpu_count()
        if ncpu > ncput:
            ncpu=ncput
        if ncpu == 0:
            ncpu=None
        with Pool(ncpu) as pool:
        #with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,pool=pool)
            if tim:
                start = time.time()
            if verbose:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 1000)
            sampler.reset()
            if verbose:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
            if tim:
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        if tim:
            start = time.time()
        if verbose:
            print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        if verbose:
            print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        if tim:
            end = time.time()
            serial_time = end - start
            print("Serial took {0:.1f} seconds".format(serial_time))
    return sampler, pos, prob, state


def sample_walkers(nsamples,flattened_chain,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,
    names0,n_lines,vals,x=0,skew=False,lorentz=False,outflow=False,powlaw=False,feii=False, data=None):
    """
    Generate model realizations from the posterior MCMC samples.

    This function randomly samples walkers from the flattened MCMC chain
    and computes the corresponding spectral models.

    Parameters
    ----------
    nsamples : int
        Number of models to draw from the chain.
    flattened_chain : ndarray
        Flattened MCMC chain.
    waves0 : array_like
        Rest wavelengths of spectral lines.
    fac0, facN0 : array_like
        Flux normalization factors.
    velfac0, velfacN0 : array_like
        Velocity scaling factors.
    fwhfac0, fwhfacN0 : array_like
        Line-width scaling factors.
    names0 : list
        Names of the spectral components.
    n_lines : int
        Number of emission lines.
    vals : dict
        Additional model parameters.
    x : array_like, optional
        Spectral axis.
    skew, lorentz, outflow, powlaw, feii : bool
        Flags controlling model components.
    data : optional
        Additional data passed to the model.

    Returns
    -------
    med_model : ndarray
        Median model spectrum.
    spread : ndarray
        Standard deviation of the sampled models.

    Notes
    -----
    The function uses the spectral model defined in
    ``MapLines.tools.models``. :contentReference[oaicite:0]{index=0}
    """
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        modt = mod.line_model(i, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=x, skew=skew, lorentz=lorentz, outflow=outflow, powlaw=powlaw, feii=feii, data=data)
        models.append(modt)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread