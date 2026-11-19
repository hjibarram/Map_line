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
- configurable numerical execution backends (CPU and CUDA)
- optional GPU acceleration of vectorized posterior probability evaluations
- automatic CPU fallback when CUDA is unavailable or unsupported

These routines are used by the fitting functions in
``MapLines.tools.line_fit`` and rely on the spectral models defined in
``MapLines.tools.models``.

The MCMC sampler performs Bayesian parameter estimation of spectral
components, allowing the exploration of parameter distributions and
the characterization of their associated uncertainties. Posterior
samples can subsequently be used to reconstruct representative
spectral models and estimate the dispersion of model realizations.

Notes
-----
The sampler is designed for flexible use with the posterior probability
functions defined in ``MapLines.tools.priors`` and supports both serial
and multiprocessing execution modes.

CUDA acceleration is optionally supported through the numerical backend
defined in ``MapLines.tools.backend``. When a compatible GPU and a
vectorized log-probability function are available, posterior probability
evaluations can be performed using CuPy while retaining the ``emcee``
ensemble sampler.

The CPU implementation remains the default reference for scientific
validation and compatibility with existing MapLine workflows. CUDA
support does not modify the underlying MCMC sampling algorithm or the
structure of the returned chains.

The extraction of posterior spectral model realizations is currently
performed using the standard CPU implementation.
"""
import numpy as np
import emcee
import MapLines.tools.models as mod


from MapLines.tools.backend import (
    get_backend,
    to_numpy,
)

def mcmc(p0,nwalkers,niter,ndim,lnprob,data,verbose=False,multi=True,tim=False,ncpu=10,
         backend="auto",lnprob_gpu=None,nburn=1000):
    """
    Run an MCMC sampler using the ``emcee`` ensemble sampler.

    This function performs parameter estimation for spectral models using
    a Markov Chain Monte Carlo (MCMC) approach. It supports both serial
    and multiprocessing execution.

    The sampler performs an initial burn-in phase followed by a production
    run, allowing the exploration of posterior probability distributions
    and the estimation of uncertainties in the fitted model parameters.

    In addition to standard CPU execution, the function optionally supports
    GPU-accelerated evaluation of vectorized log-probability functions using
    CUDA and CuPy. The ``emcee`` ensemble sampler continues to manage the
    walkers and their sampling process, while the evaluation of spectral
    models and posterior probabilities can be delegated to the GPU.

    Parameters
    ----------
    p0 : array_like
        Initial positions of the walkers in parameter space.
        Expected shape is (nwalkers, ndim).

    nwalkers : int
        Number of walkers used by the sampler.

    niter : int
        Number of iterations in the production run.

    ndim : int
        Number of model parameters.

    lnprob : callable
        Log-probability function used by the sampler.
        This function evaluates the posterior log-probability for
        an individual parameter vector during CPU execution.

    data : tuple
        Additional arguments passed to ``lnprob``.
        These arguments are also passed to ``lnprob_gpu`` when
        CUDA execution is enabled.

    verbose : bool, optional
        Print progress messages, including burn-in and production
        phases and the selected execution backend.
        Default is False.

    multi : bool, optional
        Enable multiprocessing during CPU execution.
        When CUDA is selected, multiprocessing is disabled in favor
        of vectorized GPU evaluations.
        Default is True.

    tim : bool, optional
        Measure execution time, including both burn-in and production
        phases.
        Default is False.

    ncpu : int, optional
        Number of CPUs used when multiprocessing.
        If the requested number exceeds the available CPU count,
        the number of processes is limited accordingly.
        A value of zero uses the default multiprocessing pool size.
        Default is 10.

    backend : {'auto', 'cpu', 'cuda'}, optional
        Numerical execution backend.

        - 'auto': Automatically select CUDA when available and a
          compatible GPU log-probability function is provided.
          Otherwise, use the standard CPU implementation.
        - 'cpu': Force CPU execution using the original serial
          or multiprocessing implementation.
        - 'cuda': Require CUDA execution and a compatible
          vectorized log-probability function.

        Default is 'auto'.

    lnprob_gpu : callable or None, optional
        Vectorized log-probability function for CUDA execution.

        Unlike ``lnprob``, this function must accept a two-dimensional
        array of parameter vectors with shape (nwalkers_batch, ndim)
        and return one log-probability value for each parameter vector.

        The number of walkers in each batch is determined by ``emcee``
        and may differ from the total number of walkers.

        The function should perform its numerical calculations on
        the GPU using CuPy. The resulting log-probabilities are
        transferred to NumPy before being returned to ``emcee``.

        Default is None.

    nburn : int, optional
        Number of burn-in iterations performed before the production
        run. Burn-in samples are discarded by resetting the sampler
        before production.

        Setting `nburn=0` skips the burn-in phase.

        Default is 1000.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        MCMC sampler object.
        Contains the production chain and associated sampling
        information.

    pos : ndarray
        Final walker positions.
        Shape is (nwalkers, ndim).

    prob : ndarray
        Log-probability values.
        Contains the final posterior log-probability for each walker.

    state : object
        Random state returned by ``emcee`` at the end of the
        production run, preserving the original MapLine
        return interface.

    Raises
    ------
    ValueError
        If an invalid execution backend is requested or the number
        of burn-in iterations is negative.

    RuntimeError
        If CUDA is explicitly requested but the CUDA runtime or a
        compatible GPU is unavailable.

    NotImplementedError
        If CUDA execution is explicitly requested without providing
        a GPU-compatible log-probability function.

    Notes
    -----
    This function performs a burn-in phase before the production run,
    unless `nburn=0` is specified.

    During burn-in, the walkers are allowed to explore the parameter
    space and approach regions of high posterior probability. After
    burn-in, the sampler is reset so that only production samples
    are retained in the final chain.

    The function supports two standard CPU execution modes:

    - Serial execution, where log-probability evaluations are
      performed sequentially.
    - Multiprocessing execution, where log-probability evaluations
      are distributed across multiple CPU processes using
      ``multiprocessing.Pool``.

    When CUDA execution is enabled, the sampler uses the vectorized
    evaluation interface of ``emcee``. The MCMC sampling algorithm
    remains unchanged, while batches of log-probability evaluations
    can be computed simultaneously on the GPU.

    In automatic mode, CPU execution is selected when CUDA is
    unavailable or when ``lnprob_gpu`` has not been provided.
    If CUDA execution is explicitly requested, an exception is
    raised instead of silently falling back to CPU.

    Errors occurring during the execution of an already selected
    GPU log-probability function are propagated to the caller.

    The function preserves the original return structure
    ``(sampler, pos, prob, state)`` to maintain compatibility
    with the existing MapLine fitting routines.

    Examples
    --------
    Run the sampler using CPU multiprocessing:

    >>> sampler, pos, prob, state = mcmc(
    ...     p0, nwalkers, niter, ndim,
    ...     lnprob, data,
    ...     multi=True,
    ...     ncpu=10,
    ...     backend="cpu"
    ... )

    Run the sampler in serial CPU mode:

    >>> sampler, pos, prob, state = mcmc(
    ...     p0, nwalkers, niter, ndim,
    ...     lnprob, data,
    ...     multi=False,
    ...     backend="cpu"
    ... )

    Automatically select CPU or CUDA:

    >>> sampler, pos, prob, state = mcmc(
    ...     p0, nwalkers, niter, ndim,
    ...     lnprob, data,
    ...     backend="auto",
    ...     lnprob_gpu=lnprob_vectorized
    ... )
    """
    if nburn < 0:
        raise ValueError("nburn must be non-negative")
    if tim:
        import time

    # --------------------------------------------------
    # CUDA selection (new)
    # --------------------------------------------------
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid backend")

    if backend == "cuda" and lnprob_gpu is None:
        raise NotImplementedError(
            "CUDA requires a vectorized log-probability function"
        )

    use_cuda = False

    if backend != "cpu" and lnprob_gpu is not None:
        from MapLines.tools.backend import get_backend
        _, use_cuda = get_backend(backend)

    # --------------------------------------------------
    # CUDA execution (new)
    # --------------------------------------------------
    if use_cuda:

        from MapLines.tools.backend import to_numpy

        def log_prob_cuda(theta, *args):
            result = lnprob_gpu(theta, *args)
            return to_numpy(result)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_cuda,
            args=data, vectorize=True)

        if tim:
            start = time.time()

        if verbose:
            print("Running CUDA burn-in...")

        p0, _, _ = sampler.run_mcmc(p0, nburn)
        sampler.reset()

        if verbose:
            print("Running CUDA production...")

        pos, prob, state = sampler.run_mcmc(p0, niter)

        if tim:
            end = time.time()
            cuda_time = end - start
            print(
                "CUDA took {0:.1f} seconds".format(cuda_time)
            )

    # --------------------------------------------------
    # Original CPU multiprocessing
    # --------------------------------------------------
    elif multi:
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        ncput = cpu_count()
        if ncpu > ncput:
            ncpu = ncput
        if ncpu == 0:
            ncpu = None
        with Pool(ncpu) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
                args=data, pool=pool)
            if tim:
                start = time.time()
            if verbose:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nburn)
            sampler.reset()
            if verbose:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
            if tim:
                end = time.time()
                multi_time = end - start
                print(
                    "Multiprocessing took {0:.1f} seconds".format(
                        multi_time))
    # --------------------------------------------------
    # Original CPU serial execution
    # --------------------------------------------------
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob, args=data)
        if tim:
            start = time.time()
        if verbose:
            print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        if verbose:
            print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        if tim:
            end = time.time()
            serial_time = end - start
            print(
                "Serial took {0:.1f} seconds".format(serial_time))
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