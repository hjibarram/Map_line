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

These functions are used together with the spectral models defined in
``MapLines.tools.models`` and the MCMC driver implemented in
``MapLines.tools.mcmc``.

Notes
-----
The likelihood assumes Gaussian uncertainties in the observed spectrum,
while the prior enforces physically and empirically motivated parameter
ranges defined in the configuration files.
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

