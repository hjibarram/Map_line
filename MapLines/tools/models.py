#!/usr/bin/env python
"""
MapLines.tools.models
=====================

Spectral line models used by MapLines.

This module defines the spectral models used to fit emission lines in
astronomical spectra. The models are used by the MCMC sampler to evaluate
the likelihood of the observed data given a set of model parameters.

Two main levels of modeling are implemented:

1. emission_line_model
   Computes individual emission-line profiles (Gaussian, Lorentzian,
   Voigt, or skewed Gaussian).

2. line_model
   Builds the full spectral model combining multiple emission lines,
   optional outflow components, continuum power-law emission, and
   FeII templates.

These models are used in combination with the fitting routines in
``MapLines.tools.line_fit`` and the likelihood functions defined in
``MapLines.tools.priors``.
"""
import MapLines.tools.tools as tol
import numpy as np 

def emission_line_model(x, xo=[5100], A=[1.0], dv=[0.0], fwhm=[200.0], alph=[0.0], gam=[0.0], skew=False, lorentz=False, voigt=False):
    """
    Compute emission-line profiles.

    This function generates emission-line profiles for a set of spectral
    lines using different profile shapes (Gaussian, skewed Gaussian,
    Lorentzian, or Voigt).

    Parameters
    ----------
    x : array-like
        Wavelength grid where the model is evaluated.
    xo : list of float
        Rest-frame wavelengths of the emission lines.
    A : list of float
        Amplitudes of each emission-line component.
    dv : list of float
        Velocity shifts relative to the rest-frame wavelength (km/s).
    fwhm : list of float
        Full width at half maximum (km/s) of each line.
    alph : list of float
        Skewness parameter (used when ``skew=True``).
    gam : list of float
        Lorentzian width parameter for Voigt profiles.
    skew : bool
        Use skewed Gaussian profiles.
    lorentz : bool
        Use Lorentzian profiles.
    voigt : bool
        Use Voigt profiles.

    Returns
    -------
    list of ndarray
        List containing the model profile of each emission line.

    Notes
    -----
    The line width is converted from velocity space (km/s) to wavelength
    dispersion using:

        sigma = FWHM / c * lambda / (2 * sqrt(2 ln 2))

    where ``c`` is the speed of light.
    """
    ct=299792.458
    model_out=[]
    for i in range(len(dv)):
        sigma=fwhm[i]/ct*xo[i]/(2.0*np.sqrt(2.0*np.log(2.0)))
        xm=xo[i]*(1.0+dv[i]/ct)
        #if i > 0:
        #    A1=A/fac[i-1]
        #else:
        A1=A[i]
        if skew:
            alp=alph[i]
            model=tol.gauss_K(x,sigma=sigma,xo=xm,A1=A1,alp=alp)
        else:
            if lorentz:
                model=tol.lorentz(x,sigma=(sigma*(2.0*np.sqrt(2.0*np.log(2.0)))),xo=xm,A1=A1)
            elif voigt:
                if len(gam) > 0:
                    gam1=gam[i]
                    model=tol.voigt(x,sigma=sigma,xo=xm,A1=A1,gam1=gam1)
                else:
                    model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
            else:
                model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
        model_out.extend([model])
    #if len(model_out) == 1:
    #    return model_out[0]
    #else:
    return model_out
        

def line_model(theta, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, 
    names0, n_lines, vals, x=0, powlaw=False, feii=False, data=None, ret_com=False, 
    skew=False, lorentz=False, outflow=False, voigt=False):
    """
    Construct the full spectral model for an emission-line complex.

    This function combines multiple emission-line components into a
    single spectral model. The parameters are provided as a vector
    (``theta``) sampled by the MCMC fitting procedure.

    The model may include:

    - Narrow emission lines
    - Optional outflow components
    - Power-law continuum
    - FeII emission templates

    Parameters
    ----------
    theta : array-like
        Parameter vector defining amplitudes, velocities, widths,
        and additional model parameters.
    waves0 : list
        Rest-frame wavelengths of emission lines.
    fac0, facN0 : list
        Scaling relations between line amplitudes.
    velfac0, velfacN0 : list
        Relations between velocity parameters.
    fwhfac0, fwhfacN0 : list
        Relations between line widths.
    names0 : list
        Names of emission lines.
    n_lines : int
        Number of emission lines in the model.
    vals : list
        Names of free parameters.
    x : ndarray
        Wavelength grid where the model is evaluated.

    Optional Components
    -------------------
    powlaw : bool
        Include a power-law continuum component.
    feii : bool
        Include FeII emission templates.
    outflow : bool
        Include outflow components for emission lines.
    skew : bool
        Use skewed Gaussian profiles.
    lorentz : bool
        Use Lorentzian profiles.
    voigt : bool
        Use Voigt profiles.

    Returns
    -------
    ndarray or list
        If ``ret_com=False``, returns the full model spectrum.

        If ``ret_com=True``, returns a list containing:
        - total model spectrum
        - individual emission-line components
        - optional continuum and FeII components

    Notes
    -----
    This function is the core spectral model evaluated during the MCMC
    sampling performed in ``MapLines.tools.mcmc``.
    """

    alph=[]
    alphb=[]
    gam=[]
    if skew:
        *f_parm,alp1,alpb=theta 
    else:
        if outflow:
            *f_parm,F1o,dvo,fwhmo,alpho=theta
            A1o=[]
            dvO=[]
            fwhmO=[]
            alphO=[]
        else:
            if voigt:
                *f_parm,gam1=theta
            else:
                f_parm=theta
    if powlaw:
        if feii:
            *f_parm,P0,Pa0,Fes,Fde,FA=theta
        else:
            *f_parm,P0,Pa0=theta
    else:
        if feii:
            *f_parm,Fes,Fde,FA=theta


    A1=[]
    dv=[]
    fwhm=[]
    for myt in range(0,n_lines):            
        inNaM=facN0[myt]
        velinNaM=velfacN0[myt]
        fwhinNaM=fwhfacN0[myt]
        valname='None'
        velvalname='None'
        fwhvalname='None'
        indf=-1
        velindf=-1
        fwhindf=-1
        vt1='AoN'.replace('N',str(myt))
        vt2='dvoN'.replace('N',str(myt))
        vt3='fwhmoN'.replace('N',str(myt))
        for atp in range(0, len(names0)):
            if names0[atp] == inNaM:
                valname='AoN'.replace('N',str(atp))
            if names0[atp] == velinNaM:
                velvalname='dvoN'.replace('N',str(atp))        
            if names0[atp] == fwhinNaM:
                fwhvalname='fwhmoN'.replace('N',str(atp))       
        for atp in range(0, len(vals)):
            if vals[atp] == valname:
                indf=atp
            if vals[atp] == velvalname:
                velindf=atp     
            if vals[atp] == fwhvalname:
                fwhindf=atp     
        if indf >= 0:
            A1.extend([f_parm[indf]/fac0[myt]])
            if outflow:
                A1o.extend([f_parm[indf]/fac0[myt]*F1o])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt1:
                    indfT1=atp
            A1.extend([f_parm[indfT1]])
            if outflow:
                A1o.extend([f_parm[indfT1]*F1o])
        if velindf >= 0:
            dv.extend([f_parm[velindf]*velfac0[myt]])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt2:
                    indfT2=atp  
            dv.extend([f_parm[indfT2]])              
        if fwhindf >= 0:
            fwhm.extend([f_parm[fwhindf]*fwhfac0[myt]])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt3:
                    indfT3=atp   
            fwhm.extend([f_parm[indfT3]])                 
        if skew:
            alph.extend([alp1])
            alphb.extend([alpb])
        else:
            if outflow:
                dvO.extend([dvo])
                fwhmO.extend([fwhmo])
                alphO.extend([alpho])
            else:
                if voigt:
                    gam.extend([gam1])


    ModA=emission_line_model(x, xo=waves0, A=A1, dv=dv ,fwhm=fwhm, alph=alph, gam=gam, skew=skew, lorentz=lorentz, voigt=voigt)
    if outflow:
        ModAo=emission_line_model(x, xo=waves0, A=A1o, dv=dvO ,fwhm=fwhmO, alph=alphO, skew=True)
        
    if powlaw:
        cont=tol.spow_law(x, A=P0, alpha=Pa0, xo=5100.0)
    else:
        cont=x*0.0
    if feii:
        feiis=tol.opticFeII(x, data ,sigma=Fes, xo=Fde, A1=FA)
    else:
        feiis=x*0.0

    lin=0
    for i in range(len(ModA)):
        if outflow:
            lin=ModA[i]+ModAo[i]+lin
        else:
            lin=ModA[i]+lin
    lin=lin+cont+feiis
    outvect=[]
    outvect.extend([lin])
    for i in range(len(ModA)):
        outvect.extend([ModA[i]])
    if outflow:
        for i in range(len(ModAo)):
            outvect.extend([ModAo[i]])       
    if powlaw:
        outvect.extend([cont])
    if feii:
        outvect.extend([feiis])    
    if ret_com:
        return outvect
    else:
        return lin