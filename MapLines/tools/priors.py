#!/usr/bin/env python
import MapLines.tools.models as mod
import numpy as np
'''This module contains the functions to calculate the likelihood and prior of the models'''


def lnlike_gauss_Lin(theta, spec, specE , x, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii):
    '''This function calculates the likelihood of a double model for the spectrum'''
    model=mod.line_model(theta, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=x, skew=skew, voigt=voigt, lorentz=lorentz, outflow=outflow, powlaw=powlaw, feii=feii)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike


def lnprior_gauss_Lin(theta, Infvalues, Supvalues, valsp, skew=False, outflow=False, powlaw=False, feii=False):
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

    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf    

    if boolf:
        return 0.0
    else:
        return -np.inf            
                

def lnprob_gauss_Lin(theta, spec, specE, x, Infvalues, Supvalues, valsp, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii):
    '''This function calculates the posterior of the double model for the spectrum'''
    lp = lnprior_gauss_Lin(theta, Infvalues, Supvalues, valsp, skew=skew, outflow=outflow, powlaw=powlaw, feii=feii)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin(theta, spec, specE, x, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii) 

