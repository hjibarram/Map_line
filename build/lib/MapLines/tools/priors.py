#!/usr/bin/env python
import MapLines.tools.models as mod
import numpy as np

def lnlike_gauss_Lin_hb(theta, spec, specE , x, xo1, xo2, xo3):
    model=mod.line_model_hb(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike

def lnlike_gauss_Lin(theta, spec, specE , x, xo1, xo2, xo3):
    model=mod.line_model(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike

def lnlike_gauss_Lin_s(theta, spec, specE , x, xo1, xo2, xo3, lfac12):
    model=mod.line_model_s(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3, lfac12=lfac12)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike


def lnprior_gauss_Lin(theta,At=0.05,dv1t=200,sim=False):
    dv1i=-dv1t
    dv1s=0.0
    dv2i=0.0
    dv2s=50.0
    if sim:
        dv1i=-dv1t
        dv1s=dv1t
        dv2i=-dv1t
        dv2s=dv1t    
    dA=At*0.3#0.003
    if At-dA < 0:
        Am1=0.0
    else:
        Am1=At-dA
    Am2=At+dA
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    
    if ((A1 >= 0) and (A1 <=0.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((fac >= 0.0) and (fac <=10.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 5500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv2 >= dv2i) and (dv2 <= dv2s))  and ((dv3 >= -600) and (dv3 <= 600)): 
        return 0.0
    else:
        return -np.inf
    
def lnprior_gauss_Lin_s(theta,At=0.05,dv1t=200,sim=False):
    dv1i=-dv1t
    dv1s=0.0
    if sim:
        dv1i=-dv1t
        dv1s=dv1t
    dA=At*0.3#0.003
    if At-dA < 0:
        Am1=0.0
    else:
        Am1=At-dA#decomentar
    Am2=At+dA
    A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
    
    if ((A1 >= 0) and (A1 <=0.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 5500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv3 >= -600) and (dv3 <= 600)): 
        return 0.0
    else:
        return -np.inf    

def lnprob_gauss_Lin_Hb(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t, sim):
    lp = lnprior_gauss_Lin(theta,At=At, dv1t=dv1t,sim=sim)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin_hb(theta, spec, specE, x, xo1, xo2, xo3) 

def lnprob_gauss_Lin(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t, sim):
    lp = lnprior_gauss_Lin(theta,At=At, dv1t=dv1t,sim=sim)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin(theta, spec, specE, x, xo1, xo2, xo3) 

def lnprob_gauss_Lin_s(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t, sim, lfac12):
    lp = lnprior_gauss_Lin_s(theta,At=At, dv1t=dv1t, sim=sim)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin_s(theta, spec, specE, x, xo1, xo2, xo3, lfac12) 


