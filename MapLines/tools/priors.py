#!/usr/bin/env python
import MapLines.tools.models as mod
import numpy as np
'''This module contains the functions to calculate the likelihood and prior of the models'''


def lnlike_gauss_Lin(theta, spec, specE , x, xo1, xo2, xo3, lfac12, single, skew, broad, lorentz, n_line):
    '''This function calculates the likelihood of a double model for the spectrum'''
    model=mod.line_model(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz, n_line=n_line)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike


def lnprior_gauss_Lin(theta, valsp, At=0.05,dv1t=200,sim=False, single=False, skew=False, broad=True, n_line=False):
    '''This function calculates the prior of a line model spectra'''
    '''
    a1s=1.8
    a1i=0
    a3s=0.8
    a3i=0
    fwhm1s=800.0
    fwhm1i=100.0
    fwhm2s=10500.0
    fwhm2i=700.0
    dv1s=200.0
    dv1i=-200.0
    dv3s=1000.0
    dv3i=-1000.0
    '''
    #dv1i=-dv1t
    #dv1s=0.0
    #dv2i=0.0
    #dv2s=50.0
    #if sim:
    #    dv1i=-dv1t
    #    dv1s=dv1t
    #    dv2i=-dv1t
    #    dv2s=dv1t    
    dA=At*0.3#0.3
    if At-dA*3 < 0:
        Am1=0.0
    else:
        Am1=At-dA*3
    #Am1=0.0
    Am2=At+dA
    if single:
        if skew:
            A1,A3,dv1,fwhm1,fwhm2,A7,dv3,alp1,alpb=theta
            #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 900.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 10500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv3 >= -1000) and (dv3 <= 1000)) and ((alp1 >= -10) and (alp1 <= 10)) and ((alpb >= -10) and (alpb <= 10)): 
            if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((fwhm2 >= valsp['fwhm2i']) and (fwhm2 <= valsp['fwhm2s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv3 >= valsp['dv3i']) and (dv3 <= valsp['dv3s'])) and ((alp1 >= -10) and (alp1 <= 10)) and ((alpb >= -10) and (alpb <= 10)):     
                return 0.0
            else:
                return -np.inf    
        else:
            if broad:
                A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
                #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 10500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv3 >= -1000) and (dv3 <= 1000)): 
                if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((fwhm2 >= valsp['fwhm2i']) and (fwhm2 <= valsp['fwhm2s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv3 >= valsp['dv3i']) and (dv3 <= valsp['dv3s'])): 
                    return 0.0
                else:
                    return -np.inf
            else:
                if n_line:
                    A1,dv1,fwhm1=theta
                    if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])): 
                        return 0.0
                    else:
                        return -np.inf
                else:
                    A1,A3,dv1,fwhm1=theta
                    #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)): 
                    if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])): 
                        return 0.0
                    else:
                        return -np.inf    
    else:
        if skew:
            A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3,alp1,alpb=theta
            #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((fac >= 1.0) and (fac <=30.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 10500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv2 >= dv2i) and (dv2 <= dv2s))  and ((dv3 >= -1000) and (dv3 <= 1000)) and ((alp1 >= -10) and (alp1 <= 10)) and ((alpb >= -10) and (alpb <= 10)):
            if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((fac >= 1.0) and (fac <=30.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((fwhm2 >= valsp['fwhm2i']) and (fwhm2 <= valsp['fwhm2s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv2 >= valsp['dv1i']) and (dv2 <= valsp['dv1s']))  and ((dv3 >= valsp['dv3i']) and (dv3 <= valsp['dv3s'])) and ((alp1 >= -10) and (alp1 <= 10)) and ((alpb >= -10) and (alpb <= 10)): 
                return 0.0
            else:
                return -np.inf
        else:
            if broad:
                A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
                #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((fac >= 1.0) and (fac <=30.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 10500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv2 >= dv2i) and (dv2 <= dv2s))  and ((dv3 >= -1000) and (dv3 <= 1000)): 
                if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((fac >= 1.0) and (fac <=30.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((fwhm2 >= valsp['fwhm2i']) and (fwhm2 <= valsp['fwhm2s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv2 >= valsp['dv1i']) and (dv2 <= valsp['dv1s']))  and ((dv3 >= valsp['dv3i']) and (dv3 <= valsp['dv3s'])): 
                    return 0.0
                else:
                    return -np.inf
            else:
                if n_line:
                    A1,fac,dv1,dv2,fwhm1=theta
                    if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((fac >= 0) and (fac <=30.0)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv2 >= valsp['dv1i']) and (dv2 <= valsp['dv1s'])): 
                        return 0.0
                    else:
                        return -np.inf
                else:
                    A1,A3,fac,dv1,dv2,fwhm1=theta
                    #if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=1.8)) and ((fac >= 0) and (fac <=30.0)) and ((fwhm1 >= 10.0) and (fwhm1 <= 250.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv2 >= dv2i) and (dv2 <= dv2s)): 
                    if ((A1 >= valsp['a1i']) and (A1 <= valsp['a1s'])) and ((A3 >= valsp['a3i']) and (A3 <= valsp['a3s'])) and ((fac >= 0) and (fac <=30.0)) and ((fwhm1 >= valsp['fwhm1i']) and (fwhm1 <= valsp['fwhm1s'])) and ((dv1 >= valsp['dv1i']) and (dv1 <= valsp['dv1s'])) and ((dv2 >= valsp['dv1i']) and (dv2 <= valsp['dv1s'])): 
                        return 0.0
                    else:
                        return -np.inf 

def lnprob_gauss_Lin(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t, sim, lfac12, single, skew, broad, lorentz, valsp, n_line):
    '''This function calculates the posterior of the double model for the spectrum'''
    lp = lnprior_gauss_Lin(theta, valsp, At=At, dv1t=dv1t, sim=sim, single=single, skew=skew, broad=broad, n_line=n_line)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin(theta, spec, specE, x, xo1, xo2, xo3, lfac12, single, skew, broad, lorentz, n_line) 

    
#def lnlike_gauss_Lin_s(theta, spec, specE , x, xo1, xo2, xo3, lfac12, single):
#    '''This function calculates the likelihood of a single model for the spectrum'''
#    model=mod.line_model(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3, lfac12=lfac12, single=single)
#    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
#    return LnLike

#def lnprior_gauss_Lin_s(theta,At=0.05,dv1t=200,sim=False, single=False):
#    '''This function calculates the prior of a single model for the spectrum'''
#    dv1i=-dv1t
#    dv1s=0.0
#    if sim:
#        dv1i=-dv1t
#        dv1s=dv1t
#    dA=At*0.3
#    if At-dA < 0:
#        Am1=0.0
#    else:
#        Am1=At-dA
#    Am2=At+dA
#    A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
#    if ((A1 >= 0) and (A1 <=1.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 800.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 10500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv3 >= -1000) and (dv3 <= 1000)): 
#        return 0.0
#    else:
#        return -np.inf


#def lnprob_gauss_Lin_s(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t, sim, lfac12, single):
#    '''This function calculates the posterior of the single model for the spectrum'''
#    lp = lnprior_gauss_Lin(theta,At=At, dv1t=dv1t, sim=sim, single=single)
#    if not np.isfinite(lp):
#        return -np.inf
#    else:
#        return lp + lnlike_gauss_Lin(theta, spec, specE, x, xo1, xo2, xo3, lfac12, single) 
