#!/usr/bin/env python
import MapLines.tools.tools as tol
import numpy as np

def emission_line_model(x, xo=0, A=1.0, dv=[0.0], fwhm=[200.0], fac=[0.7]):
    ct=299792.458
    model_out=[]
    for i in range(len(dv)):
        sigma=fwhm[i]/ct*xo/(2.0*np.sqrt(2.0*np.log(2.0)))
        xm=xo*(1.0+dv[i]/ct)
        if i > 0:
            A=A/fac[i-1]
        model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
        model_out.extend([model])
    return model_out
        

def line_model(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False, lfac12=2.93):
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    
    Gb,Gr=emission_line_model(x, xo=xo1, A=A1, dv=[dv1,dv2], fwhm=[fwhm1,fwhm1], fac=[fac])

    #sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    #xo_b=xo1*(1.0+dv1/ct)
    #Gb=tol.gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    #xo_r=xo1*(1.0+dv2/ct)
    #A2=A1/fac
    #Gr=tol.gauss_M(x,sigma=sigma1,xo=xo_r,A1=A2)
    

    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    hGb=tol.gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    xo_r=xo2*(1.0+dv2/ct)
    A4=A3/fac
    hGr=tol.gauss_M(x,sigma=sigma2,xo=xo_r,A1=A4)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    A5=A1/lfac12
    nGb=tol.gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    xo_r=xo3*(1.0+dv2/ct)
    A6=A2/lfac12
    nGr=tol.gauss_M(x,sigma=sigma3,xo=xo_r,A1=A6)
    
    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=tol.gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+Gr+hGb+hGr+nGb+nGr+hGbr
    if ret_com:
        return lin,Gb,Gr,hGb,hGr,nGb,nGr,hGbr
    else:
        return lin


def line_model_s(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False, lfac12=2.93):
    A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    Gb=tol.gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    hGb=tol.gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    A5=A1/lfac12
    nGb=tol.gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    
    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=tol.gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+hGb+nGb+hGbr
    if ret_com:
        return lin,Gb,hGb,nGb,hGbr
    else:
        return lin