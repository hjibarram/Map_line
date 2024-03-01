#!/usr/bin/env python
import MapLines.tools.tools as tol
import numpy as np

'''
def line_model_hb(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False):
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    #sigmaO=fwhmO/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    xo_r=xo1*(1.0+dv2/ct)
    Gb=tol.gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    A2=A1/fac
    Gr=tol.gauss_M(x,sigma=sigma1,xo=xo_r,A1=A2)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    xo_r=xo2*(1.0+dv2/ct)
    hGb=tol.gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    A4=A3/fac
    hGr=tol.gauss_M(x,sigma=sigma2,xo=xo_r,A1=A4)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    xo_r=xo3*(1.0+dv2/ct)
    A5=A1/3.0
    A6=A2/3.0
    nGb=tol.gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    nGr=tol.gauss_M(x,sigma=sigma3,xo=xo_r,A1=A6)

    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=tol.gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+Gr+hGb+hGr+nGb+nGr+hGbr
    if ret_com:
        return lin,Gb,Gr,hGb,hGr,nGb,nGr,hGbr
    else:
        return lin
'''        

def line_model(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False, lfac12=2.93):
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    xo_r=xo1*(1.0+dv2/ct)
    Gb=tol.gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    A2=A1/fac
    Gr=tol.gauss_M(x,sigma=sigma1,xo=xo_r,A1=A2)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    xo_r=xo2*(1.0+dv2/ct)
    hGb=tol.gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    A4=A3/fac
    hGr=tol.gauss_M(x,sigma=sigma2,xo=xo_r,A1=A4)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    xo_r=xo3*(1.0+dv2/ct)
    A5=A1/lfac12
    A6=A2/lfac12
    nGb=tol.gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
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