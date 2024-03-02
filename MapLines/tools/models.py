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
            A1=A/fac[i-1]
        else:
            A1=A
        model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
        model_out.extend([model])
    if len(model_out) == 1:
        return model_out[0]
    else:
        return model_out
        

def line_model(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False, lfac12=2.93, single=False):
    '''Model for the line complex'''

    if single:
        A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
        dv=[dv1]
        dvb=[dv3]
        fwhm=[fwhm1]
        fwhmb=[fwhm2]
        fact=[]
    else:
        A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
        dv=[dv1,dv2]
        dvb=[dv3]
        fwhm=[fwhm1,fwhm1]
        fwhmb=[fwhm2]
        fact=[fac]
        
    
    A5=A1/lfac12
    ModA=emission_line_model(x, xo=xo1, A=A1, dv=dv ,fwhm=fwhm, fac=fact)
    ModH=emission_line_model(x, xo=xo2, A=A3, dv=dv, fwhm=fwhm, fac=fact)
    ModB=emission_line_model(x, xo=xo3, A=A5, dv=dv, fwhm=fwhm, fac=fact)
    ModHB=emission_line_model(x, xo=xo2, A=A7, dv=dvb, fwhm=fwhmb)
    
    lin=0
    if single:
        lin=ModA+ModH+ModB
    else:
        for i in range(len(ModA)):
            lin=ModA[i]+ModH[i]+ModB[i]+lin
    lin=lin+ModHB    
    outvect=[]
    outvect.extend([lin])
    if single:
        outvect.extend([ModA])
        outvect.extend([ModH])
        outvect.extend([ModB])
        outvect.extend([ModHB])
    else:
        for i in range(len(ModA)):
            outvect.extend([ModA[i]])
        for i in range(len(ModH)):
            outvect.extend([ModH[i]])
        for i in range(len(ModA)):
            outvect.extend([ModB[i]])
        outvect.extend([ModHB])            
    
    if ret_com:
        return outvect
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