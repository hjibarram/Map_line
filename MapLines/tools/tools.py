#!/usr/bin/env python
import numpy as np
from scipy.ndimage import gaussian_filter1d as filt1d
import os
import os.path as ptt
from scipy.special import erf as errf


def wfits_ext(name,hlist):
    sycall("rm "+name+'.gz')
    if ptt.exists(name) == False:
        hlist.writeto(name)
    else:
        name1=name.replace("\ "," ")
        name1=name1.replace(" ","\ ")
        sycall("rm "+name1)
        hlist.writeto(name)

def sycall(comand):
    linp=comand
    os.system(comand)

def conv(xt,ke=2.5):
    nsf=len(xt)
    krn=ke
    xf=filt1d(xt,ke)
    return xf

def gauss_K(x,sigma=1.0,xo=0.0,A1=1.0,alp=0):
    dt=alp/np.sqrt(np.abs(1+alp**2))
    xot=xo-sigma*dt*np.sqrt(2/np.pi)
    omega=np.sqrt(sigma**2.0/(1-2*dt**2/np.pi))
    #omega=sigma
    t=(x-xot)/omega
    Phi=(1+errf(alp*t/np.sqrt(2.0)))
    Ghi=np.exp(-0.5*t**2.0)
    y=A1*Ghi*Phi
    return y

def gauss_M(x,sigma=1.0,xo=0.0,A1=1.0):
    y=A1*np.exp(-0.5*(x-xo)**2.0/sigma**2.0)
    return y

def step_vect(fluxi,sp=20,pst=True,sigma=10):
    flux_sm=conv(fluxi,ke=sigma)
    flux=fluxi-flux_sm
    nz=len(flux)
    flux_t=np.copy(flux)
    for i in range(0, nz):
        i0=int(i-sp/2.0)
        i1=int(i+sp/2.0)
        if i1 > nz:
            i1=nz
        if i0 > nz:
            i0=int(nz-sp)
        if i0 < 0:
            i0=0
        if i1 < 0:
            i1=sp   
        if pst:
            lts=np.nanpercentile(flux[i0:i1],78)
            lt0=np.nanpercentile(flux[i0:i1],50)
            lti=np.nanpercentile(flux[i0:i1],22)
            val=(np.abs(lts-lt0)+np.abs(lti-lt0))/2.0
            flux_t[i]=val#mean
        else:
            flux_t[i]=np.nanstd(flux[i0:i1])
    return flux_t