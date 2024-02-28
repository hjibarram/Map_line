#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
import types
import astropy.cosmology.funcs as cd
import importlib.machinery
import os.path as ptt
from progressbar import ProgressBar
import math
from scipy.ndimage.filters import gaussian_filter1d as filt1d
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.wcs import WCS
from scipy.interpolate.interpolate import interp1d
warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib import colors
import emcee
from scipy.special import gamma, gammaincinv, gammainc

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
    import os
    linp=comand
    os.system(comand)

def conv(xt,ke=2.5):
    nsf=len(xt)
    krn=ke
    xf=filt1d(xt,ke)
    return xf

def gauss_M(x,sigma=1.0,xo=0.0,A1=1.0):
    y=A1*np.exp(-0.5*(x-xo)**2.0/sigma**2.0)
    return y

def line_model_hb(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False):
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    #sigmaO=fwhmO/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    xo_r=xo1*(1.0+dv2/ct)
    Gb=gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    A2=A1/fac
    Gr=gauss_M(x,sigma=sigma1,xo=xo_r,A1=A2)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    xo_r=xo2*(1.0+dv2/ct)
    hGb=gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    A4=A3/fac
    hGr=gauss_M(x,sigma=sigma2,xo=xo_r,A1=A4)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    xo_r=xo3*(1.0+dv2/ct)
    A5=A1/3.0
    A6=A2/3.0
    nGb=gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    nGr=gauss_M(x,sigma=sigma3,xo=xo_r,A1=A6)

    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+Gr+hGb+hGr+nGb+nGr+hGbr
    if ret_com:
        return lin,Gb,Gr,hGb,hGr,nGb,nGr,hGbr
    else:
        return lin

def line_model(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False):
    A1,A3,fac,dv1,dv2,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    xo_r=xo1*(1.0+dv2/ct)
    Gb=gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    A2=A1/fac
    Gr=gauss_M(x,sigma=sigma1,xo=xo_r,A1=A2)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    xo_r=xo2*(1.0+dv2/ct)
    hGb=gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    A4=A3/fac
    hGr=gauss_M(x,sigma=sigma2,xo=xo_r,A1=A4)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    xo_r=xo3*(1.0+dv2/ct)
    A5=A1/2.93
    A6=A2/2.93
    nGb=gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    nGr=gauss_M(x,sigma=sigma3,xo=xo_r,A1=A6)
    
    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+Gr+hGb+hGr+nGb+nGr+hGbr
    if ret_com:
        return lin,Gb,Gr,hGb,hGr,nGb,nGr,hGbr
    else:
        return lin

def line_model_s(theta, x=0, xo1=0, xo2=0, xo3=0 ,ret_com=False):
    A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
    ct=299792.458
    sigma1=fwhm1/ct*xo1/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo1*(1.0+dv1/ct)
    Gb=gauss_M(x,sigma=sigma1,xo=xo_b,A1=A1)
    
    sigma2=fwhm1/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo2*(1.0+dv1/ct)
    hGb=gauss_M(x,sigma=sigma2,xo=xo_b,A1=A3)
    
    sigma3=fwhm1/ct*xo3/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo_b=xo3*(1.0+dv1/ct)
    A5=A1/2.93
    nGb=gauss_M(x,sigma=sigma3,xo=xo_b,A1=A5)
    
    sigma4=fwhm2/ct*xo2/(2.0*np.sqrt(2.0*np.log(2.0)))
    xo=xo2*(1.0+dv3/ct)
    hGbr=gauss_M(x,sigma=sigma4,xo=xo,A1=A7)
    
    lin=Gb+hGb+nGb+hGbr
    if ret_com:
        return lin,Gb,hGb,nGb,hGbr
    else:
        return lin

def lnlike_gauss_Lin_hb(theta, spec, specE , x, xo1, xo2, xo3):
    model=line_model_hb(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike

def lnlike_gauss_Lin(theta, spec, specE , x, xo1, xo2, xo3):
    model=line_model(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike

def lnlike_gauss_Lin_s(theta, spec, specE , x, xo1, xo2, xo3):
    model=line_model_s(theta, x=x, xo1=xo1, xo2=xo2, xo3=xo3)
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
    #and ((A3 >= 0) and (A3 <=0.8)) and ((A4 >= 0) and (A4 <=0.8))#345.0,dv1=-50,dv2=30,fac=0.9,0.5
    if ((A1 >= 0) and (A1 <=0.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((fac >= 0.0) and (fac <=10.0)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 300.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 2500.0)) and ((dv1 >= dv1i) and (dv1 <= dv1s)) and ((dv2 >= dv2i) and (dv2 <= dv2s))  and ((dv3 >= -100) and (dv3 <= 100)): 
        return 0.0
    else:
        return -np.inf
    
def lnprior_gauss_Lin_s(theta,At=0.05,dv1t=200):
    dA=At*0.3#0.003
    if At-dA < 0:
        Am1=0.0
    else:
        Am1=At-dA#decomentar
    Am2=At+dA
    A1,A3,dv1,fwhm1,fwhm2,A7,dv3=theta
    #and ((A3 >= 0) and (A3 <=0.8)) and ((A4 >= 0) and (A4 <=0.8))#345.0,dv1=-50,dv2=30,fac=0.9,0.5
    #if ((A1 >= 0) and (A1 <=0.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 500.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 2500.0)) and ((dv1 >= -dv1t) and (dv1 <= dv1t)) and ((dv3 >= -500) and (dv3 <= 500)): 
    if ((A1 >= 0) and (A1 <=0.8)) and ((A3 >= 0) and (A3 <=0.8)) and ((A7 >= Am1) and (A7 < Am2)) and ((fwhm1 >= 100.0) and (fwhm1 <= 500.0)) and ((fwhm2 >= 700.0) and (fwhm2 <= 2500.0)) and ((dv1 >= -dv1t) and (dv1 <= 0)) and ((dv3 >= -500) and (dv3 <= 500)): 
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

def lnprob_gauss_Lin_s(theta, spec, specE, x, xo1, xo2, xo3, At, dv1t):
    lp = lnprior_gauss_Lin_s(theta,At=At, dv1t=dv1t)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gauss_Lin_s(theta, spec, specE, x, xo1, xo2, xo3) 
    
def mcmc(p0,nwalkers,niter,ndim,lnprob,data,verbose=False,multi=True,tim=False):
    if tim:
        import time
    if multi:
        from multiprocessing import Pool
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,pool=pool)
            if tim:
                start = time.time()
            if verbose:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 1000)
            sampler.reset()
            if verbose:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
            if tim:
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        if tim:
            start = time.time()
        if verbose:
            print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        if verbose:
            print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        if tim:
            end = time.time()
            serial_time = end - start
            print("Serial took {0:.1f} seconds".format(serial_time))
    return sampler, pos, prob, state

def sample_walkers(nsamples,flattened_chain,x=0,xo1=0,xo2=0,xo3=0,single=False):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        if single:
            mod = line_model_s(i, x=x, xo1=xo1, xo2=xo2, xo3=xo3)#model(i)
        else:
            mod = line_model(i, x=x, xo1=xo1, xo2=xo2, xo3=xo3)#model(i)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread


def line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6450.0,lA2=6850.0,plot_f=True,pgr_bar=True,single=False,flux_f=1.0,erft=0.75,dv1t=200,sim=False,cont=False,hbfit=False):
    #lA1=6880.
    #lA2=6980.
    [pdl_cube, hdr]=fits.getdata(file1, 0, header=True)
    pdl_cubeE =fits.getdata(file2, 1, header=False)
    nz,nx,ny=pdl_cube.shape
    pdl_cube=pdl_cube*flux_f
    pdl_cubeE=pdl_cubeE*flux_f*erft
    if ptt.exists(file3):
        mask =fits.getdata(file3, 0, header=False)
        nxt,nyt=mask.shape
        if nxt != nx and nyt != ny:
            mask=np.zeros([nx,ny])
            mask[:,:]=1
    else:
        mask=np.zeros([nx,ny])
        mask[:,:]=1
    
    crpix=hdr["CRPIX3"]
    try:
        cdelt=hdr["CD3_3"]
    except:
        cdelt=hdr["CDELT3"]
    crval=hdr["CRVAL3"]
    wave=crval+cdelt*(np.arange(nz)+1-crpix)
    
    wave_f=wave/(1+z)
    
    nw=np.where((wave_f >= lA1) & (wave_f <= lA2))[0]
    wave_i=wave_f[nw]
    #m2B,m2R,mHaB,mHaR,m1B,m1R,mHaBR
    model_all=np.zeros([len(nw),nx,ny])
    model_Blue=np.zeros([len(nw),nx,ny])
    model_Red=np.zeros([len(nw),nx,ny])
    model_Broad=np.zeros([len(nw),nx,ny])
    if single:
        if cont:
            model_param=np.zeros([9,nx,ny])
        else:
            model_param=np.zeros([8,nx,ny])
    else:
        if cont:
            model_param=np.zeros([14,nx,ny])
        else:
            model_param=np.zeros([13,nx,ny])
    model_param[:,:,:]=np.nan    
    #A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f
    #i=31#25
    #j=31#27
    #i=25
    #j=27
    #i=33
    #j=26
    Loiii1=4960.36 
    LnrHb=4862.68#4853.13 
    Loiii2=5008.22
    Lnii2=6585.278
    LnrHa=6564.632
    Lnii1=6549.859
    hdr["CRVAL3"]=wave_i[0]
    try:
        hdr["CD3_3"]=cdelt
    except:
        hdr["CDELT3"]=cdelt/(1+z)
    if pgr_bar:
        progress2 = ProgressBar(maxval=nx).start()
    for i in range(0, nx):
        for j in range(0, ny):
            val=mask[i,j]
            #val=1
            #i=31#25
            #j=31#27
            #i=25
            #j=27
            #i=33
            #j=26
            #i=24
            #j=26
            #i=7
            #j=22
            #i=22
            #j=26
            #i=26
            #j=23
            #MUSE
            #i=55
            #j=54
            #i=57
            #j=45
            #i=45
            #j=47
            #i=24
            #j=24
            #J1027
            #i=26-1
            #j=22-1
            #i=21
            #j=21
            #i=19
            #j=16
            #i=26
            #j=23
            #i=21
            #j=19
            i=32
            j=26
            #i=32
            #j=36
            #i=29
            #j=18
            #i=22
            #j=16
            i=86
            j=73
            if val == 1:
                fluxt=pdl_cube[nw,i,j]
                fluxtE=pdl_cubeE[nw,i,j]
                if cont:
                    if hbfit:
                        nwt=np.where((wave_f[nw] >= 5065.0) & (wave_f[nw] <= 5085.0))[0]
                    else:
                        nwt=np.where((wave_f[nw] >= 6490.0) & (wave_f[nw] <= 6510.0))[0]  
                    fluxpt=np.nanmean(fluxt[nwt])  
                    fluxt=fluxt-fluxpt
                nwt=np.where((wave_f[nw] >= 6569.0) & (wave_f[nw] <= 6572.0))[0]
                fluxp=np.nanmean(fluxt[nwt])
                #print(fluxp)
                if fluxp < 0:
                    fluxp=0.0001
                fluxp=0.0001    
                if single:
                    data = (fluxt, fluxtE, wave_i, Lnii2, LnrHa, Lnii1, fluxp, dv1t)
                else:
                    if hbfit:
                        data = (fluxt, fluxtE, wave_i, Loiii2, LnrHb, Loiii1, fluxp, dv1t, sim)#dv0t, fwhm1t, fwhm2t)
                    else:
                        data = (fluxt, fluxtE, wave_i, Lnii2, LnrHa, Lnii1, fluxp, dv1t, sim)
                nwalkers=240
                niter=1024
                if single:
                    initial = np.array([0.04, 0.06, -20.0, 150.0, 1000.0, fluxp, 0.0])
                    #print(initial)
                else:
                    if hbfit:
                        #A1,A3,fac,dv0,dv1,fwhm1,fwhmO=theta#,fwhm2,fwhm3,A5,A7,dv3,dv2=theta
                        #A1,A3,fac,dv0,dv1,fwhm1,fwhmO,fwhm2,fwhm3,A5,A7,dv3=theta
                        initial = np.array([0.04, 0.09, 0.75, -200.0, 40.0, 150.0, 1000.0, fluxp, 0.0])#, fluxp, 0.0, 1000
                        #,A7,dv3,fwhm3
                        #1000.0, 6000.0, 0.1, fluxp, 0.0, 500
                    else:
                        #initial0 = np.array([0.04, 0.06, -20.0, 150.0, 1000.0, fluxp, 0.0])
                        #ndim0 = len(initial0)
                        #p00 = [np.array(initial0) + 1e-5 * np.random.randn(ndim0) for i in range(nwalkers)]
                        initial = np.array([0.04, 0.06, 0.75, -200.0, 40.0, 150.0, 1000.0, fluxp, 0.0])#A1,A2,dv1,dv2,fwhm1
                ndim = len(initial)
                p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
                if plot_f:
                    tim=True
                else:
                    tim=False
                if single:
                    sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_gauss_Lin_s,data,tim=tim)
                else:
                    if hbfit:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_gauss_Lin_Hb,data,tim=tim)
                    else:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_gauss_Lin,data,tim=tim)    
                samples = sampler.flatchain
                theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                if single:
                    A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                    model,m2B,mHaB,m1B,mHaBR=line_model_s(theta_max, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, ret_com=True)
                    model_all[:,i,j]=model
                    model_Blue[:,i,j]=m2B+m1B+mHaB
                    model_Broad[:,i,j]=mHaBR
                    model_param[0,i,j]=A1_f#*flux_f
                    model_param[1,i,j]=A1_f/2.93#*flux_f
                    model_param[2,i,j]=A3_f#*flux_f
                    model_param[3,i,j]=A7_f#*flux_f
                    model_param[4,i,j]=dv1_f
                    model_param[5,i,j]=dv3_f
                    model_param[6,i,j]=fwhm1_f
                    model_param[7,i,j]=fwhm2_f
                    if cont:
                        model_param[8,i,j]=fluxpt
                else:
                    if hbfit:
                        A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                        if dv2_f < dv1_f:
                            fac_f=1/fac_f
                            dt=np.copy(dv2_f)
                            dv2_f=np.copy(dv1_f)
                            dv1_f=dt
                            A1_f=A1_f*fac_f
                            A3_f=A3_f*fac_f
                            theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f
                        model,m2B,m2R,mHbB,mHbR,m1B,m1R,mHbBR=line_model_hb(theta_max, x=wave_i, xo1=Loiii2, xo2=LnrHb, xo3=Loiii1, ret_com=True)
                        model_all[:,i,j]=model
                        model_Blue[:,i,j]=m2B+m1B+mHbB
                        model_Red[:,i,j]=m2R+m1R+mHbR
                        model_Broad[:,i,j]=mHbBR
                        model_param[0,i,j]=A1_f#*flux_f
                        model_param[1,i,j]=A1_f/3.0#*flux_f
                        model_param[2,i,j]=A3_f#*flux_f
                        model_param[3,i,j]=A7_f#*flux_f
                        model_param[4,i,j]=fac_f
                        model_param[5,i,j]=A1_f/fac_f#*flux_f
                        model_param[6,i,j]=A1_f/fac_f/3.0#*flux_f
                        model_param[7,i,j]=A3_f/fac_f#*flux_f
                        model_param[8,i,j]=dv1_f
                        model_param[9,i,j]=dv2_f
                        model_param[10,i,j]=dv3_f
                        model_param[11,i,j]=fwhm1_f
                        model_param[12,i,j]=fwhm2_f
                    else:
                        A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                        model,m2B,m2R,mHaB,mHaR,m1B,m1R,mHaBR=line_model(theta_max, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, ret_com=True)
                        model_all[:,i,j]=model
                        model_Blue[:,i,j]=m2B+m1B+mHaB
                        model_Red[:,i,j]=m2R+m1R+mHaR
                        model_Broad[:,i,j]=mHaBR
                        model_param[0,i,j]=A1_f#*flux_f
                        model_param[1,i,j]=A1_f/2.93#*flux_f
                        model_param[2,i,j]=A3_f#*flux_f
                        model_param[3,i,j]=A7_f#*flux_f
                        model_param[4,i,j]=fac_f
                        model_param[5,i,j]=A1_f/fac_f#*flux_f
                        model_param[6,i,j]=A1_f/fac_f/2.93#*flux_f
                        model_param[7,i,j]=A3_f/fac_f#*flux_f
                        model_param[8,i,j]=dv1_f
                        model_param[9,i,j]=dv2_f
                        model_param[10,i,j]=dv3_f
                        model_param[11,i,j]=fwhm1_f
                        model_param[12,i,j]=fwhm2_f
                    if cont:
                        model_param[13,i,j]=fluxpt    
                if plot_f:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(7,5))#(22,6))
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
                    #ax1.plot(wave_i,fluxtE,linewidth=0.5,color='grey',label=r'$1\sigma$ Error')
                    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
                    if hbfit:
                        ax1.plot(wave_i,mHbBR,linewidth=1,color='red',label=r'Hb_n_BR')
                    else:
                        ax1.plot(wave_i,mHaBR,linewidth=1,color='red',label=r'Ha_n_BR')
                    if single:
                        ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_NR')
                        ax1.plot(wave_i,mHaB,linewidth=1,color='blue',label=r'Ha_n_NR')
                        ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_NR')
                    else:
                        if hbfit:
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'OIII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'OIII_2_r')
                            ax1.plot(wave_i,mHbB,linewidth=1,color='blue',label=r'Hb_n_b')
                            ax1.plot(wave_i,mHbR,linewidth=1,color='red',label=r'Hb_n_r')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'OIII_1_b')
                            ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'OIII_1_r')
                        else:
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'NII_2_r')
                            ax1.plot(wave_i,mHaB,linewidth=1,color='blue',label=r'Ha_n_b')
                            ax1.plot(wave_i,mHaR,linewidth=1,color='red',label=r'Ha_n_r')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_b')
                            ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'NII_1_r')
                    fontsize=14
                    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
                    ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
                    ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
                    ax1.legend(fontsize=fontsize)
                    plt.tight_layout()
                    plt.show()
                    if single:
                        labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3"]
                        labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$']#,r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']#,"FWHM_B","A7","dv3"]
                    else:
                        labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                        labels2 = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                    import corner  
                    #fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
                    #fig.set_size_inches(6.8, 6.8)
                    #plt.show()
                    #print(samples.shape)
                    fig = corner.corner(samples[:,0:8],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
                    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
                    fig.savefig('corners_NAME.pdf')#.replace('NAME',name)
                
                    
                    
                    med_model, spread = sample_walkers(10, samples, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, single=single)
                    
                    
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6*1.5,3*1.5))
                    ax1 = fig.add_subplot(1,1,1)
                    if hbfit:
                        ax1.set_xlim(4800,5100)
                    else:
                        ax1.set_xlim(6530,6600)
                    ax1.plot(wave_i,fluxt,label='Input spectrum')
                    ax1.plot(wave_i,model,label='Highest Likelihood Model')
                    plt.ylabel(r'$Flux\ [10^{-16} erg/s/cm^2/\AA]$',fontsize=16)
                    plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=16)
                    ax1.fill_between(wave_i,med_model-spread*50,med_model+spread*50,color='grey',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
                    ax1.legend(fontsize=14)
                    plt.tight_layout()
                    plt.savefig('spectra_mod.pdf')
                    #plt.show()
                if pgr_bar == False:  
                    if single:  
                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                    else:
                        print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                sys.exit()        
        if pgr_bar:
            progress2.update(i)
    
    if single:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h4=fits.ImageHDU(model_Broad)
    else:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h3=fits.ImageHDU(model_Red)
        h4=fits.ImageHDU(model_Broad)
    h_k=h1.header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        h_k[keys[i]]=hdr[keys[i]]
        h_k.comments[keys[i]]=hdr.comments[keys[i]]
    h_k['EXTNAME'] ='Model'    
    h_k.update()
    if single:
        h_t=h2.header
        for i in range(0, len(keys)):
            h_t[keys[i]]=hdr[keys[i]]
            h_t.comments[keys[i]]=hdr.comments[keys[i]]
        h_t['EXTNAME'] ='Narrow_Component'
        h_t.update()  
    else:
        h_t=h2.header
        for i in range(0, len(keys)):
            h_t[keys[i]]=hdr[keys[i]]
            h_t.comments[keys[i]]=hdr.comments[keys[i]]
        h_t['EXTNAME'] ='Blue_Component'
        h_t.update()
        h_r=h3.header
        for i in range(0, len(keys)):
            h_r[keys[i]]=hdr[keys[i]]
            h_r.comments[keys[i]]=hdr.comments[keys[i]]
        h_r['EXTNAME'] ='Red_Component'
        h_r.update()    
    h_y=h4.header
    for i in range(0, len(keys)):
        h_y[keys[i]]=hdr[keys[i]]
        h_y.comments[keys[i]]=hdr.comments[keys[i]]
    h_y['EXTNAME'] ='Broad_Component'
    h_y.update()   
    if single:
        hlist=fits.HDUList([h1,h2,h4])
    else:
        hlist=fits.HDUList([h1,h2,h3,h4])
    hlist.update_extend()
    hlist.writeto(file_out+'.fits', overwrite=True)
    sycall('gzip -f '+file_out+'.fits')  
    
    h1=fits.PrimaryHDU(model_param)
    h=h1.header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h[keys[i]]=hdr[keys[i]]
            h.comments[keys[i]]=hdr.comments[keys[i]]
    if single:
        h['Val_0'] ='NII_6585_Amplitude'
        h['Val_1'] ='NII_6549_Amplitude'
        h['Val_2'] ='H_alpha_Amplitude'
        h['Val_3'] ='H_alpha_Broad_Amplitude'
        h['Val_4'] ='Narrow_vel'
        h['Val_5'] ='Broad_vel'
        h['Val_6'] ='FWHM_Narrow'
        h['Val_7'] ='FWHM_Broad' 
        if cont:
            h['Val_8'] ='Continum' 
    else:
        if hbfit:      
            h['Val_0'] ='OIII_5007_Amplitude_blue'
            h['Val_1'] ='OIII_4959_Amplitude_blue'
            h['Val_2'] ='H_beta_Amplitude_blue'
            h['Val_3'] ='H_beta_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_5007_Amplitude_red'
            h['Val_6'] ='NII_4959_Amplitude_red'
            h['Val_7'] ='H_beta_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'  
        else:  
            h['Val_0'] ='NII_6585_Amplitude_blue'
            h['Val_1'] ='NII_6549_Amplitude_blue'
            h['Val_2'] ='H_alpha_Amplitude_blue'
            h['Val_3'] ='H_alpha_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_6585_Amplitude_red'
            h['Val_6'] ='NII_6549_Amplitude_red'
            h['Val_7'] ='H_alpha_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'   
        if cont:
            h['Val_13'] ='Continum'     
    try:    
        del h['CRVAL3']
        del h['CRPIX3']
        del h['CDELT3']    
    except:
        print('No vals')
        #del h['CRVAL3']
        #del h['CRPIX3']
        #del h['CD3_3'] 
    h.update()        
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    hlist.writeto(file_out2+'.fits', overwrite=True)
    sycall('gzip -f '+file_out2+'.fits')
    
                          
file1='../data/Holm15A_Gas.fits.gz'   
file2='../data/MUSE_sub1_ress_Gas.fits.gz'
file3='../data/Holm15A_Gas_map.fits.gz'   
file_out='../data/Holm15A_Line_models.fits.gz'  
file_out2='../data/Holm15A_Line_param.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False)#,pgr_bar=False)    
file_out='../data/Holm15A_Line_models_single.fits.gz'  
file_out2='../data/Holm15A_Line_param_single.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True)#,pgr_bar=False,erft=0.75) 
file3='../data/Holm15A_Gas_map2.fits.gz'   
file_out='../data/Holm15A_Line_models_single_v2.fits.gz'  
file_out2='../data/Holm15A_Line_param_single_v2.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True,erft=0.75) 
file3='../data/Holm15A_Gas_map3.fits.gz'   
file_out='../data/Holm15A_Line_models_single_v3'  
file_out2='../data/Holm15A_Line_param_single_v3'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True,erft=0.75)#,pgr_bar=False) 

file1='../data/MUSE_sub1_ress_Gas.fits.gz'   
file2='../data/MUSE_sub1_ress_Gas.fits.gz'
file3='../data/MUSE_sub1_ress_Gas_map.fits.gz'       
file_out='../data/MUSE_sub1_ress_Line_models_single.fits.gz'  
file_out2='../data/MUSE_sub1_ress_Line_param_single.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True) 

file3='../data/Holm15A_Gas_map.fits.gz'       
file_out='../data/MUSE_sub1_ress_Line_models_single_v2.fits.gz'  
file_out2='../data/MUSE_sub1_ress_Line_param_single_v2.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True) 
file3='../data/MUSE_sub1_ress_Gas_map3.fits.gz'       
file_out='../data/MUSE_sub1_ress_Line_models_single_v3'  
file_out2='../data/MUSE_sub1_ress_Line_param_single_v3'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True) 

file1='../data/MUSE_MEGARA_FoV_Gas.fits.gz'   
file2='../data/MUSE_MEGARA_FoV_Gas.fits.gz'
file3='../data/MUSE_MEGARA_FoV_Gas_map.fits.gz'       
file_out='../data/MUSE_Line_models_single.fits.gz'  
file_out2='../data/MUSE_Line_param_single.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True,flux_f=(1/10000.*(0.35/0.2)**2.0)) 
file_out='../data/MUSE_Line_models_single_v2.fits.gz'  
file_out2='../data/MUSE_Line_param_single_v2.fits.gz'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,lA1=6475.0,lA2=6630.0,plot_f=False,single=True,flux_f=(1/10000.*(0.35/0.2)**2.0)) 

file1='../data/J102700+174900_Gas.fits.gz'   
file2='../data/J102700+174900_Gas.fits.gz'
file3='../data/J102700+174900_Gas_map.fits.gz'       
file_out='../data/J102700+174900_models'  
file_out2='../data/J102700+174900_param'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.06662,lA1=6475.0,lA2=6630.0,plot_f=False,single=False,pgr_bar=True,dv1t=700,sim=False) 
file_out='../data/J102700+174900_models_single'  
file_out2='../data/J102700+174900_param_single'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.06662,lA1=6475.0,lA2=6630.0,plot_f=True,single=True,pgr_bar=False,dv1t=700,sim=True) 
file_out='../data/J102700+174900_modelsV2'  
file_out2='../data/J102700+174900_paramV2'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.06662,lA1=6475.0,lA2=6630.0,plot_f=False,single=False,pgr_bar=True,dv1t=700,sim=True) 
file1='../data/Mrk516_Rn.fits.gz'   
file2='../data/Mrk516_Rn.fits.gz'
file3='../data/Mrk516_Rn_map.fits.gz'  
file_out='../data/Mrk516_Rn_modelsV2'  
file_out2='../data/Mrk516_Rn_paramV2'
#line_fit(file1,file2,file3,file_out,file_out2,z=0.0284963,lA1=6475.0,lA2=6630.0,plot_f=False,single=False,pgr_bar=True,dv1t=700,sim=True) 

file1='../data/Residual_MK883_N_sp1_fin.fits.gz'
file2='../data/MK883_N.fits.gz'
file3='../data/MK883_N_sp1_fin_map.fits.gz'  
file_out='../data/MK883_N_sp1_fin_modelsV2_Ha'  
file_out2='../data/MK883_N_sp1_fin_paramV2_Ha'
zt=0.03787+0.00042+(0.00042*0.03787)
#line_fit(file1,file2,file3,file_out,file_out2,z=zt,lA1=6475.0,lA2=6630.0,plot_f=True,single=False,pgr_bar=False,dv1t=1000,sim=True,cont=True) 
file_out='../data/MK883_N_sp1_fin_modelsV2_Hb'  
file_out2='../data/MK883_N_sp1_fin_paramV2_Hb'
zt=0.03787+0.00042+(0.00042*0.03787)#-193/299792.458
#line_fit(file1,file2,file3,file_out,file_out2,z=zt,lA1=4800.0,lA2=5100.0,plot_f=True,single=False,pgr_bar=False,dv1t=1000,sim=True,cont=True,hbfit=True) 
file1='Mock/CollGalSFR-THighress-547000.cube.fits.gz'
file2='Mock/CollGalSFR-THighress-547000.cube.fits.gz'
file3='Mock/CollGalSFR-THighress-547000_map.fits.gz'
file_out='Mock/CollGalSFR-THighress-547000_modelV2_Ha'  
file_out2='Mock/CollGalSFR-THighress-547000_paramV2_Ha'
zt=0.03361535853483399
line_fit(file1,file2,file3,file_out,file_out2,z=zt,lA1=6475.0,lA2=6630.0,plot_f=True,single=False,pgr_bar=False,dv1t=1000,sim=True,cont=True) 