#!/usr/bin/env python
import numpy as np
from scipy.ndimage import gaussian_filter1d as filt1d
import os
import os.path as ptt
from scipy.special import erf as errf
import yaml
import sys
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS

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

def lorentz(x,sigma=1.0,xo=0.0,A1=1.0):
    y=A1*(0.5*sigma)**2.0/((x-xo)**2.0+(0.5*sigma)**2.0) 
    return y

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

def read_config_file(file):
    try:
        with open(file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data
    except:
        print('Config File not found')
        return None

def get_fluxline(file,path='',ind1=3,ind2=7,ind3=4,ind4=9,lo=6564.632,zt=0.0,val0=0):
    ct=299792.458
    file0=path+'/'+file
    [pdl_cube0, hdr0]=fits.getdata(file0, 0, header=True)
    Amp=pdl_cube0[ind1,:,:]
    fwhm=pdl_cube0[ind2,:,:]
    vel=pdl_cube0[ind3,:,:]
    nt=np.where(np.round(vel,decimals=3) == val0)
    vel=vel+zt*ct
    
    try:
        cont=pdl_cube0[ind4,:,:]
        conti=True
    except:
        conti=False
    sigma=fwhm/ct*lo/(2.0*np.sqrt(2.0*np.log(2.0)))
    flux=np.sqrt(2.0*np.pi)*sigma*Amp
    if conti:
        ew=flux/cont
    else:
        ew=None
    sigma=fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))    
    if len(nt) > 0:
        vel[nt]=0
        flux[nt]=0
        sigma[nt]=0
        ew[nt]=0
    return flux,vel,sigma,ew

def extract_spec(filename,dir_cube_m='',ra='',dec='',rad=1.5,sig=10,smoth=False,avgra=False,head=0):
    file=dir_cube_m+filename

    [cube0, hdr0]=fits.getdata(file, head, header=True)
    nz,nx,ny=cube0.shape
    try:
        dx=np.sqrt((hdr0['CD1_1'])**2.0+(hdr0['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr0['CD2_1'])**2.0+(hdr0['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr0['CD1_1']*3600.0
            dy=hdr0['CD2_2']*3600.0
        except:
            dx=hdr0['CDELT1']*3600.
            dy=hdr0['CDELT2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0    
    

    if ra != '':
        sky1=SkyCoord(ra+' '+dec,frame=FK5, unit=(u.hourangle,u.deg))
        val1=sky1.ra.deg
        val2=sky1.dec.deg
        wcs = WCS(hdr0)
        wcs=wcs.celestial
        ypos,xpos=skycoord_to_pixel(sky1,wcs)
    else:
        xpos=ny/2.0
        ypos=nx/2.0
        
    radis=np.zeros([nx,ny])
    for i in range(0, nx):
        for j in range(0, ny):
            x_n=i-xpos
            y_n=j-ypos
            r_n=np.sqrt((y_n)**2.0+(x_n)**2.0)*pix
            radis[i,j]=r_n
    single_T=np.zeros(nz)
    
    nt=np.where(radis <= rad)
    if avgra:
        ernt=len(nt[0])
    else:
        ernt=1.0
    for i in range(0, nz):
        tmp=cube0[i,:,:]
        #tmp[np.where(tmp <= 0)]=np.nan
        if avgra:
            single_T[i]=np.nanmean(tmp[nt])
        else:
            single_T[i]=np.nansum(tmp[nt])
        
        
    crpix=hdr0["CRPIX3"]
    try:
        cdelt=hdr0["CD3_3"]
    except:
        cdelt=hdr0["CDELT3"]
    crval=hdr0["CRVAL3"]
    wave_f=(crval+cdelt*(np.arange(nz)+1-crpix))
    
    
    
    if smoth:
        single_T=conv(single_T,ke=sig)
    
    return wave_f,single_T    

def bpt(wha,niiha,oiiihb):
    nt1=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) > 0) & (np.isfinite(oiiihb)) & (np.isnan(oiiihb) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#AGN
    nt2=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) <= 0) & (np.isfinite(oiiihb)) & (np.isnan(niiha) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#SF
    nt3=np.where((wha > 3) & (wha <6))#INT
    nt4=np.where((wha <=3))#RET
    image=np.copy(niiha)
    image=image*0
    image[:,:]=np.nan
    image[nt1]=3
    image[nt2]=1
    image[nt3]=2.5
    image[nt4]=4
    return image

def whan(wha,niiha):
    nt1=np.where((wha >  6) & (niiha >= -0.4))#sAGN
    nt2=np.where((wha >= 3) & (niiha < -0.4))#SFR
    nt3=np.where((wha >= 3) & (wha <= 6) & (niiha >= -0.4))#wAGN
    nt4=np.where((wha <  3))#RET
    image=np.copy(wha)
    image[:,:]=np.nan
    image[nt1]=4
    image[nt2]=1.7
    image[nt3]=3
    image[nt4]=1
    return image    
    