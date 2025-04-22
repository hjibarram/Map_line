#!/usr/bin/env python
import numpy as np
from scipy.ndimage import gaussian_filter1d as filt1d
import os
import os.path as ptt
from scipy.special import erf as errf
from scipy.special import voigt_profile as vprf
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

def voigt(x,sigma=1.0,xo=0.0,A1=1.0,gam1=0.0):
    At=A1/vprf(0, sigma, gam1)
    #sigma=sigma/2.0
    #gam=gam/2.0
    x1=x-xo
    #A1=A1/(np.sqrt(2.0*np.pi)*sigma)
    #y=A1*vprf(x,sigma,gam)
    y=At*vprf(x1,sigma,gam1)
    return y


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

def get_apertures(file):
    ra=[]
    dec=[]
    rad=[]
    colr=[]
    namet=[]
    l1=[]
    l2=[]
    th=[]
    typ=[]
    f=open(file,'r')
    ct=1
    for line in f:
        if not 'Region' in line and not 'fk5' in line and not 'global' in line:
            if 'circle' in line:
                data=line.replace('\n','').replace('circle(','').replace('") # color=',' , ').replace(' width=',' , ').replace(' text={',' , ').replace('}',' ')
                data=data.split(',')
                data=list(filter(None,data))
                #print(data)
                ra.extend([data[0]])
                dec.extend([data[1]])
                rad.extend([float(data[2])])
                colr.extend([data[3].replace(' ','')])
                try:
                    namet.extend([data[5].replace(' ','')])
                except:
                    namet.extend([str(int(ct))])
                l1.extend([np.nan])
                l2.extend([np.nan])
                th.extend([np.nan])    
                typ.extend(['circle'])
            if 'box' in line:
                data=line.replace('\n','').replace('box(','').replace(') # color=',' , ').replace(' width=',' , ').replace(' text={',' , ').replace('}',' ')
                data=data.split(',')
                data=list(filter(None,data))
                ra.extend([data[0]])
                dec.extend([data[1]])
                l1.extend([float(data[2].replace('"',''))])
                l2.extend([float(data[3].replace('"',''))])
                th.extend([float(data[4])])
                colr.extend([data[5].replace(' ','')])
                try:
                    namet.extend([data[7].replace(' ','')])
                except:
                    namet.extend([str(int(ct))])
                rad.extend([np.nan])    
                typ.extend(['box'])
            ct=ct+1
    ra=np.array(ra)
    dec=np.array(dec)
    rad=np.array(rad)
    colr=np.array(colr)
    namet=np.array(namet)
    typ=np.array(typ)
    l1=np.array(l1)
    l2=np.array(l2)
    th=np.array(th)
    return ra,dec,rad,l1,l2,th,colr,namet,typ

def extract_regs(map,hdr,reg_file='file.reg',avgra=False):
    try:
        dx=np.sqrt((hdr['CD1_1'])**2.0+(hdr['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr['CD2_1'])**2.0+(hdr['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr['CD1_1']*3600.0
            dy=hdr['CD2_2']*3600.0
        except:
            dx=hdr['CDELT1']*3600.
            dy=hdr['CDELT2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0  

    ra,dec,rad,l1,l2,th,colr,namet,typ=get_apertures(reg_file)
    nreg=len(ra)
    val=np.zeros(nreg)
    for i in range(0, nreg):
        if typ[i] == 'circle':
            val[i]=extract_single_reg(map,hdr,ra=ra[i],dec=dec[i],rad=rad[i],pix=pix,avgra=avgra)
        #if typ[i] == 'box':
        #    val[i]=extract_single_reg(map,hdr,ra=ra[i],dec=dec[i],rad=rad[i],pix=pix,avgra=avgra)
    return val
    

def extract_single_reg(map,hdr,ra='',dec='',rad=1.5,pix=0.35,avgra=False):
    sky1=SkyCoord(ra+' '+dec,frame=FK5, unit=(u.hourangle,u.deg))
    val1=sky1.ra.deg
    val2=sky1.dec.deg
    wcs = WCS(hdr)
    wcs=wcs.celestial
    ypos,xpos=skycoord_to_pixel(sky1,wcs)
    val1=sky1.to_string('hmsdms')
    
    nx,ny=map.shape
    radis=np.zeros([nx,ny])
    for i in range(0, nx):
        for j in range(0, ny):
            x_n=i-xpos
            y_n=j-ypos
            r_n=np.sqrt((y_n)**2.0+(x_n)**2.0)*pix
            radis[i,j]=r_n
    nt=np.where(radis <= rad)
    tmp=np.copy(map)
    if avgra:
        value=np.nanmean(tmp[nt])
    else:
        value=np.nansum(tmp[nt])
    return value


def bpt(wha,niiha,oiiihb,ret=4,agn=3,sf=1,inte=2.5,comp=5):
    nt1=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) > 0) & (np.isfinite(oiiihb)) & (np.isnan(oiiihb) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#AGN
    nt2=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) <= 0) & ((oiiihb-0.61/(niiha-0.05)-1.3) > 0) & (np.isfinite(oiiihb)) & (np.isnan(niiha) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#COMP
    nt3=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.05)-1.3) <= 0) & (np.isfinite(oiiihb)) & (np.isnan(niiha) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#SF
    nt4=np.where((wha > 3) & (wha <6))#INT
    nt5=np.where((wha <=3))#RET
    image=np.copy(niiha)
    image=image*0
    image[:,:]=np.nan
    image[nt1]=agn
    image[nt2]=comp
    image[nt3]=sf
    image[nt4]=inte
    image[nt5]=ret
    return image

def whan(wha,niiha,agn=4,sf=1.7,wagn=3,ret=1):
    nt1=np.where((wha >  6) & (niiha >= -0.4))#sAGN
    nt2=np.where((wha >= 3) & (niiha < -0.4))#SFR
    nt3=np.where((wha >= 3) & (wha <= 6) & (niiha >= -0.4))#wAGN
    nt4=np.where((wha <  3))#RET
    image=np.copy(wha)
    image[:,:]=np.nan
    image[nt1]=agn
    image[nt2]=sf
    image[nt3]=wagn
    image[nt4]=ret
    return image    


def whad(logew,logsig,agn=5,sf=3,wagn=4,ret=2,unk=1):
    nt1=np.where((logew>=np.log10(10)) & (logsig>=np.log10(57))) #AGN
    nt2=np.where((logew>=np.log10(6)) & (logsig<np.log10(57))) #SF
    nt3=np.where((logew>=np.log10(3)) & (logew<np.log10(10)) & (logsig>=np.log10(57))) #WAGN
    nt4=np.where((logew<np.log10(3))) #Ret
    nt5=np.where((logew>=np.log10(3)) & (logew<np.log10(6)) & (logsig<np.log10(57))) #Unk
    image=np.copy(logew)
    image[:,:]=np.nan
    image[nt1]=agn
    image[nt2]=sf
    image[nt3]=wagn
    image[nt4]=ret
    image[nt5]=unk
    return image


def jwst_nirspecIFU_MJy2erg(file,file_out,zt=0,path='',path_out=''):
    erg2jy=1.0e-23
    vel_light=299792458.0
    ang=1e-10
    filename=path+file
    filename_out=path_out+file_out
    #[cube0, hdr0]=fits.getdata(filename, 0, header=True)
    [cube1, hdr1]=fits.getdata(filename, 1, header=True)
    [cube2, hdr2]=fits.getdata(filename, 2, header=True)

    crpix=hdr1["CRPIX3"]
    cdelt=hdr1["CDELT3"]
    crval=hdr1["CRVAL3"]
    dx=hdr1['CDELT1']
    dy=hdr1['CDELT2']
    pix=(np.abs(dx)+np.abs(dy))/2.0 
    pixS=(pix*np.pi/180.0)
    nz,nx,ny=cube1.shape
    wave=(crval+cdelt*(np.arange(nz)+1-crpix))*1e4
    for i in range(0,nx):
        for j in range(0,ny):
            cube1[:,i,j]=cube1[:,i,j]*erg2jy*vel_light/wave**2.0/ang/1e-17*pixS**2*1e6
            cube2[:,i,j]=cube2[:,i,j]*erg2jy*vel_light/wave**2.0/ang/1e-17*pixS**2*1e6

    h1=fits.PrimaryHDU()
    h2=fits.ImageHDU(cube1,header=hdr1)
    h=h2.header
    h['CRVAL3']=h['CRVAL3']*1e4/(1+zt)
    h['CDELT3']=h['CDELT3']*1e4/(1+zt)
    h['CUNIT3']='Angstrom'
    h['BUNIT']='erg/s/cm^2/Angstrom'
    h.update()  
    h3=fits.ImageHDU(cube2,header=hdr2)
    #h=h3.header
    #h['CRVAL3']=h['CRVAL3']*1e4/(1+zt)
    #h['CDELT3']=h['CDELT3']*1e4/(1+zt)
    #h['CUNIT3']='Angstrom'
    #h.update()  
    hlist=fits.HDUList([h1,h2,h3])
    hlist.update_extend()
    hlist.writeto(filename_out, overwrite=True)
    sycall('gzip -f '+filename_out)