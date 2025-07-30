#!/usr/bin/env python
import numpy as np
from scipy.ndimage import gaussian_filter1d as filt1d
from scipy.ndimage import gaussian_filter as filtNd
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
import numpy as np
import matplotlib.tri as mtri
from stl import mesh
import warnings
warnings.filterwarnings("ignore")

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


def get_segment(reg_dir='./',reg_name='test.reg'):
    raL=[]
    decL=[]
    colr=[]
    namet=[]
    f=open(reg_dir+reg_name,'r')
    ct=1
    for line in f:
        if not 'Region' in line and not 'fk5' in line and not 'global' in line:
            if 'segment' in line:
                data=line.replace('\n','').replace('# segment(',')').split(')')
                data=list(filter(None,data))
                data1=data[0].split(',')
                data1=list(filter(None,data1))
                rat=[]
                dect=[]
                for k in range(0, len(data1),2):
                    rat.extend([data1[k]])
                    dect.extend([data1[k+1]])
                rat=np.array(rat)
                dect=np.array(dect)
                raL.extend([rat])
                decL.extend([dect])
                data2=data[1].replace('color=','').replace(' width=',' , ').replace(' text={',' , ').replace('}',' ')
                data2=data2.split(',')
                data2=list(filter(None,data2))
                colr.extend([data2[0].replace(' ','')])
                try:
                    namet.extend([data2[2].replace(' ','')])
                except:
                    namet.extend([str(int(ct))])
            ct=ct+1
    colr=np.array(colr)
    namet=np.array(namet)
    return raL,decL,colr,namet    

def extract_segment1d(file,wcs=None,reg_dir='./',reg_name='test.reg',z=0,rad=1.5,lA1=6450.0,lA2=6850.0,plot_t=False,sigT=4,cosmetic=False):
    ra,dec,colr,namet=get_segment(reg_dir=reg_dir,reg_name=reg_name)
    [pdl_cube, hdr]=fits.getdata(file, 0, header=True)
    nz,nx,ny=pdl_cube.shape
    crpix=hdr["CRPIX3"]
    try:
        cdelt=hdr["CD3_3"]
    except:
        cdelt=hdr["CDELT3"]
    crval=hdr["CRVAL3"]
    
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
    dpix=(np.abs(dx)+np.abs(dy))/2.0  
    
    wave=(crval+cdelt*(np.arange(nz)+1-crpix))#*1e4 
    
    spl=299792458.0
    wave=wave/(1+z)
    nw=np.where((wave >= lA1) & (wave <= lA2))[0]
    wave_f=wave[nw]
    if wcs == None:
        #print('TEST')
        wcs = WCS(hdr)
        wcs=wcs.celestial
    slides=[]
    vals=[]
    namesS=[]
    for i in range(0, len(ra)):
        raT=ra[i]
        decT=dec[i]
        cosT=np.zeros([len(raT)-1])
        sinT=np.zeros([len(raT)-1])
        rtf=np.zeros([len(raT)-1])
        xposT=np.zeros([len(raT)-1])
        yposT=np.zeros([len(raT)-1])
        lT=np.zeros([len(raT)-1],dtype=int)
        slidesT=[]
        ltf=0
        flux1t=pdl_cube[nw,:,:]
        namesST=[]
        for j in range(0, len(raT)-1):
            sky1=SkyCoord(raT[j]+' '+decT[j],frame=FK5, unit=(u.hourangle,u.deg))
            sky2=SkyCoord(raT[j+1]+' '+decT[j+1],frame=FK5, unit=(u.hourangle,u.deg))
            ypos1,xpos1=skycoord_to_pixel(sky1,wcs)
            ypos2,xpos2=skycoord_to_pixel(sky2,wcs)
            rt=np.sqrt((xpos2-xpos1)**2.0+(ypos2-ypos1)**2.0)
            cosT[j]=(ypos2-ypos1)/rt
            sinT[j]=(xpos2-xpos1)/rt
            xposT[j]=xpos1
            yposT[j]=ypos1
            rtf[j]=rt*dpix
            lt=int(np.round(rt))+1
            lT[j]=lt
            
            slideT=np.zeros(len(nw))
            radis=np.zeros([nx,ny])
            for ii in range(0, nx):
                for jj in range(0, ny):
                    x_n=ii-xpos1
                    y_n=jj-ypos1
                    r_n=np.sqrt((y_n)**2.0+(x_n)**2.0)*dpix
                    radis[ii,jj]=r_n
            ntp=np.where(radis <= rad)
            for ii in range(0, len(nw)):
                slideT[ii]=np.nansum(flux1t[ii,ntp])
            namesST.extend([str(int(j))])
            #for k in range(0, lt):
            #    yt=int(np.round(ypos1+k*cosT[j]))
            #    xt=int(np.round(xpos1+k*sinT[j]))
                
                
                #flux1t=flux1t*spl/(wave[nw]*(1+z)*1e-10)**2.*1e-10*1e-23*2.35040007004737e-13/1e-16/1e3
            if cosmetic:
                slideT=conv(slideT,ke=sigT)
                #slideT[k,:]=flux1t
            slidesT.extend([slideT])
            ltf=1+ltf 
            
            if j == len(raT)-2:
                slideT=np.zeros(len(nw))
                radis=np.zeros([nx,ny])
                for ii in range(0, nx):
                    for jj in range(0, ny):
                        x_n=ii-xpos2
                        y_n=jj-ypos2
                        r_n=np.sqrt((y_n)**2.0+(x_n)**2.0)*dpix
                        radis[ii,jj]=r_n
                ntp=np.where(radis <= rad)
                for ii in range(0, len(nw)):
                    slideT[ii]=np.nansum(flux1t[ii,ntp])
                slidesT.extend([slideT])
                ltf=1+ltf 
                namesST.extend([str(int(j+1))])
        namesS.extend([namesST])
        slide=np.zeros([ltf,len(nw)])
        #ct=0
        for j in range(0, len(raT)):
            sldT=slidesT[j]
            #for k in range(0, lT[j]):
            slide[j,:]=sldT#[k,:]     
               # ct=ct+1
        #out={'Slide':slide,'Lt':lt}
        slides.extend([slide])
        vals.extend([[cosT,sinT,rtf,yposT,xposT]])
        if plot_t:
            cm=plt.cm.get_cmap('jet')    
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(slide,origin='lower',cmap=cm,extent=[wave_f[0],wave_f[len(nw)-1],0,ltf*dpix],aspect='auto')
            plt.xlim(wave_f[0],wave_f[len(nw)-1])
            plt.ylim(0,ltf*dpix) 
            plt.show()
    return slides,wave_f,dpix,vals,hdr,colr,namet,namesS


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


def bpt(wha,niiha,oiiihb,ret=4,agn=3,sf=1,inte=2.5,comp=5,save=False,path='',name='BPT_map',hdr=None):
    nt1=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) > 0) & (np.isfinite(oiiihb)) & (np.isnan(oiiihb) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#AGN
    nt2=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.47)-1.19) <= 0) & ((oiiihb-0.61/(niiha-0.05)-1.3) > 0) & (np.isfinite(oiiihb)) & (np.isnan(niiha) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#COMP
    nt3=np.where((wha >=6) & ((oiiihb-0.61/(niiha-0.05)-1.3) <= 0) & (np.isfinite(oiiihb)) & (np.isnan(niiha) == False) & (np.isfinite(niiha)) & (np.isnan(niiha) == False))#SF
    nt4=np.where((wha > 3) & (wha <6))#INT
    nt5=np.where((wha <=3) & (wha > 0))#RET
    image=np.copy(niiha)
    image=image*0
    image[:,:]=np.nan
    image[nt1]=agn
    image[nt2]=comp
    image[nt3]=sf
    image[nt4]=inte
    image[nt5]=ret
    if save:
        filename=path+name+'.fits'
        if hdr:
            h1=fits.PrimaryHDU(image,header=hdr)
        else:
            h1=fits.PrimaryHDU(image)
        hlist=fits.HDUList([h1])
        hlist.update_extend()
        hlist.writeto(filename, overwrite=True)
        sycall('gzip -f '+filename)
    return image

def whan(wha,niiha,agn=4,sf=1.7,wagn=3,ret=1,save=False,path='',name='WHAN_map',hdr=None):
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
    if save:
        filename=path+name+'.fits'
        if hdr:
            h1=fits.PrimaryHDU(image,header=hdr)
        else:
            h1=fits.PrimaryHDU(image)
        hlist=fits.HDUList([h1])
        hlist.update_extend()
        hlist.writeto(filename, overwrite=True)
        sycall('gzip -f '+filename)
    return image    


def whad(logew,logsig,agn=5,sf=3,wagn=4,ret=2,unk=1,save=False,path='',name='WHAD_map',hdr=None):
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
    if save:
        filename=path+name+'.fits'
        if hdr:
            h1=fits.PrimaryHDU(image,header=hdr)
        else:
            h1=fits.PrimaryHDU(image)
        hlist=fits.HDUList([h1])
        hlist.update_extend()
        hlist.writeto(filename, overwrite=True)
        sycall('gzip -f '+filename)
    return image

def get_map_to_stl(map, nameid='', path_out='',sig=2,smoth=False, pval=27, mval=0, border=False,logP=False,ofsval=-1,maxval=None,minval=None):
    """
    Convert a 2D map to an STL file.
    
    Parameters:
    - file_out: Output STL file name.
    - path_out: Path to save the output STL file.
    """
    
    indx = np.where(map == 0)
    indxt= np.where(map != 0)
    map[indx] = np.nan
    if logP:
        map=np.log10(map)      
    if smoth:
        map[np.where(map < ofsval)]=ofsval
        map[np.where(np.isfinite(map) == False)]=ofsval
        map=filtNd(map, sigma=sig)
    if maxval is None:
        maxval=np.nanmax(map[indxt])
    if minval is None:
        minval=np.nanmin(map[indxt])
    map=(map-minval)/(maxval-minval)*pval+mval
    map[np.where(np.isfinite(map) == False)]=0
    map[indx]=0
    map[np.where(map < 0)]=0
    if border:
        nx,ny=map.shape
        map[0:3,0:ny]=0#1
        map[nx-3:nx,0:ny]=0
        map[0:nx,0:3]=0
        map[0:nx,ny-3:ny]=0
    # Convert the map to STL format
    map_to_stl(map, nameid, path_out)

def get_maps_to_stl(file_in, nameid='', path_in='', path_out='',sig=2,smoth=False, pval=27, mval=0, border=False):
    """
    Convert a 2D map from a FITS file to an STL file.
    
    Parameters:
    - file_in: Input FITS file containing the map.
    - file_out: Output STL file name.
    - path_in: Path to the input FITS file.
    - path_out: Path to save the output STL file.
    """
    # Read the FITS file
    mapdata, hdr = fits.getdata(path_in + file_in, header=True)
    keys = list(hdr.keys())
    for key in keys:
        if 'VAL_' in key:
            head_val= hdr[key]
            if 'Continum' in head_val:
                idx = int(key.replace('VAL_', ''))
                cont=mapdata[idx,:,:]
                indx = np.where(cont == 0)
                indxt= np.where(cont != 0)
    for key in keys:
        if 'VAL_' in key:
            head_val= hdr[key]
            idx = int(key.replace('VAL_', ''))
            map=mapdata[idx,:,:]
            map[indx] = np.nan
            if 'Amplitude' in head_val or 'Continum' in head_val:
                map=np.log10(map)      
            if smoth:
                map[np.where(np.isfinite(map) == False)]=-2
                map=filtNd(map, sigma=sig)
            maxval=np.nanmax(map[indxt])
            minval=np.nanmin(map[indxt])
            map=(map-minval)/(maxval-minval)*pval+mval
            map[np.where(np.isfinite(map) == False)]=0
            map[indx]=0
            map[np.where(map < 0)]=0
            if border:
                nx,ny=map.shape
                map[0:1,0:ny]=0
                map[nx-1:nx,0:ny]=0
                map[0:nx,0:1]=0
                map[0:nx,ny-1:ny]=0
            # Convert the map to STL format
            map_to_stl(map, head_val+nameid, path_out)

def map_to_stl(map, file_out, path_out=''):
    ny, nx = map.shape
    x = np.arange(nx)
    y = np.arange(ny) 
    X, Y = np.meshgrid(x, y)
    # 1. Flatten X, Y, Z for triangulation
    Z = map.flatten()
    X = X.flatten()
    Y = Y.flatten()
    # 2. Triangulate the data
    triang = mtri.Triangulation(X, Y)
    # 3. Create numpy-stl mesh
    data = np.zeros(len(triang.triangles), dtype=mesh.Mesh.dtype)
    surface_mesh = mesh.Mesh(data, remove_empty_areas=False)
    surface_mesh.x[:] = X[triang.triangles]
    surface_mesh.y[:] = Y[triang.triangles]
    surface_mesh.z[:] = Z[triang.triangles]
    # 4. Save to STL
    surface_mesh.save(path_out+file_out+'.stl')

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