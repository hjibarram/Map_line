#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS
from astropy.io import fits
import MapLines.tools.tools as tools
import MapLines.tools.mcmc as mcm
import corner
import cmasher as cmr
from scipy.optimize import curve_fit


def plot_mapapertures(titf,vals_map,nlins=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],titp='Velocity~ shift',lab=r'[km\ s^{-1}]',cont=False,alpha=0.5,lamds=[6549.859,6564.632,6585.278],path='',nzeros=False,hdu=0,wcs=None,file0='J102700+174900_Gas.fits.gz',reg_dir='',reg_aper='apertu.reg',dtex=0,dtey=0,rad=1.5,cosmetic=False,reg_name='paths_J1027_C.reg',zt=0,facs=1,lA1=6520.0,lA2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],av=[0.10,0.03,0.09,0.03],sigT=2,loc=3,facx=0.8,facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,x_max=1,txt_size=18,ylabel='y-value',xlabel='x-value',dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    slides,wavet,dpix,vals,hdr,colr,widt,namet,namesS=tools.extract_segment1d(file0,path=path,wcs=wcs,reg_dir=reg_dir,reg_name=reg_name,nzeros=nzeros,rad=rad,z=zt,lA1=lA1,lA2=lA2,sigT=sigT,cosmetic=cosmetic,hdu=hdu)
    pix=dpix
    if facy == -1:
        facy=facx
    dx1=av[0]/facx
    dx2=av[1]/facx
    dy1=av[2]/facy
    dy2=av[3]/facy
    dx=(1.0-(dx1+dx2))/float(1.0)
    dy=(1.0-(dy1+dy2))/float(1.0)
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/float(nx)
    dy=(1.0-(dy1+dy2))/float(ny)
    xfi=6*nx*facx*facs#6
    yfi=6*ny*facy#*1.2#5.5
    fig = plt.figure(figsize=(xfi,yfi))
    dyt=0.0#0.85
    ax = fig.add_axes([dx1+pro1[0]*dx-dx*0.1, dy1+pro2[0]*dy*dyt, dx, dy*(1.0-dyt)])  
    flux,vmax,vmin=vals_map
    get_plot_map(plt,flux,vmax,vmin,pix=pix,tit=titp,lab=lab,cont=cont,alpha=alpha)
    nxt,nyt=flux.shape
    for i in range(0, len(vals)):
        cosT,sinT,rtf,ytf,xtf=vals[i]
        namesT=namesS[i]
        hwith=widt[i]/5.0*0.25
        for j in range(0, len(cosT)):
            tp=np.arange(0,100)/99.*rtf[j]/pix
            yt=(ytf[j]+cosT[j]*tp-nyt/2.+1)*pix
            xt=(xtf[j]+sinT[j]*tp-nxt/2.+1)*pix
            plt.plot(yt,xt,lw=widt[i],color=colr[i],ls=':')
            plot_circle(ax,xtf[j],ytf[j],nxt,nyt,pix,rad=rad,color=colr[i],name=namesT[j],dtex=dtex,dtey=dtey)
        plot_circle(ax,xt[99]/pix+nxt/2.-1,yt[99]/pix+nyt/2.-1,nxt,nyt,pix,rad=rad,color=colr[i],name=namesT[j+1],dtex=dtex,dtey=dtey)    
        #plt.arrow(yt[98], xt[98], yt[99]-yt[98],  xt[99]-xt[98], color=colr[i],lw=widt[i],head_width=hwith,zorder=2)    
    slideA=slides[0]
    namesT=namesS[0]
    nls,nlt=slideA.shape
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[1]*dx+dx*dxR, dy1+pro2[1]*dy*dyt+dy*(nls-i-1)/nls*(1.0-dyt), dx, dy/(nls)*(1.0-dyt)])
        spectra=slideA[i,:]
        ymax=np.nanmax(spectra)*1.2
        nw=len(spectra)
        plt.plot(wavet,spectra,lw=3,color='black')
        for j in range(0, len(lamds)):
            plt.plot([lamds[j],lamds[j]],[0,ymax],lw=5,ls='--',color='blue')
        plt.xlim(wavet[0],wavet[nw-1])
        plt.ylim(0,ymax)
        plt.ylabel(r'$Flux$',fontsize=18)
        plt.text(0.05,0.35,namesT[i],fontsize=20,transform=ax.transAxes,color=colr[0],weight='bold')
        if i < nls-1:
            ax.set_xlabel('').set_visible(False)
            plt.setp( ax.get_xticklabels(), visible=False)           
        else:
            plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=18)
        if i == 0:
            dtp=-5
            for j in range(0, len(lamds)):
                x_la_t1=lamds[j]+dtp
                y_la_t1=ymax*1.1
                plt.text(x_la_t1, y_la_t1, nlins[j % len(nlins)] , fontsize=18, va='center',color='black',weight='bold')
    if savef:        
        plt.savefig(dir+'/'+titf+'.pdf')
    else:
        plt.show()

def plot_velana2x(titf,vals_map1,vals_map2,dyt=0.95,path='',DA=None,model='helic',alpha=1.0,fitmod=False,file0='J102700+174900_Gas.fits.gz',nlinsA=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],lamdsA=[6549.859,6564.632,6585.278],nlinsB=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],lamdsB=[6549.859,6564.632,6585.278],hdu=0,reg_dir='',reg_aper='apertu.reg',reg_name='paths_J1027_C.reg',zt=0,facs=1,lA1=6520.0,lA2=6610,lB1=6520.0,lB2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],av=[0.10,0.03,0.09,0.03],sigT=2,loc=3,facx=0.8,facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,x_max=1,txt_size=18,ylabel='y-value',xlabel='x-value',dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    slides1,wavet1,dpix,vals1,hdr,colr1,widt1,namet1=tools.extract_segment(file0,path=path,reg_dir=reg_dir,reg_name=reg_name,z=zt,lA1=lA1,lA2=lA2,sigT=sigT,cosmetic=True,hdu=hdu)
    slides2,wavet2,dpix,vals2,hdr,colr2,widt2,namet2=tools.extract_segment(file0,path=path,reg_dir=reg_dir,reg_name=reg_name,z=zt,lA1=lB1,lA2=lB2,sigT=sigT,cosmetic=True,hdu=hdu)
    pix=dpix
    
    if facy == -1:
        facy=facx
    dx1=av[0]/facx
    dx2=av[1]/facx
    dy1=av[2]/facy
    dy2=av[3]/facy
    dx=(1.0-(dx1+dx2))/float(1.0)
    dy=(1.0-(dy1+dy2))/float(1.0)
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/float(nx)
    dy=(1.0-(dy1+dy2))/float(ny)
    xfi=6*nx*facx*facs#6
    yfi=6*ny*facy#5.5
    fig = plt.figure(figsize=(xfi,yfi))
    flux1,vmax1,vmin1=vals_map1
    flux2,vmax2,vmin2=vals_map2
    ax = fig.add_axes([dx1+pro1[0]*dx-dx*0.1, dy1+pro2[0]*dy*dyt, dx, dy*(2.0-dyt)])  
    get_plot_map(plt,flux1,vmax1,vmin1,pix=pix,tit='Velocity shift',lab=r'[km\ s^{-1}]',alpha=alpha)
    nxt,nyt=flux1.shape
    slides_v=tools.extract_segment_val(flux1,hdr,pix,reg_dir=reg_dir,reg_name=reg_name)
    if reg_aper is not None:
        plot_apertures(ax,hdr,plt,nxt,nyt,pix,reg_dir=reg_dir,reg_file=reg_aper)
    for i in range(0, len(vals1)):
        cosT,sinT,rtf,ytf,xtf=vals1[i]
        hwith=widt1[i]/5.0*0.25
        for j in range(0, len(cosT)):
            tp=np.arange(0,100)/99.*rtf[j]/pix
            yt=(ytf[j]+cosT[j]*tp-nyt/2.+1)*pix
            xt=(xtf[j]+sinT[j]*tp-nxt/2.+1)*pix
            plt.plot(yt,xt,lw=widt1[i],color=colr1[i])
        plt.arrow(yt[0], xt[0], yt[99]-yt[0],  xt[99]-xt[0], color=colr1[i],lw=widt1[i],head_width=hwith,zorder=2) 
    
    ax = fig.add_axes([dx1+pro1[1]*dx-dx*0.1, dy1+pro2[1]*dy*dyt, dx, dy*(2.0-dyt)])  
    get_plot_map(plt,flux2,vmax2,vmin2,pix=pix,tit='Velocity shift',lab=r'[km\ s^{-1}]',alpha=alpha)
    nxt,nyt=flux2.shape
    slides_v=tools.extract_segment_val(flux2,hdr,pix,reg_dir=reg_dir,reg_name=reg_name)
    if reg_aper is not None:
        plot_apertures(ax,hdr,plt,nxt,nyt,pix,reg_dir=reg_dir,reg_file=reg_aper)
    for i in range(0, len(vals2)):
        cosT,sinT,rtf,ytf,xtf=vals2[i]
        hwith=widt2[i]/5.0*0.25
        for j in range(0, len(cosT)):
            tp=np.arange(0,100)/99.*rtf[j]/pix
            yt=(ytf[j]+cosT[j]*tp-nyt/2.+1)*pix
            xt=(xtf[j]+sinT[j]*tp-nxt/2.+1)*pix
            plt.plot(yt,xt,lw=widt2[i],color=colr2[i])
        plt.arrow(yt[0], xt[0], yt[99]-yt[0],  xt[99]-xt[0], color=colr2[i],lw=widt2[i],head_width=hwith,zorder=2)            
    if DA is not None:
        daf=DA 
    else:
        daf=1.0
        
    lev=np.sqrt(np.arange(0.0,10.0,1.5)+0.008)/np.sqrt(10.008)
    nls=len(slides1)
    cm='cmr.amber'
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[0]*dx+dx*dxR, dy1+pro2[2]*dy*dyt+dy*(nls-i-1)/nls*(dyt*0.8), dx*0.5, dy/(nls)*(dyt*0.8)])
        slide=slides1[i]
        lt,nw=slide.shape
        slide=slide/np.nanmax(slide)  
        ict=plt.imshow(slide,origin='lower',cmap=cm,extent=[wavet1[0],wavet1[len(wavet1)-1],0,lt*pix*daf],aspect='auto',interpolation='bicubic',vmin=0,vmax=1)
        plt.contour(slide,lev,colors='white',linewidths=1.5,extent=[wavet1[0],wavet1[len(wavet1)-1],0,lt*pix*daf],interpolation='bicubic')
        plt.plot([0,10000],[0,0],lw=5,color='white')
        for j in range(0, len(lamdsA)):
            plt.plot([lamdsA[j],lamdsA[j]],[0,lt*pix*daf],lw=5,ls='--',color='blue')
        plt.xlim(wavet1[0],wavet1[nw-1])
        if DA is not None:
            plt.ylim(0.0001,lt*pix*daf)
            plt.ylabel(r'$R\ [kpc]$',fontsize=18)
        else:
            plt.ylim(0.0001,lt*pix)
            plt.ylabel(r'$R\ [arcsec]$',fontsize=18)
        plt.text(0.05,0.35,namet1[i],fontsize=20,transform=ax.transAxes,color=colr1[i],weight='bold')
        if i < nls-1:
            ax.set_xlabel('').set_visible(False)
            plt.setp( ax.get_xticklabels(), visible=False)           
        else:
            plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=18)
        if i == 0:
            dtp=-5
            for j in range(0, len(lamdsA)):
                x_la_t1=lamdsA[j]+dtp
                y_la_t1=lt*pix*1.05*daf
                plt.text(x_la_t1, y_la_t1, nlinsA[j % len(nlinsA)] , fontsize=18, va='center',color='black',weight='bold')     
    
    nls=len(slides2)
    cm='cmr.amber'
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[0]*dx+dx*(dxR+0.5), dy1+pro2[2]*dy*dyt+dy*(nls-i-1)/nls*(dyt*0.8), dx*0.5, dy/(nls)*(dyt*0.8)])
        slide=slides2[i]
        lt,nw=slide.shape
        slide=slide/np.nanmax(slide)  
        ict=plt.imshow(slide,origin='lower',cmap=cm,extent=[wavet2[0],wavet2[len(wavet2)-1],0,lt*pix*daf],aspect='auto',interpolation='bicubic',vmin=0,vmax=1)
        plt.contour(slide,lev,colors='white',linewidths=1.5,extent=[wavet2[0],wavet2[len(wavet2)-1],0,lt*pix*daf],interpolation='bicubic')
        plt.plot([0,10000],[0,0],lw=5,color='white')
        for j in range(0, len(lamdsB)):
            plt.plot([lamdsB[j],lamdsB[j]],[0,lt*pix*daf],lw=5,ls='--',color='blue')
        plt.xlim(wavet2[0],wavet2[nw-1])
        if DA is not None:
            plt.ylim(0.0001,lt*pix*daf)
            plt.ylabel(r'$R\ [kpc]$',fontsize=18)
        else:
            plt.ylim(0.0001,lt*pix)
            plt.ylabel(r'$R\ [arcsec]$',fontsize=18)
        plt.text(0.05,0.35,namet2[i],fontsize=20,transform=ax.transAxes,color=colr2[i],weight='bold')
        if i < nls-1:
            ax.set_xlabel('').set_visible(False)
            plt.setp( ax.get_xticklabels(), visible=False)           
        else:
            plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=18)
        if i == 0:
            dtp=-5
            for j in range(0, len(lamdsB)):
                x_la_t1=lamdsB[j]+dtp
                y_la_t1=lt*pix*1.05*daf
                plt.text(x_la_t1, y_la_t1, nlinsB[j % len(nlinsB)] , fontsize=18, va='center',color='black',weight='bold')    

    '''
    fl=0.06 # factor to give the label spaces between the plots
    fx=1-fl*(nls-1)
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[1]*dx*2.0*i/nls*fx+dx*2.0*fl*i, dy1+pro2[2]*dy, dx*2.0*fx/nls, dy*dyt*0.8])
    
        vel_vec=slides_v[i]
        xtp=np.arange(0, len(vel_vec))*pix
        plt.plot(xtp*daf,vel_vec,lw=4,color=colr[i])
        plt.scatter(xtp*daf,vel_vec,s=63,color=colr[i])
        if DA is not None:
            plt.xlabel(r'$R\ [kpc]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec)-1)*pix*DA)
        else:
            plt.xlabel(r'$R\ [arcsec]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec)-1)*pix)
        plt.ylabel(r'$Velocity\ shift\ [km\ s^{-1}]$',fontsize=18)
        plt.text(0.8,0.8,namet[i],fontsize=20,transform=ax.transAxes,color=colr[i],weight='bold') 
    if fitmod:
        nt=np.isfinite(vel_vec)
        xtp=xtp[nt]
        vel_vec=vel_vec[nt]
        xta=np.arange(0,1000)*np.nanmax(xtp)/1000.
        if model == 'vmax':
            vo,ro,vc,k=-45,5.2,100,1
            popt, pcov = curve_fit(vmax_func, xtp*daf, vel_vec, p0=[vo, ro, vc, k])
            perr = np.sqrt(np.diag(pcov))
            print('vo=',popt[0],'+-',perr[0],'ro=',popt[1],'+-',perr[1],'vc=',popt[2],'+-',perr[2],'k=',popt[3],'+-',perr[3],'gm=1') 
            print('V_max=',vmax_func(1000+popt[0],popt[0],popt[1],popt[2],popt[3]))
            yfit=vmax_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)
        if model == 'helic':
            alpha,beta,gama,theta=64,-3,-11,-60
            popt, pcov = curve_fit(helic_func, xtp*daf, vel_vec, p0=[alpha,beta,gama,theta])
            perr = np.sqrt(np.diag(pcov))
            print('alpha=',popt[0],'+-',perr[0],'beta=',popt[1],'+-',perr[1],'gamma=',popt[2],'+-',perr[2],'theta=',popt[3],'+-',perr[3]) 
            yfit=helic_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)
        if model == 'sin':
            alpha,beta,gama,theta=30,3.5,np.pi/2,-60
            popt, pcov = curve_fit(sin_func, xtp*daf, vel_vec, p0=[alpha,beta,gama,theta])
            perr = np.sqrt(np.diag(pcov))
            print('alpha=',popt[0],'+-',perr[0],'beta=',popt[1],'+-',perr[1],'gamma=',popt[2],'+-',perr[2],'theta=',popt[3],'+-',perr[3]) 
            yfit=sin_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)    
    
    '''
    if savef:        
        plt.savefig(dir+'/'+titf+'.pdf')
    else:
        plt.show()


def plot_velana(titf,vals_map,path='',DA=None,model='helic',alpha=1.0,fitmod=False,file0='J102700+174900_Gas.fits.gz',nlins=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],lamds=[6549.859,6564.632,6585.278],hdu=0,reg_dir='',reg_aper='apertu.reg',reg_name='paths_J1027_C.reg',zt=0,facs=1,lA1=6520.0,lA2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],av=[0.10,0.03,0.09,0.03],sigT=2,loc=3,facx=0.8,facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,x_max=1,txt_size=18,ylabel='y-value',xlabel='x-value',dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    slides,wavet,dpix,vals,hdr,colr,widt,namet=tools.extract_segment(file0,path=path,reg_dir=reg_dir,reg_name=reg_name,z=zt,lA1=lA1,lA2=lA2,sigT=sigT,cosmetic=True,hdu=hdu)
    pix=dpix
    
    if facy == -1:
        facy=facx
    dx1=av[0]/facx
    dx2=av[1]/facx
    dy1=av[2]/facy
    dy2=av[3]/facy
    dx=(1.0-(dx1+dx2))/float(1.0)
    dy=(1.0-(dy1+dy2))/float(1.0)
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/float(nx)
    dy=(1.0-(dy1+dy2))/float(ny)
    xfi=6*nx*facx*facs#6
    yfi=6*ny*facy#5.5
    fig = plt.figure(figsize=(xfi,yfi))
    dyt=0.85
    ax = fig.add_axes([dx1+pro1[0]*dx-dx*0.1, dy1+pro2[0]*dy*dyt, dx, dy*(2.0-dyt)])  
    flux,vmax,vmin=vals_map
    get_plot_map(plt,flux,vmax,vmin,pix=pix,tit='Velocity shift',lab=r'[km\ s^{-1}]',alpha=alpha)
    nxt,nyt=flux.shape
    slides_v=tools.extract_segment_val(flux,hdr,pix,reg_dir=reg_dir,reg_name=reg_name)
    if reg_aper is not None:
        plot_apertures(ax,hdr,plt,nxt,nyt,pix,reg_dir=reg_dir,reg_file=reg_aper)
    #for i in range(0, len(vals)):
    #    cosT,sinT,rtf,ytf,xtf=vals[i]
    #    tp=np.arange(0,100)/99.*rtf/pix
    #    yt=(ytf+cosT*tp-nyt/2.+1)*pix
    #    xt=(xtf+sinT*tp-nxt/2.+1)*pix
    #    plt.plot(yt,xt,lw=10,color="green")
    for i in range(0, len(vals)):
        cosT,sinT,rtf,ytf,xtf=vals[i]
        hwith=widt[i]/5.0*0.25
        for j in range(0, len(cosT)):
            tp=np.arange(0,100)/99.*rtf[j]/pix
            yt=(ytf[j]+cosT[j]*tp-nyt/2.+1)*pix
            xt=(xtf[j]+sinT[j]*tp-nxt/2.+1)*pix
            plt.plot(yt,xt,lw=widt[i],color=colr[i])
        plt.arrow(yt[0], xt[0], yt[99]-yt[0],  xt[99]-xt[0], color=colr[i],lw=widt[i],head_width=hwith,zorder=2)    
        
    if DA is not None:
        daf=DA 
    else:
        daf=1.0
    lev=np.sqrt(np.arange(0.0,10.0,1.5)+0.008)/np.sqrt(10.008)
    nls=len(slides)
    cm='cmr.amber'
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[1]*dx+dx*dxR, dy1+pro2[1]*dy*dyt+dy*(nls-i-1)/nls*(2.0-dyt), dx, dy/(nls)*(2.0-dyt)])
        slide=slides[i]
        lt,nw=slide.shape
        slide=slide/np.nanmax(slide)  
        ict=plt.imshow(slide,origin='lower',cmap=cm,extent=[wavet[0],wavet[len(wavet)-1],0,lt*pix*daf],aspect='auto',interpolation='bicubic',vmin=0,vmax=1)
        plt.contour(slide,lev,colors='white',linewidths=1.5,extent=[wavet[0],wavet[len(wavet)-1],0,lt*pix*daf],interpolation='bicubic')
        plt.plot([0,10000],[0,0],lw=5,color='white')
        for j in range(0, len(lamds)):
            plt.plot([lamds[j],lamds[j]],[0,lt*pix*daf],lw=5,ls='--',color='blue')
        plt.xlim(wavet[0],wavet[nw-1])
        if DA is not None:
            plt.ylim(0.0001,lt*pix*daf)
            plt.ylabel(r'$R\ [kpc]$',fontsize=18)
        else:
            plt.ylim(0.0001,lt*pix)
            plt.ylabel(r'$R\ [arcsec]$',fontsize=18)
        plt.text(0.05,0.35,namet[i],fontsize=20,transform=ax.transAxes,color=colr[i],weight='bold')
        if i < nls-1:
            ax.set_xlabel('').set_visible(False)
            plt.setp( ax.get_xticklabels(), visible=False)           
        else:
            plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=18)
        if i == 0:
            dtp=-5
            for j in range(0, len(lamds)):
                x_la_t1=lamds[j]+dtp
                y_la_t1=lt*pix*1.05*daf
                plt.text(x_la_t1, y_la_t1, nlins[j % len(nlins)] , fontsize=18, va='center',color='black',weight='bold')     
    
    fl=0.06 # factor to give the label spaces between the plots
    fx=1-fl*(nls-1)
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[1]*dx*2.0*i/nls*fx+dx*2.0*fl*i, dy1+pro2[2]*dy, dx*2.0*fx/nls, dy*dyt*0.8])
    
        vel_vec=slides_v[i]
        xtp=np.arange(0, len(vel_vec))*pix
        plt.plot(xtp*daf,vel_vec,lw=4,color=colr[i])
        plt.scatter(xtp*daf,vel_vec,s=63,color=colr[i])
        if DA is not None:
            plt.xlabel(r'$R\ [kpc]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec)-1)*pix*DA)
        else:
            plt.xlabel(r'$R\ [arcsec]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec)-1)*pix)
        plt.ylabel(r'$Velocity\ shift\ [km\ s^{-1}]$',fontsize=18)
        plt.text(0.8,0.8,namet[i],fontsize=20,transform=ax.transAxes,color=colr[i],weight='bold') 
    if fitmod:
        nt=np.isfinite(vel_vec)
        xtp=xtp[nt]
        vel_vec=vel_vec[nt]
        xta=np.arange(0,1000)*np.nanmax(xtp)/1000.
        if model == 'vmax':
            vo,ro,vc,k=-45,5.2,100,1
            popt, pcov = curve_fit(vmax_func, xtp*daf, vel_vec, p0=[vo, ro, vc, k])
            perr = np.sqrt(np.diag(pcov))
            print('vo=',popt[0],'+-',perr[0],'ro=',popt[1],'+-',perr[1],'vc=',popt[2],'+-',perr[2],'k=',popt[3],'+-',perr[3],'gm=1') 
            print('V_max=',vmax_func(1000+popt[0],popt[0],popt[1],popt[2],popt[3]))
            yfit=vmax_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)
        if model == 'helic':
            alpha,beta,gama,theta=64,-3,-11,-60
            popt, pcov = curve_fit(helic_func, xtp*daf, vel_vec, p0=[alpha,beta,gama,theta])
            perr = np.sqrt(np.diag(pcov))
            print('alpha=',popt[0],'+-',perr[0],'beta=',popt[1],'+-',perr[1],'gamma=',popt[2],'+-',perr[2],'theta=',popt[3],'+-',perr[3]) 
            yfit=helic_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)
        if model == 'sin':
            alpha,beta,gama,theta=30,3.5,np.pi/2,-60
            popt, pcov = curve_fit(sin_func, xtp*daf, vel_vec, p0=[alpha,beta,gama,theta])
            perr = np.sqrt(np.diag(pcov))
            print('alpha=',popt[0],'+-',perr[0],'beta=',popt[1],'+-',perr[1],'gamma=',popt[2],'+-',perr[2],'theta=',popt[3],'+-',perr[3]) 
            yfit=sin_func(xta*daf,popt[0],popt[1],popt[2],popt[3]) 
            plt.plot(xta*daf,yfit,color='black',lw=3)    
    if savef:        
        plt.savefig(dir+'/'+titf+'.pdf')
    else:
        plt.show()

def helic_func(r,alpha,beta,gama,theta):
    vr=alpha*r**(1/2)*np.sin(beta*r**(1/2)+gama)+theta
    return vr

def sin_func(r,alpha,beta,gama,theta):
    vr=alpha*np.sin(r/beta*np.pi+gama)+theta
    return vr    

def plot_bpt_map(file,name='',alpha=1,orientation=None,hd=0,ewsing=1,max_typ=5,location=None,savef=False,fig_path='',fwcs=False,scale=0,facp=0.8,tit='BPT',cont=False,path='',indEwHa=769,indOIII=76,indNII=123,indHa=124,indHb=63,ret=1,agn=5,sf=3,inte=2,comp=4):
    basefigname='BPT_map_NAME'
    [data,hdr]=fits.getdata(path+'/'+file, hd, header=True)
    try:
        dx=np.sqrt((hdr['CD1_1'])**2.0+(hdr['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr['CD2_1'])**2.0+(hdr['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr['CD1_1']*3600.0
            dy=hdr['CD2_2']*3600.0
        except:
            try:
                dx=hdr['PC1_1']*3600.
                dy=hdr['PC2_2']*3600.
            except:
                dx=hdr['CDELT1']*3600.
                dy=hdr['CDELT2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0 
    fluxOIII=data[indOIII,:,:]
    fluxNII=data[indNII,:,:]
    fluxHa=data[indHa,:,:]
    fluxHb=data[indHb,:,:]
    ewHa=ewsing*data[indEwHa,:,:]

    ratio1=np.log10(fluxOIII/fluxHb)
    ratio2=np.log10(fluxNII/fluxHa)
    bounds = np.arange(0, max_typ + 1) + 0.5  # Para centrar los ticks
    map_bpt=tools.bpt(ewHa,ratio2,ratio1,ret=ret,agn=agn,sf=sf,inte=inte,comp=comp)
    
    type_p=r'log($[OIII]H\beta$)~vs~log($[NII]H\alpha$)'
    type_n=r'log($[OIII]/H\beta$) vs log($[NII]/H\alpha$)'
    vmax=None
    vmin=None
    ticks = [1,2,3,4,5]
    labels = ['Ret','Int','SF','Comp','sAGN']
    colores = ['orange','dodgerblue','mediumspringgreen','#A788CF','darkslateblue']

    plt.rcParams['figure.figsize'] = [6.5*facp, 7.6*facp]
    if fwcs:
        wcs = WCS(hdr).celestial
        plt.subplot(projection=wcs)
        try:
            objsys=hdr['RADESYS']
        except:
            objsys='J2000'
    else:
        objsys='J2000'

    cm = ListedColormap(colores)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cm.N)#niveles, len(colores))
    get_plot_map(plt,map_bpt,vmax,vmin,cmt=cm,ticks=ticks,labels=labels,norm=norm,fwcs=fwcs,objsys=objsys,pix=pix,tit=tit,scale=scale,lab=type_n,cont=cont,orientation=orientation,location=location,alpha=alpha)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
        plt.tight_layout()
    else:
        plt.show()


def plot_single_map(file,valmax,valmin,name='',scale=0,sb=False,fwcs=False,logs=False,zerofil=False,valz=None,scalef=1.0,basefigname='Ha_vel_map_NAME',sumc=False,path='',hd=0,indx=0,indx2=None,tit='',lab='',facp=0.8,facx=6.5,facy=7.6,cont=False,alpha=1,orientation=None,location=None,savef=False,fig_path=''):

    [data,hdr]=fits.getdata(path+'/'+file, hd, header=True)
    try:
        dx=np.sqrt((hdr['CD1_1'])**2.0+(hdr['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr['CD2_1'])**2.0+(hdr['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr['CD1_1']*3600.0
            dy=hdr['CD2_2']*3600.0
        except:
            try:
                dx=hdr['CDELT1']*3600.
                dy=hdr['CDELT2']*3600.
            except:
                dx=hdr['PC1_1']*3600.
                dy=hdr['PC2_2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0
    if sumc:
        try:
            map_val=np.nansum(data[indx,:,:],axis=0)*scalef
        except:
            print('It is not possible to integrate the data cube within the indexes provided, we will integrate all the cube')
            map_val=np.nansum(data,axis=0)*scalef
    else:
        map_val=data[indx,:,:]*scalef
    if indx2 != None:
        val2=data[indx2,:,:]*scalef
        map_val=map_val/val2
    if zerofil:
        if valz == None:
            map_val[np.where(map_val == 0)]=np.nan
        else:
            map_val[np.where(map_val <= valz)]=np.nan
    if sb:
        map_val=map_val/pix**2
    if logs:
        map_val=np.log10(map_val)
    
    plt.rcParams['figure.figsize'] = [facx*facp, facy*facp]
    if fwcs:
        wcs = WCS(hdr).celestial
        plt.subplot(projection=wcs)
        try:
            objsys=hdr['RADESYS']
        except:
            objsys='J2000'
    else:
        objsys='J2000'
    get_plot_map(plt,map_val,valmax,valmin,fwcs=fwcs,objsys=objsys,pix=pix,tit=tit,scale=scale,lab=lab,cont=cont,orientation=orientation,location=location)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.tight_layout()
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
    else:
        plt.show()

def get_plot_map(plt,flux,vmax,vmin,pix=0.2,scale=0,ticks=None,labels=None,cmt=None,norm=None,fwcs=False,objsys='J2000',tit='flux',lab='[10^{-16}erg/s/cm^2/arcsec^2]',cont=False,alpha=1,orientation=None,location=None):
    nx,ny=flux.shape
    if fwcs:
        pix=3600.
        scale=2
        xlab=r'\alpha\ '
        ylab=r'\delta\ '
        dx=ny*pix/2.0
        dy=nx*pix/2.0
    else:
        xlab=r'\Delta \alpha\ '
        ylab=r'\Delta \delta\ '
        dx=0.0
        dy=0.0
    if scale == 0:
        fac=1
        labs='[arcsec]'
    elif scale == 1:
        fac=60
        labs='[arcmin]'
    elif scale == 2:
        fac=3600
        if fwcs:
            labs='['+objsys+']'
        else:
            labs='[arcdeg]'
    else:
        fac=1
        labs='[arcsec]'
    dx=dx/fac
    dy=dy/fac
    if cont:
        max_f=vmax-(vmax-vmin)*0.05
        min_f=vmin+(vmax-vmin)*0.05
        n_b=15
        flux_range=(np.arange(0,n_b)/float(n_b-1))*(max_f-min_f)+min_f    
        lev=flux_range
    if cmt is not None:
        cm=cmt
    else: 
        cm=plt.get_cmap('jet')
    if location != 'top':
        plt.title(r'$'+tit+'$',fontsize=18)
    plt.xlabel(r'$'+xlab+labs+'$',fontsize=18)
    plt.ylabel(r'$'+ylab+labs+'$',fontsize=18)
    ict=plt.imshow(flux,cmap=cm,norm=norm,origin='lower',extent=[-ny*pix/2./fac+dx,ny*pix/2./fac+dx,-nx*pix/2./fac+dy,nx*pix/2./fac+dy],vmax=vmax,vmin=vmin,alpha=alpha)#,norm=LogNorm(0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    if cont:
        plt.contour(flux,lev,colors='black',linewidths=2,extent=[-ny*pix/2./fac+dx,ny*pix/2./fac+dx,-nx*pix/2./fac+dy,nx*pix/2./fac+dy],zorder=1)
    if ticks is not None:
        cbar=plt.colorbar(ict,orientation=orientation,location=location,ticks = ticks,pad=0.01)
    else:
        cbar=plt.colorbar(ict,orientation=orientation,location=location)
    plt.xlim(-ny*pix/2/fac+dx,ny*pix/2/fac+dx)
    plt.ylim(-nx*pix/2/fac+dy,nx*pix/2/fac+dy)  
    if location == 'top':
        cbar.set_label(r"$"+tit+r"\ "+lab+"$",fontsize=18)
    else:
        cbar.set_label(r"$"+lab+"$",fontsize=18)  
    if labels is not None:
        cbar.set_ticklabels(labels)       

def get_plot(flux,savef=True,pix=0.2,name='Residual',tit='flux',outs=[],title=None,cbtr=True,bpte=False,maxmin=[],ewp=False):
    nx,ny=flux.shape
    if len(outs) > 0:
        aptr=True
        ypos0=(outs[0]-nx/2.0)*pix
        ypos1=(outs[1]-nx/2.0)*pix
        xpos0=(outs[2]-ny/2.0)*pix
        xpos1=(outs[3]-ny/2.0)*pix
        cpos='red'#outs[4]
        lpos=outs[5]
    else:
        aptr=False
    if not bpte:
        flux=flux/pix**2.0
    At=np.nanmax(flux)#flux[x_m,y_m]
    max_f=At
    min_f=At*0.005#6.5#3.8#1.5#3#*0.01#6.5#At*0.05#6.5#At*0.1#6.5
    n_b=15#25#15
    flux_range=10**((np.arange(0,n_b)/float(n_b)+0.02)*(np.log10(max_f)-np.log10(min_f))+np.log10(min_f))    
    
    
    lev=flux_range
    cm=plt.cm.get_cmap('jet')
    if savef:
        fig, ax = plt.subplots(figsize=(6.8*1.2,5.5*1.2))
    else:
        fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
    plt.title(title,fontsize=18)
    plt.xlabel(r'$\Delta \alpha\ [arcsec]$',fontsize=18)
    plt.ylabel(r'$\Delta \delta\ [arcsec]$',fontsize=18)
    if not bpte:
        if len(maxmin) > 0:
            mav=maxmin[1]
            miv=maxmin[0]
        else:
            mav=100
            miv=1
        ict=plt.imshow(flux,cmap=cm,origin='lower',extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],norm=LogNorm(miv,mav))#0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    else:
        if len(maxmin) > 0:
            mav=maxmin[1]
            miv=maxmin[0]
        else:
            mav=4
            miv=1
        ict=plt.imshow(flux,cmap=cm,origin='lower',extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],vmax=mav,vmin=miv)#0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    if cbtr:
        cbar=plt.colorbar(ict)
    if aptr:
        for kk in range(0, len(xpos0)):
            if not 'T' in lpos[kk]:# and not 'K' in lpos[kk]:
                plt.plot([xpos0[kk],xpos1[kk]],[ypos0[kk],ypos0[kk]],lw=2,color=cpos)
                plt.plot([xpos0[kk],xpos1[kk]],[ypos1[kk],ypos1[kk]],lw=2,color=cpos)
                plt.plot([xpos0[kk],xpos0[kk]],[ypos0[kk],ypos1[kk]],lw=2,color=cpos)
                plt.plot([xpos1[kk],xpos1[kk]],[ypos0[kk],ypos1[kk]],lw=2,color=cpos)
                if not 'E' in lpos[kk] and not 'M' in lpos[kk] and not 'G' in lpos[kk] and not 'K' in lpos[kk] and not 'F' in lpos[kk]: 
                    plt.text((xpos0[kk]+xpos1[kk])/2.0-2,ypos1[kk]+(ypos1[0]-ypos0[0])*0.09,lpos[kk], fontsize=18,color=cpos)
                else:
                    if 'E' in lpos[kk] or 'F' in lpos[kk]:
                        dxt=+1
                    else:
                        dxt=-(xpos0[kk]-xpos1[kk])-5
                    plt.text(xpos1[kk]+(xpos0[kk]-xpos1[kk])+dxt,(ypos1[kk]+ypos0[kk])/2.0-1,lpos[kk], fontsize=18,color=cpos)    
    
    plt.xlim(-ny*pix/2,ny*pix/2)
    plt.ylim(-nx*pix/2,nx*pix/2)
    if cbtr:    
        if not ewp:
            cbar.set_label(r"$"+tit+r"\ [10^{-16}erg/s/cm^2/arcsec^2]$",fontsize=18)
        else:
            cbar.set_label(r"$"+tit+"$",fontsize=18)
    fig.tight_layout()
    if savef:
        plt.savefig(name+'_map.pdf')
    else:
        plt.show()   


def plot_apertures(ax,hdr,plt,nx,ny,dpix,reg_dir='./',reg_file='test.reg'):
    file=reg_dir+reg_file
    ra,dec,rad,l1,l2,th,colr,namet,typ=tools.get_apertures(file)
    for i in range(0, len(ra)):
        sky1=SkyCoord(ra[i]+' '+dec[i],frame=FK5, unit=(u.hourangle,u.deg))
        wcs = WCS(hdr)
        wcs=wcs.celestial
        ypos,xpos=skycoord_to_pixel(sky1,wcs)
        xposf=(xpos-nx/2.0+1)*dpix
        yposf=(ypos-ny/2.0+1)*dpix
        c = Circle((yposf, xposf), rad[i], edgecolor=colr[i], facecolor='none',lw=5,zorder=3)
        ax.add_patch(c)
        if namet[i] == '1':
            plt.text(yposf+dpix*0.5,xposf-dpix*2,namet[i], fontsize=25,color=colr[i],weight='bold')
        else:
            plt.text(yposf+dpix*0.5,xposf,namet[i], fontsize=25,color=colr[i],weight='bold')

def plot_circle(ax,xpos,ypos,nx,ny,dpix,rad=2,color='black',name='1',dtex=0,dtey=0):
    xposf=(xpos-nx/2.0+1)*dpix
    yposf=(ypos-ny/2.0+1)*dpix
    c = Circle((yposf, xposf), rad, edgecolor=color, facecolor='none',lw=5,zorder=3)
    ax.add_patch(c)
    #if name == '1':
    #    plt.text(yposf+dpix*0.5+dtey,xposf-dpix*2+dtex,name, fontsize=25,color=color,weight='bold')
    #else:
    plt.text(yposf+dpix*0.5+dtey,xposf+dtex,name, fontsize=25,color=color,weight='bold')    

def plot_outputfits(wave_i,fluxt,fluxtE,model,modsI,n_lines,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,names0,vals,valsL,samples,errp=True,fontsize=14,colors=['blue','red','purple','brown','pink'],name_out='name',dir_out='',res_norm=True,labplot=True,dataFe=None,lorentz=False,skew=False,outflow=False,powlaw=False,feii=False):
    facy=1
    facx=1
    xf=7
    yf=5
    nx=1
    ny=1
    dx1=0.12/facx
    dx2=0.02/facx
    dy1=0.13/facy
    dy2=0.07/facy
    dx=(1.0-(dx1+dx2))/1.0
    dy=(1.0-(dy1+dy2))/1.0
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/nx
    dy=(1.0-(dy1+dy2))/ny
    xfi=xf*nx*facx
    yfi=yf*ny*facy
    fig = plt.figure(figsize=(xfi,yfi))
    ax1 = fig.add_axes([dx1, dy1+0.2*dy, dx, dy*0.8])
    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
    ax1.plot(wave_i,fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
    if powlaw:
        contm=modsI[n_lines]
    else:
        contm=0
    if feii:
        contm=contm+modsI[n_lines+1]
    #ax1.plot(wave_i,fluxt-model-np.nanmax(fluxt)*0.25,linewidth=1,color='olive',label=r'Residual')                  
    for namel in names0:
        if namel != 'None':
            indl=names0.index(namel)
            ax1.plot(wave_i,contm+modsI[indl],linewidth=1,label=namel,color=colors[indl % len(colors)])
    if outflow:
        ct1a=0
        for namel in names0:
            if namel != 'None':
                indl=names0.index(namel)
                if ct1a == 0:
                    ax1.plot(wave_i,modsI[indl+n_lines],linewidth=1,color='orange',label=r'Outflow')
                else:
                    ax1.plot(wave_i,modsI[indl+n_lines],linewidth=1,color='orange')
                ct1a=ct1a+1
    if powlaw:
        ax1.plot(wave_i,modsI[n_lines],linewidth=1,color='orange',label=r'PowerLaw')
    if feii:
        ax1.plot(wave_i,modsI[n_lines+1],linewidth=1,color='red',label=r'FeII')         
    if len(names0) < 5:
        fontsizeL=14
    elif len(names0) < 10:
        fontsizeL=12
    elif len(names0) < 15:
        fontsizeL=10
    else:
        fontsizeL=6
    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
    ax1.set_xlabel(r'$Wavelength\ [\rm{\AA}]$',fontsize=fontsize)
    ax1.set_ylabel(r'Flux [10$^{-16}$erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]',fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.setp( ax1.get_xticklabels(), visible=False)
    if labplot:
        ax1.legend(fontsize=fontsizeL)
    #plt.tight_layout()
    ax1 = fig.add_axes([dx1, dy1, dx, dy*0.2])
    if res_norm:
        ax1.plot(wave_i,(fluxt-model)/fluxt*100,linewidth=1,color='olive',label=r'Residual')
        if errp:
            ax1.plot(wave_i,fluxtE/fluxt*100,linewidth=1,color='grey',label=r'$1\sigma$ Error')
            ax1.plot(wave_i,-fluxtE/fluxt*100,linewidth=1,color='grey',label=r'$1\sigma$ Error')
    else:
        ax1.plot(wave_i,(fluxt-model),linewidth=1,color='olive',label=r'Residual')
        if errp:
            ax1.plot(wave_i,fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
            ax1.plot(wave_i,-fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
    ax1.plot(wave_i,wave_i*0,linestyle='--',color='black',lw=1)
    if res_norm:
        ax1.set_ylim(-12,12)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_xlabel(r'$Wavelength\ [\rm{\AA}]$',fontsize=fontsize)
    if res_norm:
        ax1.set_ylabel(r'Res [$\%$]',fontsize=12)
    else:
        ax1.set_ylabel(r'Res',fontsize=12)
    fig.savefig(dir_out+'spectraFit_NAME.pdf'.replace('NAME',name_out))
    plt.show()

    if skew:
        labels2 = [*valsL,r'$\alpha_n$',r'$\alpha_b$']
    else:
        if outflow:
            labels2 = [*valsL,r'$F_{out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$\alpha_{out}$']
        else:
            labels2 = valsL
    if powlaw:
        if feii:
            labels2 = [*valsL,r'$P_1$',r'$P_2$',r'$\sigma_{FeII}$',r'$\Delta\lambda_{FeII}$',r'$A_{FeII}$']
        else:
            labels2 = [*valsL,r'$P_1$',r'$P_2$']
                               
    fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
    fig.savefig(dir_out+'corners_NAME.pdf'.replace('NAME',name_out))
                
                    
    med_model, spread = mcm.sample_walkers(10, samples, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, skew=skew, lorentz=lorentz, outflow=outflow, powlaw=powlaw, feii=feii, data=dataFe)
                    
    fig = plt.figure(figsize=(6*1.5,3*1.5))
    ax1 = fig.add_subplot(1,1,1)
    #ax1.set_xlim(lA1,lA2)
    ax1.plot(wave_i,fluxt,label='Input spectrum')
    ax1.plot(wave_i,model,label='Highest Likelihood Model')
    plt.ylabel(r'$Flux\ [10^{-16} erg/s/cm^2/\AA]$',fontsize=16)
    plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=16)
    ax1.fill_between(wave_i,med_model-spread*10,med_model+spread*10,color='grey',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
    ax1.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(dir_out+'spectra_mod_NAME.pdf'.replace('NAME',name_out))