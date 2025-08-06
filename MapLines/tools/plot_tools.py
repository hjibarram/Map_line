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


def plot_single_map(file,valmax,valmin,name='',scale=0,sb=False,fwcs=False,logs=False,zerofil=False,valz=None,scalef=1.0,basefigname='Ha_vel_map_NAME',path='',hd=0,indx=0,indx2=None,tit='',lab='',facp=0.8,cont=False,alpha=1,orientation=None,location=None,savef=False,fig_path=''):

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
    get_plot_map(plt,map_val,valmax,valmin,fwcs=fwcs,objsys=objsys,pix=pix,tit=tit,scale=scale,lab=lab,cont=cont,orientation=orientation,location=location)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
        plt.tight_layout()
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
            cbar.set_label(r"$"+tit+"\ [10^{-16}erg/s/cm^2/arcsec^2]$",fontsize=18)
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
    if name == '1':
        plt.text(yposf+dpix*0.5+dtey,xposf-dpix*2+dtex,name, fontsize=25,color=color,weight='bold')
    else:
        plt.text(yposf+dpix*0.5+dtey,xposf+dtex,name, fontsize=25,color=color,weight='bold')    

def plot_outputfits(wave_i,fluxt,fluxtE,model,modsI,n_lines,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,names0,vals,valsL,samples,fontsize=14,colors=['blue','red','purple','brown','pink'],name_out='name',dir_out='',labplot=True,dataFe=None,lorentz=False,skew=False,outflow=False,powlaw=False,feii=False):
    

    facy=1
    facx=1
    xf=7
    yf=5
    nx=1
    ny=1
    dx1=0.12/facx
    dx2=0.02/facx
    dy1=0.13/facy
    dy2=0.08/facy
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
    #ax1.plot(wave_i,fluxt-model-np.nanmax(fluxt)*0.25,linewidth=1,color='olive',label=r'Residual')                  
    for namel in names0:
        if namel != 'None':
            indl=names0.index(namel)
            ax1.plot(wave_i,modsI[indl],linewidth=1,label=namel,color=colors[indl % len(colors)])
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
    else:
        fontsizeL=10
    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
    ax1.set_xlabel(r'$Wavelength\ [\rm{\AA}]$',fontsize=fontsize)
    ax1.set_ylabel(r'Flux [10$^{-16}$erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]',fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.setp( ax1.get_xticklabels(), visible=False)
    if labplot:
        ax1.legend(fontsize=fontsizeL)
    #plt.tight_layout()
    ax1 = fig.add_axes([dx1, dy1, dx, dy*0.2])
    ax1.plot(wave_i,(fluxt-model)/fluxt*100,linewidth=1,color='olive',label=r'Residual')
    #ax1.plot(wave_i,fluxtE/fluxt*100,linewidth=1,color='grey',label=r'$1\sigma$ Error')
    #ax1.plot(wave_i,-fluxtE/fluxt*100,linewidth=1,color='grey',label=r'$1\sigma$ Error')
    #plt.plot(wave_qsrt_mock,np.abs(spec_qsrt_mock-spec_mod)/spec_in*100,linestyle='-',color='blue' ,label='SDSS spec',lw=1.5,zorder=1)
    ax1.plot(wave_i,wave_i*0,linestyle='--',color='black',lw=1)
    #ax.set_xlim(3700/(1+zt),10300/(1+zt))
    ax1.set_ylim(-10,10)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_xlabel(r'$Wavelength\ [\rm{\AA}]$',fontsize=fontsize)
    ax1.set_ylabel(r'Res [$\%$]',fontsize=12)
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
    ax1.fill_between(wave_i,med_model-spread*50,med_model+spread*50,color='grey',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
    ax1.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(dir_out+'spectra_mod_NAME.pdf'.replace('NAME',name_out))