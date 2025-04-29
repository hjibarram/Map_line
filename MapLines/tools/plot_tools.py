#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_plot_map(plt,flux,vmax,vmin,pix=0.2,tit='flux',lab='[10^{-16}erg/s/cm^2/arcsec^2]',cont=False,alpha=1):
    nx,ny=flux.shape
    if cont:
        max_f=vmax-(vmax-vmin)*0.05
        min_f=vmin+(vmax-vmin)*0.05
        n_b=15
        flux_range=(np.arange(0,n_b)/np.float(n_b-1))*(max_f-min_f)+min_f    
        lev=flux_range
    cm=plt.cm.get_cmap('jet')
    plt.title(r'$'+tit+'$',fontsize=18)
    plt.xlabel(r'$\Delta \alpha\ [arcsec]$',fontsize=18)
    plt.ylabel(r'$\Delta \delta\ [arcsec]$',fontsize=18)
    ict=plt.imshow(flux,cmap=cm,origin='lower',extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],vmax=vmax,vmin=vmin,alpha=alpha)#,norm=LogNorm(0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    if cont:
        plt.contour(flux,lev,colors='black',linewidths=2,extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],zorder=1)
    cbar=plt.colorbar(ict)
    plt.xlim(-ny*pix/2,ny*pix/2)
    plt.ylim(-nx*pix/2,nx*pix/2)    
    cbar.set_label(r"$"+lab+"$",fontsize=18)      

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