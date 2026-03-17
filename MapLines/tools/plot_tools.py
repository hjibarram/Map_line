#!/usr/bin/env python
"""
MapLines.tools.plot_tools
=========================

Visualization and diagnostic plotting utilities for MapLines.

This module contains the plotting routines used to inspect and present
the outputs of the MapLines spectral fitting workflow. The functions
implemented here are designed to work with both one-dimensional spectra
and two-dimensional parameter maps extracted from integral-field or
imaging spectroscopy products.

The module provides tools for:

- plotting grids of spatial maps derived from fitted data products
- overlaying apertures, circular regions, and pseudo-slit extractions
- visualizing spectra extracted along user-defined paths or apertures
- displaying velocity-structure diagnostics and simple kinematic models
- building and visualizing BPT, WHAN, and WHAD classification maps
- generating diagnostic plots for spectral fits and posterior samples

These routines rely heavily on FITS headers and WCS information to place
apertures and paths consistently on sky-projected maps. They also use
supporting routines from ``MapLines.tools.tools`` for extracting regions,
spectra, and parameter maps, and can use posterior samples from the MCMC
module for fit visualization.

Notes
-----
The plotting functions in this module are primarily intended for
interactive analysis, diagnostic inspection, and the production of
publication-quality figures. In most cases they save figures directly
to PDF files.

See Also
--------
MapLines.tools.tools
    Data extraction utilities used by many plotting routines.
MapLines.tools.line_fit
    Spectral fitting driver that generates many of the products shown
    by these plotting functions.
MapLines.tools.mcmc
    Posterior-sampling utilities used for visualizing uncertainties.
"""
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


def plot_maps_grid(datalist,hdr,fig_path='',tit='',lab='[10^{-16}erg/s/cm^2/arcsec^2]',labelst=['Vaule1','Value2','Value3'],
                   alpha=0.6,fontsizest=20,colorst='black',fontsize=20,basefigname='maps_NAME',sumc=False,scale=0,sb=False,
                   fwcs=False,logs=False,zerofil=False,valz=None,cont=False,maxmin=[],vt='',name='Name',
                   indx=0,indx2=None,cmt=None,scalef=1.0,facs=1,av=[0.15,0.2,0.12,0.03],pro1=[0,0,0],pro2=[0,1,2],
                   orientation=None,location=None,ticks=None,labels=None):
    """
    Plot a grid of 2D maps or slices extracted from 3D cubes.

    This routine arranges multiple maps in a custom multi-panel layout and
    draws a shared color scale. It supports direct plotting of 2D arrays or
    extraction of 2D slices from 3D cubes, optional surface-brightness
    normalization, logarithmic scaling, WCS-aware axes, and panel labels.

    Parameters
    ----------
    datalist : list of ndarray
        List of maps or cubes to display. If an element is 3D, the function
        uses ``indx`` to select a slice or, if ``sumc=True``, integrates
        along the spectral axis.
    hdr : astropy.io.fits.Header
        FITS header used to derive the pixel scale and, optionally, the WCS.
    fig_path : str, optional
        Output directory for the generated PDF.
    tit : str, optional
        Quantity name used in the color-bar label.
    lab : str, optional
        Physical units displayed in the color-bar label.
    labelst : list of str, optional
        Text labels placed on each panel.
    alpha : float, optional
        Transparency used by the map display.
    fontsizest : int, optional
        Font size for panel labels.
    colorst : str, optional
        Color used for panel labels.
    fontsize : int, optional
        Font size for the color-bar label.
    basefigname : str, optional
        Output base file name. The token ``NAME`` is replaced by ``name``.
    sumc : bool, optional
        If True and the inputs are 3D cubes, sum along the spectral axis.
    scale : float, optional
        Display scale passed to the internal map-plotting helper.
    sb : bool, optional
        If True, divide the maps by pixel area to display surface brightness.
    fwcs : bool, optional
        If True, draw axes in WCS coordinates.
    logs : bool, optional
        If True, plot the logarithm of the maps.
    zerofil : bool, optional
        If True, replace zero-valued or low-valued pixels with NaN.
    valz : float, optional
        Threshold used when ``zerofil=True``.
    cont : bool, optional
        Forwarded to the internal plotting helper for contour handling.
    maxmin : list, optional
        Two-element list with explicit ``[vmin, vmax]`` values.
    vt : str, optional
        Reserved text parameter.
    name : str, optional
        Object identifier used in the output file name.
    indx : int, optional
        Index of the plane to extract from 3D cubes.
    indx2 : int, optional
        Secondary index used to divide one slice by another.
    cmt : matplotlib colormap, optional
        Colormap used for the images.
    scalef : float, optional
        Global multiplicative scaling factor.
    facs : float, optional
        Global figure-size scaling factor.
    av : list of float, optional
        Margins used to construct the custom panel layout.
    pro1, pro2 : list of int, optional
        Panel positions in the custom grid.
    orientation : {'vertical', 'horizontal'}, optional
        Color-bar orientation.
    location : str, optional
        Color-bar location.
    ticks : list, optional
        Explicit color-bar tick positions.
    labels : list, optional
        Explicit color-bar tick labels.

    Returns
    -------
    None

    Notes
    -----
    This function writes the figure directly to a PDF file and then displays
    it. The internal map rendering is delegated to ``get_plot_map``.
    """
    n_maps=len(datalist)
    if n_maps == 0:
        print('No maps to plot')
        return    
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
    dims=datalist[0].shape
    mapslist=[]
    if len(dims) == 3:
        if sumc:
            try:
                for it in range(0, n_maps):
                    data=datalist[it]
                    mapslist.extend([np.nansum(data[indx,:,:],axis=0)*scalef])
            except:
                print('It is not possible to integrate the data cubes within the indexes provided, we will integrate all the cube')
                for it in range(0, n_maps):
                    data=datalist[it]
                    mapslist.extend([np.nansum(data,axis=0)*scalef])
        else:
            for it in range(0, n_maps):
                data=datalist[it]
                mapslist.extend([data[indx,:,:]*scalef])
        if indx2 != None:
            for it in range(0, n_maps):
                data=datalist[it]
                valt=data[indx2,:,:]*scalef
                mapslist[it]=mapslist[it]/valt
    elif len(dims) == 2:
        for it in range(0, n_maps):
            data=datalist[it]
            mapslist.extend([data*scalef])
    if zerofil:
        for it in range(0, n_maps):
            valt=mapslist[it]
            if valz == None:
                valt[np.where(valt == 0)]=np.nan
            else:
                valt[np.where(valt <= valz)]=np.nan
            mapslist[it]=valt    
    if sb:
        for it in range(0, n_maps):
            mapslist[it]=mapslist[it]/pix**2
    if logs:
        for it in range(0, n_maps):
            mapslist[it]=np.log10(mapslist[it])
    if fwcs:
        wcs = WCS(hdr).celestial
        plt.subplot(projection=wcs)
        try:
            objsys=hdr['RADESYS']
        except:
            objsys='J2000'
    else:
        objsys='J2000'        

    if len(maxmin) > 0:
        vmax=maxmin[1]
        vmin=maxmin[0]
    else:
        vmax=np.nammax(mapslist[0])*1.1
        vmin=0.001
    facx=0.99
    facy=0.99
    nx=np.nanmax(pro1)+1
    ny=np.nanmax(pro2)+1
    dx1=av[0]/facx
    dx2=av[1]/facx
    dy1=av[2]/facy
    dy2=av[3]/facy
    dx=(1.0-(dx1+dx2))
    dy=(1.0-(dy1+dy2))
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/float(nx)
    dy=(1.0-(dy1+dy2))/float(ny)
    xfi=6*nx*facx*facs
    yfi=6*ny*facy
    fig = plt.figure(figsize=(xfi,yfi))
    for it in range(0, n_maps):
        flux=mapslist[it]
        ax = fig.add_axes([dx1+pro1[it]*dx, dy1+pro2[it]*dy, dx, dy])
        ict=get_plot_map(plt,flux,vmax,vmin,fwcs=fwcs,objsys=objsys,pix=pix,tit=tit,scale=scale,lab=lab,cont=cont,alpha=alpha,orientation=orientation,location=location,cmt=cmt,cbarp=False)
        plt.text(0.05, 0.96, labelst[it % len(labelst)], fontsize=fontsizest, color=colorst, va='center',transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        if pro1[it] != 0:
            ax.set_ylabel('').set_visible(False)
            plt.setp(ax.get_yticklabels(), visible=False)
        if pro2[it] != 0:
            ax.set_xlabel('').set_visible(False)
            plt.setp(ax.get_xticklabels(), visible=False)  
        if pro1[it] == np.nanmax(pro1) or pro2[it] == np.nanmax(pro2):
            if pro1[it] == np.nanmax(pro1) and orientation == 'vertical':
                ax2 = fig.add_axes([dx1+pro1[it]*dx+dx, dy1+pro2[it]*dy, dx*0.05, dy]) 
            elif pro2[it] == np.nanmax(pro2) and orientation == 'horizontal':
                ax2 = fig.add_axes([dx1+pro1[it]*dx, dy1+pro2[it]*dy+dy, dx, dy*0.05]) 
            ax2.tick_params(axis='both',which='major',labelsize=18)
            if ticks is not None:
                cbar=plt.colorbar(ict,cax=ax2,orientation=orientation,location=location,ticks=ticks,pad=0.01)
            else:
                cbar=plt.colorbar(ict,cax=ax2,orientation=orientation,location=location)
            if location == 'top':
                cbar.set_label(r"$"+tit+r"\ "+lab+"$",fontsize=fontsize)
            else:
                cbar.set_label(r"$"+lab+"$",fontsize=fontsize)
            if labels is not None:
                cbar.set_ticklabels(labels)    
    plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
    plt.show()
    plt.close()


def plot_mapapertures(titf,vals_map,nlins=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],titp='Velocity~ shift',lab=r'[km\ s^{-1}]',
    cont=False,alpha=0.5,lamds=[6549.859,6564.632,6585.278],path='',nzeros=False,hdu=0,wcs=None,file0='J102700+174900_Gas.fits.gz',
    reg_dir='',reg_aper='apertu.reg',dtex=0,dtey=0,rad=1.5,cosmetic=False,reg_name='paths_J1027_C.reg',zt=0,facs=1,lA1=6520.0,
    lA2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],av=[0.10,0.03,0.09,0.03],sigT=2,loc=3,facx=0.8,
    facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,x_max=1,txt_size=18,ylabel='y-value',xlabel='x-value',
    dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    """
    Plot a parameter map together with circular apertures and extracted spectra.

    This routine combines a 2D map with aperture or path overlays and a set
    of spectra extracted along user-defined segment nodes. It is useful for
    visually connecting spatial locations in a map with the corresponding
    extracted one-dimensional spectra.

    Parameters
    ----------
    titf : str
        Output figure name.
    vals_map : tuple
        Tuple of the form ``(map, vmax, vmin)`` for the main image panel.
    nlins : list of str, optional
        Labels used for the vertical line markers in the extracted spectra.
    titp : str, optional
        Title of the map quantity.
    lab : str, optional
        Units of the map quantity.
    cont : bool, optional
        Forwarded to the internal plotting helper.
    alpha : float, optional
        Transparency of the map.
    lamds : list of float, optional
        Reference wavelengths used to draw vertical markers in the spectra.
    path : str, optional
        Directory of the input cube.
    nzeros : bool, optional
        If True, negative values may be converted to NaN during extraction.
    hdu : int, optional
        FITS HDU index used for extraction.
    wcs : astropy.wcs.WCS, optional
        WCS object. If not provided, it is built from the cube header.
    file0 : str, optional
        Input cube file used for the extraction.
    reg_dir : str, optional
        Directory containing DS9 region files.
    reg_aper : str, optional
        Region file used for plotting apertures.
    dtex, dtey : float, optional
        Text offsets for aperture labels.
    rad : float, optional
        Radius of the plotted extraction circles in arcseconds.
    cosmetic : bool, optional
        If True, smooth the extracted spectra for display.
    reg_name : str, optional
        DS9 segment file used to define the extracted path.
    zt : float, optional
        Redshift used to shift spectra to the rest frame.
    facs : float, optional
        Global figure-size scaling factor.
    lA1, lA2 : float, optional
        Rest-frame wavelength interval shown in the extracted spectra.
    dxR : float, optional
        Horizontal offset used in the figure layout.
    savef : bool, optional
        If True, save the figure as a PDF.
    dir : str, optional
        Output directory.

    Returns
    -------
    None

    Notes
    -----
    This routine depends on ``tools.extract_segment1d`` and on the helper
    function ``plot_circle`` to annotate extraction nodes on the map.
    """
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

def plot_velana2x(titf,vals_map1,vals_map2,dyt=0.95,pltlabel=r'$Velocity\ shift\ [km\ s^{-1}]$',
    unitplots=[r'[km\ s^{-1}]',r'[km\ s^{-1}]'],labplots=['Velocity shift Blue','Velocity shift Red'],
    path='',DA=None,model='helic',alpha=1.0,fitmod=False,file0='J102700+174900_Gas.fits.gz',file1='J102700+174900_Gas.fits.gz',
    nlinsA=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],lamdsA=[6549.859,6564.632,6585.278],nlinsB=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],
    lamdsB=[6549.859,6564.632,6585.278],hdu=0,reg_dir='',reg_aper='apertu.reg',reg_name='paths_J1027_C.reg',zt=0,facs=1,
    lA1=6520.0,lA2=6610,lB1=6520.0,lB2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],av=[0.10,0.03,0.09,0.03],
    sigT=2,loc=3,facx=0.8,facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,x_max=1,txt_size=18,
    ylabel='y-value',xlabel='x-value',dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    """
    Compare two velocity maps and their extracted pseudo-slit spectra.

    This function builds a composite figure with two velocity maps,
    extraction paths, the corresponding pseudo-slit spectra, and radial
    velocity profiles extracted along the same DS9-defined paths.

    Parameters
    ----------
    titf : str
        Output figure name.
    vals_map1, vals_map2 : tuple
        Tuples of the form ``(map, vmax, vmin)`` for the two maps.
    dyt : float, optional
        Vertical scaling factor used in the figure layout.
    pltlabel : str, optional
        Y-axis label for the extracted velocity curves.
    unitplots : list of str, optional
        Units of the two displayed maps.
    labplots : list of str, optional
        Titles of the two map panels.
    path : str, optional
        Directory containing the cube files.
    DA : float, optional
        Angular-diameter-distance scaling to convert arcseconds into kpc.
    model : {'helic', 'sin', 'vmax'}, optional
        Functional form used when ``fitmod=True``.
    alpha : float, optional
        Transparency of the maps.
    fitmod : bool, optional
        If True, fit a simple kinematic model to the extracted velocity curve.
    file0, file1 : str, optional
        Input cubes used for the blue and red spectral ranges.
    reg_dir, reg_aper, reg_name : str, optional
        DS9 region files used for paths and apertures.
    zt : float, optional
        Redshift used to shift the spectral axis to the rest frame.
    lA1, lA2, lB1, lB2 : float, optional
        Wavelength windows used for the two pseudo-slit extractions.
    savef : bool, optional
        If True, save the figure as a PDF.
    dir : str, optional
        Output directory.

    Returns
    -------
    None

    Notes
    -----
    The routine uses ``tools.extract_segment`` and ``tools.extract_segment_val``
    to obtain spatially resolved spectra and velocity profiles.
    """
    slides1,wavet1,dpix,vals1,hdr,colr1,widt1,namet1=tools.extract_segment(file0,path=path,reg_dir=reg_dir,reg_name=reg_name,z=zt,lA1=lA1,lA2=lA2,sigT=sigT,cosmetic=True,hdu=hdu)
    slides2,wavet2,dpix,vals2,hdr,colr2,widt2,namet2=tools.extract_segment(file1,path=path,reg_dir=reg_dir,reg_name=reg_name,z=zt,lA1=lB1,lA2=lB2,sigT=sigT,cosmetic=True,hdu=hdu)
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
    get_plot_map(plt,flux1,vmax1,vmin1,pix=pix,tit=labplots[0],lab=unitplots[0],alpha=alpha)
    nxt,nyt=flux1.shape
    slides_v1=tools.extract_segment_val(flux1,hdr,pix,reg_dir=reg_dir,reg_name=reg_name)
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
    get_plot_map(plt,flux2,vmax2,vmin2,pix=pix,tit=labplots[1],lab=unitplots[1],alpha=alpha)
    nxt,nyt=flux2.shape
    slides_v2=tools.extract_segment_val(flux2,hdr,pix,reg_dir=reg_dir,reg_name=reg_name)
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
        ax = fig.add_axes([dx1+pro1[0]*dx+dx*dxR, dy1+pro2[2]*dy*dyt+dy*(nls-i-1)/nls*(dyt*0.8), dx*0.48, dy/(nls)*(dyt*0.8)])
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
        ax = fig.add_axes([dx1+pro1[0]*dx+dx*(dxR+0.5), dy1+pro2[2]*dy*dyt+dy*(nls-i-1)/nls*(dyt*0.8), dx*0.48, dy/(nls)*(dyt*0.8)])
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
        ax.set_ylabel('').set_visible(False)
        plt.setp( ax.get_yticklabels(), visible=False)
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

    ofset=dx1+pro1[0]*dx+dx*(dxR+1)
    fl=0.06 # factor to give the label spaces between the plots
    fx=1-fl*(nls-1)
    for i in range(0, nls):
        ax = fig.add_axes([dx1+pro1[1]*dx*2.0*i/nls*fx+dx*2.0*fl*i+ofset, dy1+pro2[2]*dy, dx*2.0*fx/nls*0.4, dy*dyt*0.8])
        vel_vec1=slides_v1[i]
        vel_vec2=slides_v2[i]
        xtp1=np.arange(0, len(vel_vec1))*pix
        xtp2=np.arange(0, len(vel_vec2))*pix
        plt.plot(xtp1*daf,vel_vec1,lw=4,color=colr1[i])
        plt.plot(xtp2*daf,vel_vec2,lw=4,color=colr2[i],ls='--')
        plt.scatter(xtp1*daf,vel_vec1,s=63,color=colr1[i])
        plt.scatter(xtp2*daf,vel_vec2,s=63,color=colr2[i])
        if DA is not None:
            plt.xlabel(r'$R\ [kpc]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec1)-1)*pix*DA)
        else:
            plt.xlabel(r'$R\ [arcsec]$',fontsize=18)
            plt.xlim(0.0,(len(vel_vec1)-1)*pix)
        plt.ylabel(pltlabel,fontsize=18)
        plt.text(0.8,0.8,namet1[i],fontsize=20,transform=ax.transAxes,color=colr1[i],weight='bold') 
    if fitmod:
        nt1=np.isfinite(vel_vec1)
        xtp1=xtp1[nt1]
        vel_vec1=vel_vec1[nt1]
        nt2=np.isfinite(vel_vec2)
        xtp2=xtp2[nt2]
        vel_vec2=vel_vec2[nt2]
        xtp=xtp1
        xtp=np.concatenate((xtp,xtp2),axis=0)
        vel_vec=vel_vec1
        vel_vec=np.concatenate((vel_vec,vel_vec2),axis=0)
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


def plot_velana(titf,vals_map,path='',DA=None,model='helic',alpha=1.0,fitmod=False,file0='J102700+174900_Gas.fits.gz',
    nlins=[r'$[NII]$',r'$H_{\alpha}$',r'$[NII]$'],lamds=[6549.859,6564.632,6585.278],hdu=0,reg_dir='',reg_aper='apertu.reg',
    reg_name='paths_J1027_C.reg',zt=0,facs=1,lA1=6520.0,lA2=6610.0,dxR=0.25,savef=True,pro1=[0,1,2],nx=2,ny=4,pro2=[0,0,0],
    av=[0.10,0.03,0.09,0.03],sigT=2,loc=3,facx=0.8,facy=-1,tpt=1,obt=['C','D','E','G','J','L'],y_min=0,y_max=1,x_min=0,
    x_max=1,txt_size=18,ylabel='y-value',xlabel='x-value',dxl=0.2,dyl=0.9,color=['blue','green','red'],lin=['-','--',':'],dir='./'):
    """
    Plot a velocity map together with pseudo-slit spectra and radial profiles.

    This function generates a multi-panel diagnostic figure showing a
    single velocity map, the extraction path, a set of pseudo-slit spectra
    along the path, and the corresponding extracted velocity profiles.

    Parameters
    ----------
    titf : str
        Output figure name.
    vals_map : tuple
        Tuple ``(map, vmax, vmin)`` for the velocity map.
    path : str, optional
        Directory containing the input cube.
    DA : float, optional
        Conversion factor from arcseconds to kpc.
    model : {'helic', 'sin', 'vmax'}, optional
        Functional form used when ``fitmod=True``.
    alpha : float, optional
        Transparency of the displayed map.
    fitmod : bool, optional
        If True, fit a simple analytic model to the extracted velocity profile.
    file0 : str, optional
        Input cube file.
    reg_dir, reg_aper, reg_name : str, optional
        DS9 region files used for extraction paths and aperture overlays.
    zt : float, optional
        Redshift applied to the wavelength axis.
    lA1, lA2 : float, optional
        Wavelength window used to build the pseudo-slit spectra.
    savef : bool, optional
        If True, save the result to PDF.
    dir : str, optional
        Output directory.

    Returns
    -------
    None
    """
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
    """
    Evaluate a helical-like velocity profile.

    Parameters
    ----------
    r : array-like
        Radial coordinate.
    alpha : float
        Overall amplitude parameter.
    beta : float
        Oscillation or winding coefficient.
    gama : float
        Phase term.
    theta : float
        Systemic offset.

    Returns
    -------
    ndarray
        Model velocity profile.
    """
    vr=alpha*r**(1/2)*np.sin(beta*r**(1/2)+gama)+theta
    return vr

def sin_func(r,alpha,beta,gama,theta):
    """
    Evaluate a sinusoidal velocity profile.

    Parameters
    ----------
    r : array-like
        Radial coordinate.
    alpha : float
        Oscillation amplitude.
    beta : float
        Radial scale parameter.
    gama : float
        Phase offset.
    theta : float
        Additive velocity offset.

    Returns
    -------
    ndarray
        Model velocity profile.
    """
    vr=alpha*np.sin(r/beta*np.pi+gama)+theta
    return vr    

def plot_bpt_map(file,name='',alpha=1,orientation=None,hd=0,ewsing=1,max_typ=5,location=None,savef=False,
    fig_path='',fwcs=False,scale=0,facp=0.8,tit='BPT',cont=False,path='',indEwHa=769,indOIII=76,indNII=123,
    indHa=124,indHb=63,ret=1,agn=5,sf=3,inte=2,comp=4):
    """
    Build and plot a BPT classification map from a fitted parameter cube.

    This routine reads line-flux and equivalent-width maps from a FITS cube,
    computes the standard BPT diagnostic ratios, classifies each pixel, and
    displays the resulting classification map using a discrete colormap.

    Parameters
    ----------
    file : str
        Input FITS file containing the parameter cube.
    name : str, optional
        Object identifier used in the output file name.
    alpha : float, optional
        Map transparency.
    orientation : str, optional
        Color-bar orientation.
    hd : int, optional
        FITS HDU index.
    ewsing : float, optional
        Multiplicative scaling applied to the Halpha equivalent width.
    max_typ : int, optional
        Maximum class label shown by the colormap.
    location : str, optional
        Color-bar location.
    savef : bool, optional
        If True, save the figure as a PDF.
    fig_path : str, optional
        Output directory.
    fwcs : bool, optional
        If True, plot in WCS coordinates.
    scale : float, optional
        Display scale passed to the internal plotting helper.
    tit : str, optional
        Plot title.
    cont : bool, optional
        Forwarded to the internal plotting helper.
    path : str, optional
        Directory containing the FITS file.
    indEwHa, indOIII, indNII, indHa, indHb : int, optional
        Indices of the required parameter maps in the FITS cube.
    ret, agn, sf, inte, comp : int, optional
        Numeric class labels used by the BPT map.

    Returns
    -------
    None

    Notes
    -----
    The classification itself is performed by ``tools.bpt``.
    """
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


def plot_bpt_map2(fileR,fileB,name='',zt=0,alpha=1,tight=True,maskB=None,fontsize=18,maskR=None,orientation=None,
    hd=0,ewsing=1,max_typ=5,location=None,savef=False,fig_path='',fwcs=False,scale=0,facp=0.8,tit='BPT',cont=False,
    path='',indcB=769,indcR=769,indOIII=76,indNII=123,indHa=124,indHb=63,ret=1,agn=5,sf=3,inte=2,comp=4):
    """
    Build and plot a BPT classification map from separate red and blue products.

    This version of the BPT plotter reads the required emission-line maps
    from two different fitted files, typically corresponding to different
    spectral windows, combines them into the classical BPT ratios, and
    plots the resulting classification map.

    Parameters
    ----------
    fileR, fileB : str
        Input fitted files for the red and blue spectral ranges.
    name : str, optional
        Object identifier used in the output file name.
    zt : float, optional
        Redshift correction applied when deriving line maps.
    alpha : float, optional
        Map transparency.
    tight : bool, optional
        If True, call ``plt.tight_layout()`` when saving.
    maskB, maskR : ndarray, optional
        Boolean masks applied to the blue and red products.
    fontsize : int, optional
        Color-bar font size.
    orientation : str, optional
        Color-bar orientation.
    savef : bool, optional
        If True, save the figure.
    fig_path : str, optional
        Output directory.
    fwcs : bool, optional
        If True, plot in WCS coordinates.
    indcB, indcR, indOIII, indNII, indHa, indHb : int, optional
        Map indices used to recover the required line and continuum products.
    ret, agn, sf, inte, comp : int, optional
        Numeric class labels.

    Returns
    -------
    None
    """
    basefigname='BPT_map_NAME'
    flux1,vel1,sigma1,ew1=tools.get_fluxline(fileR,path=path,ind1=indHa,  ind2=indHa+2,  ind3=indHa+1,  ind4=indcR,lo=6564.63,zt=zt,val0=0)
    flux2,vel2,sigma2,ew2=tools.get_fluxline(fileR,path=path,ind1=indNII, ind2=indNII+2, ind3=indNII+1, ind4=indcR,lo=6585.27,zt=zt,val0=0)
    flux3,vel3,sigma3,ew3=tools.get_fluxline(fileB,path=path,ind1=indHb,  ind2=indHb+2,  ind3=indHb+1,  ind4=indcB,lo=4862.68,zt=zt,val0=0)
    flux4,vel4,sigma4,ew4=tools.get_fluxline(fileB,path=path,ind1=indOIII,ind2=indOIII+2,ind3=indOIII+1,ind4=indcB,lo=5008.24,zt=zt,val0=0)
    hdr=fits.getheader(path+'/'+fileB)
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
    if maskB is not None:
        flux3[maskB]=np.nan
        flux4[maskB]=np.nan
    if maskR is not None:
        ew1[maskR]=np.nan
        flux1[maskR]=np.nan
        flux2[maskR]=np.nan
    fluxOIII=flux4
    fluxNII=flux2
    fluxHa=flux1
    fluxHb=flux3
    ewHa=ewsing*ew1

    ratio1=np.log10(fluxOIII/fluxHb)
    ratio2=np.log10(fluxNII/fluxHa)
    bounds = np.arange(0, max_typ + 1) + 0.5  # Para centrar los ticks
    map_bpt=tools.bpt(ewHa,ratio2,ratio1,ret=ret,agn=agn,sf=sf,inte=inte,comp=comp)
    
    type_p=r'log($[OIII]H\beta$)~vs~log($[NII]H\alpha$)'
    type_n=r'log([OIII]/H\beta) vs log([NII]/H\alpha)'
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
    get_plot_map(plt,map_bpt,vmax,vmin,cmt=cm,ticks=ticks,fontsize=fontsize,labels=labels,norm=norm,fwcs=fwcs,
        objsys=objsys,pix=pix,tit=tit,scale=scale,lab=type_n,cont=cont,orientation=orientation,location=location,alpha=alpha)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
        if tight:
            plt.tight_layout()
    else:
        plt.show()


def plot_whan_map2(fileR,name='',zt=0,alpha=1,tight=True,fontsize=18,maskR=None,orientation=None,hd=0,
    ewsing=1,max_typ=5,location=None,savef=False,fig_path='',fwcs=False,scale=0,facp=0.8,tit='WHaN',cont=False,path='',
    indcR=769,indNII=123,indHa=124,ret=1,agn=5,sf=3,wagn=4):
    """
    Build and plot a WHAN classification map from fitted red-side products.

    This routine derives Halpha equivalent width and [NII]/Halpha ratio
    maps from a fitted product and displays the resulting WHAN classes.

    Parameters
    ----------
    fileR : str
        Input fitted file for the red spectral range.
    name : str, optional
        Object identifier used in the output file name.
    zt : float, optional
        Redshift correction used when deriving line maps.
    alpha : float, optional
        Map transparency.
    tight : bool, optional
        If True, call ``plt.tight_layout()`` when saving.
    fontsize : int, optional
        Color-bar font size.
    maskR : ndarray, optional
        Boolean mask applied to the red product.
    orientation : str, optional
        Color-bar orientation.
    savef : bool, optional
        If True, save the figure as a PDF.
    fig_path : str, optional
        Output directory.
    fwcs : bool, optional
        If True, plot in WCS coordinates.
    path : str, optional
        Directory containing the input file.
    indcR, indNII, indHa : int, optional
        Indices used to recover the continuum and line products.
    ret, agn, sf, wagn : int, optional
        Numeric class labels used by the WHAN map.

    Returns
    -------
    None
    """
    basefigname='whan_map_NAME'
    flux1,vel1,sigma1,ew1=tools.get_fluxline(fileR,path=path,ind1=indHa,  ind2=indHa+2,  ind3=indHa+1,  ind4=indcR,lo=6564.63,zt=zt,val0=0)
    flux2,vel2,sigma2,ew2=tools.get_fluxline(fileR,path=path,ind1=indNII, ind2=indNII+2, ind3=indNII+1, ind4=indcR,lo=6585.27,zt=zt,val0=0)
    hdr=fits.getheader(path+'/'+fileR)
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
    if maskR is not None:
        ew1[maskR]=np.nan
        flux1[maskR]=np.nan
        flux2[maskR]=np.nan
    fluxNII=flux2
    fluxHa=flux1
    ewHa=ewsing*ew1

    ratio2=np.log10(fluxNII/fluxHa)
    bounds = np.arange(0, max_typ + 1) + 0.5  # Para centrar los ticks
    map_whan=tools.whan(ewHa,ratio2,agn=agn,sf=sf,wagn=wagn,ret=ret)
    
    type_p=r'log($[OIII]H\beta$)~vs~log($[NII]H\alpha$)'
    type_n=r'EW_H\alpha vs log([NII]/H\alpha)'
    vmax=None
    vmin=None 
    ticks = [1,2,3,4]
    labels = ['Ret','SF','wAGN','sAGN']
    colores = ['orange','mediumspringgreen','#A788CF','darkslateblue']

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
    get_plot_map(plt,map_whan,vmax,vmin,cmt=cm,ticks=ticks,fontsize=fontsize,labels=labels,norm=norm,fwcs=fwcs,
        objsys=objsys,pix=pix,tit=tit,scale=scale,lab=type_n,cont=cont,orientation=orientation,location=location,alpha=alpha)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
        if tight:
            plt.tight_layout()
    else:
        plt.show()        

def plot_whad_map2(fileR,name='',zt=0,alpha=1,tight=True,fontsize=18,maskR=None,orientation=None,hd=0,ewsing=1,max_typ=5,
    location=None,savef=False,fig_path='',fwcs=False,scale=0,facp=0.8,tit='WHaD',cont=False,path='',indcR=769,indHa=124,
    ret=1,agn=5,sf=3,wagn=4,unk=2):
    """
    Build and plot a WHAD classification map.

    This routine derives Halpha equivalent width and velocity-dispersion
    maps from a fitted red-side product, converts them to logarithmic space,
    classifies each spaxel using the WHAD diagnostic, and plots the result.

    Parameters
    ----------
    fileR : str
        Input fitted red-side file.
    name : str, optional
        Object identifier used in the output file name.
    zt : float, optional
        Redshift correction used when deriving line maps.
    alpha : float, optional
        Map transparency.
    tight : bool, optional
        If True, call ``plt.tight_layout()`` when saving.
    fontsize : int, optional
        Color-bar font size.
    maskR : ndarray, optional
        Boolean mask applied to the red product.
    orientation : str, optional
        Color-bar orientation.
    savef : bool, optional
        If True, save the result as a PDF.
    fig_path : str, optional
        Output directory.
    fwcs : bool, optional
        If True, plot in WCS coordinates.
    path : str, optional
        Directory containing the input file.
    indcR, indHa : int, optional
        Indices of the continuum and Halpha products.
    ret, agn, sf, wagn, unk : int, optional
        Numeric class labels used by the WHAD map.

    Returns
    -------
    None
    """
    basefigname='whad_map_NAME'
    flux1,vel1,sigma1,ew1=tools.get_fluxline(fileR,path=path,ind1=indHa,  ind2=indHa+2,  ind3=indHa+1,  ind4=indcR,lo=6564.63,zt=zt,val0=0)
    hdr=fits.getheader(path+'/'+fileR)
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
    if maskR is not None:
        ew1[maskR]=np.nan
        sigma1[maskR]=np.nan
    logew=np.log10(ewsing*ew1)
    logsig=np.log10(sigma1)
    bounds = np.arange(0, max_typ + 1) + 0.5  # Para centrar los ticks
    map_whad=tools.whad(logew,logsig,agn=agn,sf=sf,wagn=wagn,ret=ret,unk=unk)
    
    type_p=r'log($[OIII]H\beta$)~vs~log($[NII]H\alpha$)'
    type_n=r'\sigma_H\alpha vs EW_H\alpha '
    vmax=None
    vmin=None 
    ticks = [1,2,3,4,5]
    labels = ['Ret','UNK','SF','wAGN','sAGN']
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
    get_plot_map(plt,map_whad,vmax,vmin,cmt=cm,ticks=ticks,fontsize=fontsize,labels=labels,norm=norm,fwcs=fwcs,
        objsys=objsys,pix=pix,tit=tit,scale=scale,lab=type_n,cont=cont,orientation=orientation,location=location,alpha=alpha)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
        if tight:
            plt.tight_layout()
    else:
        plt.show()        

def plot_single_map(file,valmax,valmin,name='',scale=0,sb=False,fwcs=False,logs=False,zerofil=False,valz=None,scalef=1.0,
    basefigname='Ha_vel_map_NAME',sumc=False,path='',hd=0,indx=0,indx2=None,tit='',lab='',facp=0.8,facx=6.5,facy=7.6,
    cont=False,alpha=1,orientation=None,location=None,savef=False,fig_path=''):
    """
    Plot a single map or cube slice from a FITS product.

    This routine reads a FITS file, extracts a 2D map or a slice from a 3D
    cube, optionally applies normalization or logarithmic scaling, and plots
    the result using the internal map-display helper.

    Parameters
    ----------
    file : str
        Input FITS file.
    valmax, valmin : float
        Maximum and minimum values used for the display scale.
    name : str, optional
        Object identifier used in the output file name.
    scale : float, optional
        Display scale passed to the internal plotting helper.
    sb : bool, optional
        If True, divide by pixel area to obtain surface brightness.
    fwcs : bool, optional
        If True, plot the image in WCS coordinates.
    logs : bool, optional
        If True, display the logarithm of the map.
    zerofil : bool, optional
        If True, replace zero or low-valued pixels with NaN.
    valz : float, optional
        Threshold used when ``zerofil=True``.
    scalef : float, optional
        Multiplicative scaling applied to the data.
    basefigname : str, optional
        Output base file name. ``NAME`` is replaced by ``name``.
    sumc : bool, optional
        If True and the input is a cube, integrate along the spectral axis.
    path : str, optional
        Directory containing the FITS file.
    hd : int, optional
        FITS HDU index.
    indx : int, optional
        Index of the plane extracted from a 3D cube.
    indx2 : int, optional
        Secondary index used to form a ratio map.
    tit : str, optional
        Display title.
    lab : str, optional
        Physical units shown on the color bar.
    savef : bool, optional
        If True, save the figure as a PDF.
    fig_path : str, optional
        Output directory.

    Returns
    -------
    None
    """

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
    get_plot_map(plt,map_val,valmax,valmin,fwcs=fwcs,objsys=objsys,pix=pix,tit=tit,scale=scale,lab=lab,cont=cont,
        orientation=orientation,location=location,alpha=alpha)
    if fwcs:
        plt.grid(color='black', ls='solid')
    if savef:
        plt.tight_layout()
        plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
    else:
        plt.show()

def get_plot_map(plt,flux,vmax,vmin,pix=0.2,scale=0,ticks=None,fontsize=18,labels=None,cmt=None,norm=None,cbarp=True,
    fwcs=False,objsys='J2000',tit='flux',lab='[10^{-16}erg/s/cm^2/arcsec^2]',cont=False,alpha=1,orientation=None,location=None):
    """
    Internal helper used to display a 2D map with consistent styling.

    This function handles the low-level display of a 2D image, including
    colormap selection, normalization, optional contours, coordinate
    labeling, pixel-to-arcsecond scaling, and color-bar creation.

    Parameters
    ----------
    plt : matplotlib.pyplot
        Pyplot interface used for plotting.
    flux : ndarray
        2D map to display.
    vmax, vmin : float
        Display limits.
    pix : float, optional
        Pixel scale in arcseconds.
    scale : float, optional
        Additional display scaling control.
    ticks : list, optional
        Color-bar tick positions.
    fontsize : int, optional
        Font size used in labels and the color bar.
    labels : list, optional
        Custom tick labels for the color bar.
    cmt : matplotlib colormap, optional
        Colormap used to display the image.
    norm : matplotlib.colors.Normalize, optional
        Explicit normalization object.
    cbarp : bool, optional
        If True, draw a color bar.
    fwcs : bool, optional
        If True, assume the plot is being drawn in WCS coordinates.
    objsys : str, optional
        Name of the coordinate system.
    tit : str, optional
        Quantity name used in labels.
    lab : str, optional
        Physical units shown on the color bar.
    cont : bool, optional
        If True, overlay contours.
    orientation : str, optional
        Color-bar orientation.
    location : str, optional
        Color-bar location.
    alpha : float, optional
        Image transparency.

    Returns
    -------
    matplotlib.image.AxesImage
        Handle to the displayed image.

    Notes
    -----
    This is the common rendering backend used by most map-plotting routines
    in ``MapLines.tools.plot_tools``.
    """
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
        if tit != '':
            plt.title(r'$'+tit+'$',fontsize=fontsize)
    plt.xlabel(r'$'+xlab+labs+'$',fontsize=fontsize)
    plt.ylabel(r'$'+ylab+labs+'$',fontsize=fontsize)
    ict=plt.imshow(flux,cmap=cm,norm=norm,origin='lower',extent=[-ny*pix/2./fac+dx,ny*pix/2./fac+dx,-nx*pix/2./fac+dy,nx*pix/2./fac+dy],vmax=vmax,vmin=vmin,alpha=alpha)#,norm=LogNorm(0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    plt.xlim(-ny*pix/2/fac+dx,ny*pix/2/fac+dx)
    plt.ylim(-nx*pix/2/fac+dy,nx*pix/2/fac+dy)  
    if cont:
        plt.contour(flux,lev,colors='black',linewidths=2,extent=[-ny*pix/2./fac+dx,ny*pix/2./fac+dx,-nx*pix/2./fac+dy,nx*pix/2./fac+dy],zorder=1)
    if cbarp:
        if ticks is not None:
            cbar=plt.colorbar(ict,orientation=orientation,location=location,ticks = ticks,pad=0.01)
        else:
            cbar=plt.colorbar(ict,orientation=orientation,location=location)
        if location == 'top':
            cbar.set_label(r"$"+tit+r"\ "+lab+"$",fontsize=fontsize)
        else:
            cbar.set_label(r"$"+lab+"$",fontsize=fontsize)  
        if labels is not None:
            cbar.set_ticklabels(labels) 
    else:
        return ict              

def get_plot(flux,savef=True,pix=0.2,name='Residual',tit='flux',outs=[],title=None,cbtr=True,bpte=False,maxmin=[],ewp=False):
    """
    Plot a 2D map with optional aperture overlays and color bar.

    This function displays a two-dimensional map using Matplotlib and can
    optionally overlay rectangular apertures or regions defined through
    the ``outs`` parameter. It supports both logarithmic scaling for
    intensity-like maps and linear scaling for diagnostic-classification
    maps such as BPT-like products.

    Parameters
    ----------
    flux : ndarray
        Two-dimensional map to be displayed.
    savef : bool, optional
        If True, save the figure to a PDF file named
        ``name + '_map.pdf'``. If False, display the figure interactively.
    pix : float, optional
        Pixel scale in arcseconds per pixel. This is used to define the
        spatial extent of the displayed image axes.
    name : str, optional
        Base name of the output figure when ``savef=True``.
    tit : str, optional
        Quantity name shown in the color-bar label.
    outs : list, optional
        Optional list containing aperture or region coordinates and labels.
        When provided, rectangular outlines and text labels are overplotted
        on the image.
    title : str, optional
        Figure title.
    cbtr : bool, optional
        If True, draw a color bar.
    bpte : bool, optional
        If True, use linear image scaling appropriate for classification
        maps. If False, use logarithmic scaling appropriate for flux-like
        quantities.
    maxmin : list, optional
        Two-element list defining the minimum and maximum display values
        ``[vmin, vmax]``. If not given, internal defaults are used.
    ewp : bool, optional
        If True, the color-bar label is displayed without the default
        flux surface-brightness units.

    Returns
    -------
    None

    Notes
    -----
    By default, the input map is divided by ``pix**2`` when ``bpte=False``
    so that it is displayed as a surface-brightness-like quantity. The
    plotting extent is centered on the map and expressed in arcseconds.

    The optional ``outs`` structure is interpreted as a collection of
    rectangular regions with labels. These are drawn directly on top of
    the displayed image and are useful for highlighting apertures or
    selected spatial zones.
    """
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
    """
    Overlay DS9 circular apertures on an existing map axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the apertures will be drawn.
    hdr : astropy.io.fits.Header
        FITS header containing WCS information.
    plt : matplotlib.pyplot
        Matplotlib pyplot handle used for text annotations.
    nx, ny : int
        Map dimensions.
    dpix : float
        Pixel scale in arcseconds.
    reg_dir : str, optional
        Directory containing the DS9 region file.
    reg_file : str, optional
        Name of the DS9 aperture file.

    Returns
    -------
    None

    Notes
    -----
    The aperture definitions are read through ``tools.get_apertures`` and
    plotted as circular outlines with labels.
    """
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
    """
    Draw a labeled circular marker on a map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the circle will be drawn.
    xpos, ypos : float
        Pixel coordinates of the circle center.
    nx, ny : int
        Map dimensions.
    dpix : float
        Pixel scale in arcseconds.
    rad : float, optional
        Circle radius in arcseconds.
    color : str, optional
        Circle and label color.
    name : str, optional
        Text label placed next to the circle.
    dtex, dtey : float, optional
        Positional offsets for the text label.

    Returns
    -------
    None
    """
    xposf=(xpos-nx/2.0+1)*dpix
    yposf=(ypos-ny/2.0+1)*dpix
    c = Circle((yposf, xposf), rad, edgecolor=color, facecolor='none',lw=5,zorder=3)
    ax.add_patch(c)
    #if name == '1':
    #    plt.text(yposf+dpix*0.5+dtey,xposf-dpix*2+dtex,name, fontsize=25,color=color,weight='bold')
    #else:
    plt.text(yposf+dpix*0.5+dtey,xposf+dtex,name, fontsize=25,color=color,weight='bold')    

def plot_outputfits(wave_i,fluxt,fluxtE,model,modsI,n_lines,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,
    names0,vals,valsL,samples,errp=True,fontsize=14,scl='-16',colors=['blue','red','purple','brown','pink'],
    name_out='name',dir_out='',res_norm=True,labplot=True,dataFe=None,lorentz=False,skew=False,outflow=False,
    powlaw=False,feii=False):
    """
    Plot the result of a spectral-line fit and save a diagnostic figure.

    This routine generates the standard fit-diagnostic plot used by
    MapLines after fitting a spectrum. It shows the observed spectrum,
    uncertainties, best-fit model, individual components, and optionally
    posterior diagnostics derived from the MCMC samples.

    Parameters
    ----------
    wave_i : ndarray
        Wavelength array of the fitted spectral region.
    fluxt : ndarray
        Observed flux vector.
    fluxtE : ndarray
        Flux uncertainty vector.
    model : ndarray
        Best-fit total model spectrum.
    modsI : list of ndarray
        Individual spectral components returned by the model.
    n_lines : int
        Number of fitted emission lines.
    waves0 : list
        Rest wavelengths of the fitted transitions.
    fac0, facN0 : list
        Amplitude-scaling factors and linked-component names.
    velfac0, velfacN0 : list
        Velocity-scaling factors and linked-component names.
    fwhfac0, fwhfacN0 : list
        Width-scaling factors and linked-component names.
    names0 : list
        Names of the model components.
    vals : list
        Internal parameter names.
    valsL : list
        Human-readable parameter labels.
    samples : ndarray
        Posterior MCMC samples.
    errp : bool, optional
        If True, show error information in the plot.
    fontsize : int, optional
        Base font size.
    scl : str, optional
        Label used for flux scaling.
    colors : list of str, optional
        Colors used for the component curves.
    name_out : str, optional
        Output base name.
    dir_out : str, optional
        Output directory.
    res_norm : bool, optional
        If True, display normalized residuals.
    labplot : bool, optional
        If True, annotate the panel with labels.
    dataFe : ndarray, optional
        FeII template data when used in the model.
    lorentz, skew, outflow, powlaw, feii : bool, optional
        Flags describing the model family.

    Returns
    -------
    None

    Notes
    -----
    This is the main plotting routine called by the fitting functions in
    ``MapLines.tools.line_fit`` after an MCMC fit is completed.
    """
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
    contm0=np.copy(contm)   
    if feii:
        if powlaw:
            contm=contm+modsI[n_lines+1]
        else:
            contm=contm+modsI[n_lines]
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
        ax1.plot(wave_i,contm0+modsI[n_lines+1],linewidth=1,color='red',label=r'FeII')         
    if len(names0) < 5:
        fontsizeL=14
    elif len(names0) < 10:
        fontsizeL=12
    elif len(names0) < 15:
        fontsizeL=10
    else:
        fontsizeL=6
    ax1.set_title("Observed Input Spectrum",fontsize=fontsize)
    ax1.set_xlabel(r'$Wavelength\ [\rm{\AA}]$',fontsize=fontsize)
    ax1.set_ylabel(r'Flux [10$^{'+scl+r'}$erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]',fontsize=fontsize)
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