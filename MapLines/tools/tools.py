#!/usr/bin/env python
import numpy as np
from scipy.ndimage import gaussian_filter1d as filt1d
from scipy.ndimage import gaussian_filter as filtNd
import os
import os.path as ptt
from scipy.special import erf as errf
from scipy.special import voigt_profile as vprf
from scipy.interpolate import interp1d
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
    """
    Write a FITS HDUList to disk, removing an existing compressed file if needed.

    Parameters
    ----------
    name : str
        Output FITS file name.
    hlist : astropy.io.fits.HDUList
        FITS HDUList to be written.

    Returns
    -------
    None

    Notes
    -----
    If a compressed version ``name + '.gz'`` exists, it is removed before
    writing the new file.
    """
    sycall("rm "+name+'.gz')
    if ptt.exists(name) == False:
        hlist.writeto(name)
    else:
        name1=name.replace("\ "," ")
        name1=name1.replace(" ","\ ")
        sycall("rm "+name1)
        hlist.writeto(name)

def sycall(comand):
    """
    Execute a shell command.

    Parameters
    ----------
    comand : str
        Shell command to be executed.

    Returns
    -------
    None
    """
    linp=comand
    os.system(comand)

def conv(xt,ke=2.5):
    """
    Smooth a one-dimensional array with a Gaussian kernel.

    Parameters
    ----------
    xt : array-like
        Input vector or spectrum.
    ke : float, optional
        Gaussian kernel width in pixels.

    Returns
    -------
    ndarray
        Smoothed array.
    """
    nsf=len(xt)
    krn=ke
    xf=filt1d(xt,ke)
    return xf

def voigt(x,sigma=1.0,xo=0.0,A1=1.0,gam1=0.0):
    """
    Evaluate a normalized Voigt line profile.

    Parameters
    ----------
    x : array-like
        Wavelength or coordinate grid.
    sigma : float, optional
        Gaussian width of the Voigt profile.
    xo : float, optional
        Central position of the profile.
    A1 : float, optional
        Peak amplitude scaling.
    gam1 : float, optional
        Lorentzian width parameter.

    Returns
    -------
    ndarray
        Voigt profile evaluated on ``x``.
    """
    At=A1/vprf(0, sigma, gam1)
    #sigma=sigma/2.0
    #gam=gam/2.0
    x1=x-xo
    #A1=A1/(np.sqrt(2.0*np.pi)*sigma)
    #y=A1*vprf(x,sigma,gam)
    y=At*vprf(x1,sigma,gam1)
    return y

def spow_law(x, A=1.0, alpha=0.0, xo=5100.0):
    """
    Evaluate a power-law continuum model.

    Parameters
    ----------
    x : array-like
        Wavelength grid.
    A : float, optional
        Amplitude at the reference wavelength ``xo``.
    alpha : float, optional
        Power-law index.
    xo : float, optional
        Reference wavelength.

    Returns
    -------
    ndarray
        Power-law continuum evaluated on ``x``.
    """
    #ct=299792.458
    #x=x/ct*xo
    y=A*(x/xo)**(-alpha)
    return y

def lorentz(x,sigma=1.0,xo=0.0,A1=1.0):
    """
    Evaluate a Lorentzian line profile.

    Parameters
    ----------
    x : array-like
        Wavelength or coordinate grid.
    sigma : float, optional
        Profile width parameter.
    xo : float, optional
        Central position of the profile.
    A1 : float, optional
        Peak amplitude.

    Returns
    -------
    ndarray
        Lorentzian profile evaluated on ``x``.
    """
    y=A1*(0.5*sigma)**2.0/((x-xo)**2.0+(0.5*sigma)**2.0) 
    return y

def gauss_K(x,sigma=1.0,xo=0.0,A1=1.0,alp=0):
    """
    Evaluate a skewed Gaussian line profile.

    Parameters
    ----------
    x : array-like
        Wavelength or coordinate grid.
    sigma : float, optional
        Gaussian width parameter.
    xo : float, optional
        Central position of the line.
    A1 : float, optional
        Amplitude scaling.
    alp : float, optional
        Skewness parameter.

    Returns
    -------
    ndarray
        Skewed Gaussian profile evaluated on ``x``.
    """
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
    """
    Evaluate a Gaussian line profile.

    Parameters
    ----------
    x : array-like
        Wavelength or coordinate grid.
    sigma : float, optional
        Gaussian dispersion.
    xo : float, optional
        Central position of the line.
    A1 : float, optional
        Peak amplitude.

    Returns
    -------
    ndarray
        Gaussian profile evaluated on ``x``.
    """
    y=A1*np.exp(-0.5*(x-xo)**2.0/sigma**2.0)
    return y

def opticFeII(x, data, sigma=1.0, xo=0.0, A1=1.0):
    """
    Evaluate an optical FeII template model.

    Parameters
    ----------
    x : array-like
        Wavelength grid.
    data : ndarray
        Template array containing wavelength and flux columns.
    sigma : float, optional
        Smoothing width applied to the template.
    xo : float, optional
        Wavelength shift applied to the template.
    A1 : float, optional
        Amplitude scaling factor.

    Returns
    -------
    ndarray
        Interpolated and smoothed FeII template spectrum.

    Notes
    -----
    The implementation follows the optical FeII template approach
    referenced in Kovacevic et al. (2010). :contentReference[oaicite:2]{index=2}
    """
    '''Optical FeII model from Kovacevic+10'''
    #data=np.loadtxt('data/FeII_optical_Kovacevic10.txt')
    #dir=os.path.join(MapLines.__path__[0], 'data')+'/'
    #data=np.loadtxt(dir+'FeII.dat')
    wave=data[:,0]
    flux=data[:,1]
    wave=wave+xo
    flux=flux/np.nanmax(flux)*A1
    spec1=interp1d(wave, flux,kind='linear',bounds_error=False,fill_value=0.)(x)
    spec=conv(spec1,ke=sigma)
    #flux_t=np.zeros(len(x))
    #for i in range(0, len(x)):
    #    wt=x[i]
    #    nt=np.where((wave >= wt-3*sigma) & (wave <= wt+3*sigma))[0]
    #    if len(nt) > 0:
    #        flux_t[i]=np.nansum(flux[nt]*np.exp(-0.5*(wt-wave[nt])**2.0/sigma**2.0))
    #    else:
    #        flux_t[i]=0
    return spec


def step_vect(fluxi,sp=20,pst=True,sigma=10):
    """
    Estimate a local noise or step-like uncertainty vector from a spectrum.

    Parameters
    ----------
    fluxi : array-like
        Input spectrum.
    sp : int, optional
        Window size used to estimate the local scatter.
    pst : bool, optional
        If True, use percentile-based robust estimation. If False, use
        the standard deviation.
    sigma : float, optional
        Smoothing width used to remove large-scale structure before
        estimating local scatter.

    Returns
    -------
    ndarray
        Estimated uncertainty vector.
    """
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
    """
    Read a YAML configuration file.

    Parameters
    ----------
    file : str
        Path to the YAML file.

    Returns
    -------
    dict or None
        Parsed configuration dictionary, or ``None`` if the file could
        not be read.
    """
    try:
        with open(file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data
    except:
        print('Config File '+file+' not found')
        return None

def get_priorsvalues(filename):
    """
    Parse the line-model configuration file and assemble fitting priors.

    Parameters
    ----------
    filename : str
        Path to the YAML configuration file describing the emission-line
        setup and priors.

    Returns
    -------
    tuple
        Tuple containing the prior dictionary, number of lines, continuum
        windows, initial values, lower and upper limits, line names,
        wavelengths, colors, scaling relations, and parameter labels.

    Notes
    -----
    This function builds the internal parameter bookkeeping used by
    ``MapLines.tools.line_fit`` and ``MapLines.tools.models``. It reads
    the ``lines``, ``continum`` and ``priors`` sections from the YAML
    file. :contentReference[oaicite:3]{index=3}
    """
    data_lines=read_config_file(filename)
    if data_lines:
        n_lines=len(data_lines['lines'])
        pac=['AoN','dvoN','fwhmoN']
        pacL=[r'$A_{N}$',r'$\Delta v_{N}$',r'$FWHM_{N}$']
        pacH=['N_Amplitude','N_Velocity','N_FWHM']
        waves0=[]
        names0=[]
        colors0=[]
        vals0=[]
        vals=[]
        valsL=[]
        valsH=[]
        fac0=[]
        facN0=[]
        velfac0=[]
        velfacN0=[]
        fwhfac0=[]
        fwhfacN0=[]
        for i in range(0, n_lines):
            parameters=data_lines['lines'][i]
            npar=len(parameters)
            waves0.extend([parameters['wave']])
            names0.extend([parameters['name']])
            try:
                colors0.extend([parameters['color']])
            except:
                colors0=['NoNe'] 
            try:
                facN0.extend([parameters['fac_Name']])
                fac0.extend([parameters['fac']])
                facp=True
            except:
                facN0.extend(['NoNe'])
                fac0.extend([None])
                facp=False
            try:
                velfacN0.extend([parameters['vel_Name']])
                velfac0.extend([parameters['velF']])
                velfacp=True
            except:
                velfacN0.extend(['NoNe'])
                velfac0.extend([None])
                velfacp=False    
            try:
                fwhfacN0.extend([parameters['fwh_Name']])
                fwhfac0.extend([parameters['fwhF']])
                fwhfacp=True
            except:
                fwhfacN0.extend(['NoNe'])
                fwhfac0.extend([None])
                fwhfacp=False    
            inr=0    
            for a in pac:
                val_t=a.replace('N',str(i))
                val_tL=pacL[inr].replace('N',names0[i])
                val_tH=pacH[inr].replace('N',names0[i])
                if 'AoN' in a:
                    if facp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])
                elif 'dvoN' in a:
                    if velfacp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])        
                elif 'fwhmoN' in a:
                    if fwhfacp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])
                else:    
                    vals.extend([val_t])
                    valsL.extend([val_tL])
                valsH.extend([val_tH])
                vals0.extend([val_t])    
                inr=inr+1
        region=data_lines['continum'][0]['region']
        wavec1=data_lines['continum'][0]['wave1']
        wavec2=data_lines['continum'][0]['wave2']
        valsp=data_lines['priors']

        Inpvalues=[]
        Infvalues=[]
        Supvalues=[]
        for valt in vals:
            try:
                Inpvalues.extend([valsp[valt]])
            except:
                print('The keyword '+valt+' is missing in the line config file')
                sys.exit()
            try:
                Infvalues.extend([valsp[valt.replace('o','i')]])
            except:
                print('The keyword '+valt.replace('o','i')+' is missing in the line config file')
                sys.exit()
            try:
                Supvalues.extend([valsp[valt.replace('o','s')]])
            except:
                print('The keyword '+valt.replace('o','s')+' is missing in the line config file')
                sys.exit()
        return  valsp,n_lines,wavec1,wavec2,Inpvalues,Infvalues,Supvalues,waves0,names0,colors0,vals0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,vals,valsL,valsH#,region
    else:
        print('No configuration line model file')
        sys.exit()    

def get_oneDspectra(file1,flux_f=1,erft=0,input_format='SDSS',error_c=True):
    """
    Read a one-dimensional spectrum from several supported formats.

    Parameters
    ----------
    file1 : str
        Input file name.
    flux_f : float, optional
        Global multiplicative flux factor.
    erft : float, optional
        Additional multiplicative scaling applied to the error vector.
    input_format : {'TableFits', 'SDSS', 'IrafFits', 'CSV', 'ASCII'}, optional
        Input spectrum format.
    error_c : bool, optional
        If True, also read or estimate the uncertainty vector.

    Returns
    -------
    pdl_data : ndarray
        Flux array.
    pdl_dataE : ndarray
        Error array.
    wave : ndarray
        Wavelength array.

    Notes
    -----
    For SDSS spectra, the routine converts ``LOGLAM`` to linear wavelength
    and ``IVAR`` to uncertainties. :contentReference[oaicite:4]{index=4}
    """
    if input_format == 'TableFits':
        try:
            hdu_list = fits.open(file1)
        except:
            hdu_list = fits.open(file1+'.gz')
        table_hdu = hdu_list[1]
        table_data = table_hdu.data
        try:
            pdl_data=table_data.field('FLUX')
        except:
            pdl_data=table_data.field('flux')
        try:
            wave=table_data.field('LAMBDA')
        except:
            try:
                wave=table_data.field('lambda')
            except:
                try:
                    wave=table_data.field('wave')
                except:
                    wave=table_data.field('WAVE')
        if error_c:
            try:
                pdl_dataE=table_data.field('ERROR')
            except:
                try:
                    pdl_dataE=table_data.field('fluxE')
                except:
                    pdl_dataE=table_data.field('FLUXE')
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f
        else:
            pdl_dataE=None
    elif input_format == 'SDSS':
        hdu_list = fits.open(file1)
        table_hdu = hdu_list[1]
        table_data = table_hdu.data
        pdl_data=table_data.field('FLUX')
        wave=table_data.field('LOGLAM')
        wave=10**wave
        if error_c:
            pdl_dataE=table_data.field('IVAR')
            pdl_dataE=1/np.sqrt(pdl_dataE)
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f
    elif input_format == 'IrafFits':
        [pdl_data, hdr]=fits.getdata(file1, 0, header=True)
        if error_c:
            pdl_dataE =fits.getdata(file1, 1, header=False)
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f
        crpix=hdr["CRPIX3"]
        try:
            cdelt=hdr["CD3_3"]
        except:
            cdelt=hdr["CDELT3"]
        crval=hdr["CRVAL3"]
        wave=crval+cdelt*(np.arange(len(pdl_data))+1-crpix)  
    elif input_format == 'CSV':
        ft=open(file1,'r')
        wave=[]
        pdl_data=[]
        if error_c:
            pdl_dataE=[]
        for line in ft:
            if not 'Wave' in line:
                data=line.replace('\n','')
                data=data.split(',')
                data=list(filter(None,data))
                if len(data) > 1:
                    wave.extend([float(data[0])])
                    pdl_data.extend([float(data[1])])
                    if error_c:
                        pdl_dataE.extend([float(data[2])])
        wave=np.array(wave)
        pdl_data=np.array(pdl_data)
        if error_c:
            pdl_dataE=np.array(pdl_dataE)
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f
    elif input_format == 'ASCII':
        ft=open(file1,'r')
        wave=[]
        pdl_data=[]
        if error_c:
            pdl_dataE=[]
        for line in ft:
            if not 'Wave' in line:
                data=line.replace('\n','')
                data=data.split(' ')
                data=list(filter(None,data))
                if len(data) > 1:
                    wave.extend([float(data[0])])
                    pdl_data.extend([float(data[1])])
                    if error_c:
                        pdl_dataE.extend([float(data[2])])
        wave=np.array(wave)
        pdl_data=np.array(pdl_data)
        if error_c:
            pdl_dataE=np.array(pdl_dataE)
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f        
    else:
        print('Error: input_format not recognized')
        print('Options are: TableFits, IrafFits, CSV, ASCII')
        sys.exit()   
    return pdl_data,pdl_dataE,wave

def get_cubespectra(file1,file3,flux_f=1,erft=0,error_c=True):
    """
    Read a spectral cube, associated uncertainty cube, and spatial mask.

    Parameters
    ----------
    file1 : str
        Input spectral cube.
    file3 : str
        Mask file. If it does not exist, a full-valid mask is created.
    flux_f : float, optional
        Global multiplicative flux factor.
    erft : float, optional
        Additional multiplicative scaling applied to the error cube.
    error_c : bool, optional
        If True, read or estimate the error cube.

    Returns
    -------
    pdl_cube : ndarray
        Flux cube with shape ``(nz, nx, ny)``.
    pdl_cubeE : ndarray or None
        Error cube.
    mask : ndarray
        Spatial mask.
    wave : ndarray
        Wavelength vector.
    hdr : astropy.io.fits.Header
        FITS header of the cube.

    Notes
    -----
    The function attempts multiple common FITS extension names such as
    ``FLUX``, ``SCI``, ``ERROR``, ``ERR`` and ``IVAR``. :contentReference[oaicite:5]{index=5}
    """
    try:
        [pdl_cube, hdr]=fits.getdata(file1, 'FLUX', header=True)
    except:
        try:
            [pdl_cube, hdr]=fits.getdata(file1, 'SCI', header=True)
        except:
            try:
                [pdl_cube, hdr]=fits.getdata(file1, 0, header=True)
            except:
                print('Error: file '+file1+' with flux extention 0 is not found or not recognized')
                sys.exit()
    if error_c:
        try:
            try:
                pdl_cubeE =fits.getdata(file1, 'ERROR', header=False)
            except:
                pdl_cubeE =fits.getdata(file1, 'ERR', header=False)
        except:
            try:
                pdl_cubeE =fits.getdata(file1, 'IVAR', header=False)
                pdl_cubeE=1.0/np.sqrt(pdl_cubeE)
            except:
                try:
                    pdl_cubeE =fits.getdata(file1, 1, header=False)
                except:
                    print('Warnings: file '+file1+' with error extention 1 is not found or not recognized, reescale flux as error')
                    if erft != 0:
                        pdl_cubeE=pdl_cube*flux_f*erft
                    else:
                        print('Warnings: reescale flux by 10 percent as error')
                        pdl_cubeE=pdl_cube*flux_f*0.1
        if erft != 0:
            pdl_cubeE=pdl_cubeE*flux_f*erft
    else:
        pdl_cubeE=None
    nz,nx,ny=pdl_cube.shape
    pdl_cube=pdl_cube*flux_f
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
    return pdl_cube,pdl_cubeE,mask,wave,hdr



def get_fluxline(file,path='',ind1=3,ind2=7,ind3=4,ind4=9,lo=6564.632,zt=0.0,val0=0):
    """
    Derive line flux, velocity, dispersion, and equivalent width maps.

    Parameters
    ----------
    file : str
        FITS file containing parameter maps.
    path : str, optional
        Directory containing the file.
    ind1, ind2, ind3, ind4 : int, optional
        Indices of amplitude, FWHM, velocity, and continuum maps.
    lo : float, optional
        Rest wavelength of the emission line in Angstrom.
    zt : float, optional
        Redshift correction applied to the velocity field.
    val0 : float, optional
        Sentinel velocity value used to identify invalid pixels.

    Returns
    -------
    flux : ndarray
        Integrated line-flux map.
    vel : ndarray
        Velocity map.
    sigma : ndarray
        Velocity-dispersion map.
    ew : ndarray or None
        Equivalent-width map if a continuum map is available.
    """
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
    """
    Extract a 1D spectrum from a circular aperture in a spectral cube.

    Parameters
    ----------
    filename : str
        Cube file name.
    dir_cube_m : str, optional
        Directory containing the cube.
    ra, dec : str, optional
        Sky coordinates of the aperture center. If not provided, the
        cube center is used.
    rad : float, optional
        Aperture radius in arcseconds.
    sig : float, optional
        Smoothing width applied if ``smoth=True``.
    smoth : bool, optional
        If True, smooth the extracted spectrum.
    avgra : bool, optional
        If True, average the flux inside the aperture. Otherwise sum it.
    head : int, optional
        FITS HDU index to read.

    Returns
    -------
    wave_f : ndarray
        Wavelength vector.
    single_T : ndarray
        Extracted spectrum.
    """
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
    """
    Read circular and box apertures from a DS9 region file.

    Parameters
    ----------
    file : str
        DS9 region file.

    Returns
    -------
    tuple of ndarray
        Arrays containing RA, Dec, radius, box sizes, position angles,
        colors, names, and aperture types.
    """
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
    """
    Read DS9 segment regions from a region file.

    Parameters
    ----------
    reg_dir : str, optional
        Directory containing the region file.
    reg_name : str, optional
        Region file name.

    Returns
    -------
    raL, decL, colr, widt, namet : tuple
        Segment coordinates, colors, line widths, and labels.
    """
    raL=[]
    decL=[]
    colr=[]
    widt=[]
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
                    widt.extend([float(data2[1].replace(' ',''))])
                except:
                    widt.extend([5])
                try:
                    namet.extend([data2[2].replace(' ','')])
                except:
                    namet.extend([str(int(ct))])
            ct=ct+1
    colr=np.array(colr)
    namet=np.array(namet)
    return raL,decL,colr,widt,namet    

def extract_segment1d(file,path='',wcs=None,reg_dir='./',reg_name='test.reg',z=0,
    rad=1.5,lA1=6450.0,lA2=6850.0,plot_t=False,sigT=4,cosmetic=False,hdu=0,nzeros=False):
    """
    Extract 1D spectra along DS9 segment regions from a spectral cube.

    Parameters
    ----------
    file : str
        Input cube file.
    path : str, optional
        Directory containing the cube.
    wcs : astropy.wcs.WCS, optional
        WCS object. If not provided, it is built from the FITS header.
    reg_dir, reg_name : str, optional
        DS9 region file location.
    z : float, optional
        Redshift used to transform wavelengths to rest frame.
    rad : float, optional
        Circular extraction radius around each segment node.
    lA1, lA2 : float, optional
        Wavelength range to extract.
    plot_t : bool, optional
        If True, display the extracted pseudo-slit.
    sigT : float, optional
        Smoothing width used when ``cosmetic=True``.
    cosmetic : bool, optional
        If True, smooth the extracted spectra for display.
    hdu : int, optional
        FITS HDU index.
    nzeros : bool, optional
        If True, replace negative values with NaN before extraction.

    Returns
    -------
    tuple
        Extracted spectra, wavelength array, spatial scale, geometry
        values, header, colors, widths, names, and segment labels.
    """
    ra,dec,colr,widt,namet=get_segment(reg_dir=reg_dir,reg_name=reg_name)
    [pdl_cube, hdr]=fits.getdata(path+file, hdu, header=True)
    nz,nx,ny=pdl_cube.shape
    if nzeros:
        pdl_cube[np.where(pdl_cube < 0)]=np.nan
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
        #flux1t=pdl_cube[nw,:,:]
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
                map_temp=np.copy(pdl_cube[nw[ii],:,:])
                slideT[ii]=np.nansum(map_temp[ntp])
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
                    map_temp=np.copy(pdl_cube[nw[ii],:,:])
                    slideT[ii]=np.nansum(map_temp[ntp])
                if cosmetic:
                    slideT=conv(slideT,ke=sigT)
                    #slideT[k,:]=flux1t    
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
    return slides,wave_f,dpix,vals,hdr,colr,widt,namet,namesS


def extract_regs(map,hdr,reg_file='file.reg',avgra=False):
    """
    Extract values from multiple DS9 apertures on a 2D map.

    Parameters
    ----------
    map : ndarray
        Input 2D map.
    hdr : astropy.io.fits.Header
        FITS header containing the spatial WCS.
    reg_file : str, optional
        DS9 region file.
    avgra : bool, optional
        If True, average values in each aperture. Otherwise sum them.

    Returns
    -------
    ndarray
        Extracted values for all apertures.
    """
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
    """
    Extract the value of a single circular aperture from a 2D map.

    Parameters
    ----------
    map : ndarray
        Input 2D image.
    hdr : astropy.io.fits.Header
        FITS header containing WCS information.
    ra, dec : str
        Aperture center coordinates.
    rad : float, optional
        Aperture radius in arcseconds.
    pix : float, optional
        Pixel scale in arcseconds per pixel.
    avgra : bool, optional
        If True, compute the average value. Otherwise compute the sum.

    Returns
    -------
    float
        Aperture-integrated or aperture-averaged value.
    """
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


def extract_segment_val(flux,hdr,dpix,reg_dir='./',reg_name='test.reg'):
    """
    Extract values along DS9 segment regions from a 2D map.

    Parameters
    ----------
    flux : ndarray
        Input 2D map.
    hdr : astropy.io.fits.Header
        FITS header containing WCS information.
    dpix : float
        Spatial scale in arcseconds per pixel.
    reg_dir, reg_name : str, optional
        DS9 region file location.

    Returns
    -------
    list of ndarray
        Extracted 1D profiles along each segment.
    """
    ra,dec,colr,widt,namet=get_segment(reg_dir=reg_dir,reg_name=reg_name)
    nx,ny=flux.shape
    wcs = WCS(hdr)
    wcs=wcs.celestial
    slides=[]
    for i in range(0, len(ra)):
        raT=ra[i]
        decT=dec[i]
        lT=np.zeros([len(raT)-1],dtype=int)
        slidesT=[]
        ltf=0
        for j in range(0, len(raT)-1):
            sky1=SkyCoord(raT[j]+' '+decT[j],frame=FK5, unit=(u.hourangle,u.deg))
            sky2=SkyCoord(raT[j+1]+' '+decT[j+1],frame=FK5, unit=(u.hourangle,u.deg))
            ypos1,xpos1=skycoord_to_pixel(sky1,wcs)
            ypos2,xpos2=skycoord_to_pixel(sky2,wcs)
            rt=np.sqrt((xpos2-xpos1)**2.0+(ypos2-ypos1)**2.0)
            cosT=(ypos2-ypos1)/rt
            sinT=(xpos2-xpos1)/rt
            lt=int(np.round(rt))+1
            lT[j]=lt
            slideT=np.zeros([lt])
            for k in range(0, lt):
                yt=int(np.round(ypos1+k*cosT))
                xt=int(np.round(xpos1+k*sinT))
                flux1t=flux[xt,yt]
                slideT[k]=flux1t
            slidesT.extend([slideT])
            ltf=lt+ltf 
        slide=np.zeros([ltf])
        ct=0
        for j in range(0, len(raT)-1):
            sldT=slidesT[j]
            for k in range(0, lT[j]):
                slide[ct]=sldT[k]     
                ct=ct+1
        slides.extend([slide])    
    return slides    

def extract_segment(file,path='',reg_dir='./',reg_name='test.reg',z=0,lA1=6450.0,lA2=6850.0,plot_t=False,sigT=4,cosmetic=False,hdu=0):
    """
    Extract pseudo-slit spectra along DS9 segment regions from a cube.

    Parameters
    ----------
    file : str
        Input cube file.
    path : str, optional
        Directory containing the cube.
    reg_dir, reg_name : str, optional
        DS9 region file location.
    z : float, optional
        Redshift used to shift the wavelength axis to rest frame.
    lA1, lA2 : float, optional
        Wavelength interval to extract.
    plot_t : bool, optional
        If True, display the extracted pseudo-slit.
    sigT : float, optional
        Smoothing width used for cosmetic plotting.
    cosmetic : bool, optional
        If True, smooth extracted spectra.
    hdu : int, optional
        FITS HDU index.

    Returns
    -------
    tuple
        Pseudo-slit spectra, wavelength axis, pixel scale, geometric
        metadata, header, colors, widths, and names.
    """
    ra,dec,colr,widt,namet=get_segment(reg_dir=reg_dir,reg_name=reg_name)
    [pdl_cube, hdr]=fits.getdata(path+file, hdu, header=True)
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
    wcs = WCS(hdr)
    wcs=wcs.celestial
    slides=[]
    vals=[]
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
            slideT=np.zeros([lt,len(nw)])
            for k in range(0, lt):
                yt=int(np.round(ypos1+k*cosT[j]))
                xt=int(np.round(xpos1+k*sinT[j]))
                flux1t=pdl_cube[nw,xt,yt]
                #flux1t=flux1t*spl/(wave[nw]*(1+z)*1e-10)**2.*1e-10*1e-23*2.35040007004737e-13/1e-16/1e3
                if cosmetic:
                    flux1t=conv(flux1t,ke=sigT)
                slideT[k,:]=flux1t
            slidesT.extend([slideT])
            ltf=lt+ltf 
        slide=np.zeros([ltf,len(nw)])
        ct=0
        for j in range(0, len(raT)-1):
            sldT=slidesT[j]
            for k in range(0, lT[j]):
                slide[ct,:]=sldT[k,:]     
                ct=ct+1
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
    return slides,wave_f,dpix,vals,hdr,colr,widt,namet


def extract_line_val(flux,hdr,dpix,reg_dir='./',reg_name='test.reg'):
    """
    Extract values along DS9 line regions from a 2D map.

    Parameters
    ----------
    flux : ndarray
        Input 2D map.
    hdr : astropy.io.fits.Header
        FITS header with WCS information.
    dpix : float
        Pixel scale in arcseconds.
    reg_dir, reg_name : str, optional
        DS9 region file location.

    Returns
    -------
    list of ndarray
        Profiles extracted along each line region.
    """
    ra1,dec1,ra2,dec2,colr,namet=get_line(reg_dir=reg_dir,reg_name=reg_name)
    nx,ny=flux.shape
    wcs = WCS(hdr)
    wcs=wcs.celestial
    slides=[]
    for i in range(0, len(ra1)):
        sky1=SkyCoord(ra1[i]+' '+dec1[i],frame=FK5, unit=(u.hourangle,u.deg))
        sky2=SkyCoord(ra2[i]+' '+dec2[i],frame=FK5, unit=(u.hourangle,u.deg))
        ra_deg=sky1.ra.deg
        dec_deg=sky1.dec.deg
        ypos1,xpos1=skycoord_to_pixel(sky1,wcs)
        ypos2,xpos2=skycoord_to_pixel(sky2,wcs)
        rt=np.sqrt((xpos2-xpos1)**2.0+(ypos2-ypos1)**2.0)
        cosT=(ypos2-ypos1)/rt
        sinT=(xpos2-xpos1)/rt
        rtf=rt*dpix
        lt=int(np.round(rt))+1
        slide=np.zeros([lt])
        for j in range(0, lt):
            yt=int(np.round(ypos1+j*cosT))
            xt=int(np.round(xpos1+j*sinT))
            flux1t=flux[xt,yt]
            slide[j]=flux1t
        slides.extend([slide])
    return slides

def extract_line(file,reg_dir='./',reg_name='test.reg',z=0,lA1=6450.0,lA2=6850.0,plot_t=False,sigT=4,cosmetic=False):
    """
    Extract pseudo-slit spectra along DS9 line regions from a cube.

    Parameters
    ----------
    file : str
        Input cube file.
    reg_dir, reg_name : str, optional
        DS9 region file location.
    z : float, optional
        Redshift used to shift the wavelength axis to rest frame.
    lA1, lA2 : float, optional
        Wavelength interval to extract.
    plot_t : bool, optional
        If True, display the extracted pseudo-slit.
    sigT : float, optional
        Smoothing width used for cosmetic display.
    cosmetic : bool, optional
        If True, smooth extracted spectra.

    Returns
    -------
    tuple
        Extracted pseudo-slit spectra, wavelength axis, pixel scale,
        geometric metadata, and FITS header.
    """
    ra1,dec1,ra2,dec2,colr,namet=get_line(reg_dir=reg_dir,reg_name=reg_name)
    #print(ra1[0])
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
    
    wave=crval+cdelt*(np.arange(nz)+1-crpix) 
    wave=wave/(1+z)
    nw=np.where((wave >= lA1) & (wave <= lA2))[0]
    wave_f=wave[nw]
    wcs = WCS(hdr)
    wcs=wcs.celestial
    slides=[]
    vals=[]
    for i in range(0, len(ra1)):
        sky1=SkyCoord(ra1[i]+' '+dec1[i],frame=FK5, unit=(u.hourangle,u.deg))
        sky2=SkyCoord(ra2[i]+' '+dec2[i],frame=FK5, unit=(u.hourangle,u.deg))
        ra_deg=sky1.ra.deg
        dec_deg=sky1.dec.deg
        #sky00=SkyCoord(ra1,dec1,frame=FK5, unit=(u.deg,u.deg))
        ypos1,xpos1=skycoord_to_pixel(sky1,wcs)
        ypos2,xpos2=skycoord_to_pixel(sky2,wcs)
        rt=np.sqrt((xpos2-xpos1)**2.0+(ypos2-ypos1)**2.0)
        cosT=(ypos2-ypos1)/rt
        sinT=(xpos2-xpos1)/rt
        rtf=rt*dpix
        lt=int(np.round(rt))+1
        slide=np.zeros([lt,len(nw)])
        for j in range(0, lt):
            yt=int(np.round(ypos1+j*cosT))
            xt=int(np.round(xpos1+j*sinT))
            flux1t=pdl_cube[nw,xt,yt]
            if cosmetic:
                flux1t=conv(flux1t,ke=sigT)
            slide[j,:]=flux1t
        #out={'Slide':slide,'Lt':lt}
        slides.extend([slide])
        vals.extend([[cosT,sinT,rtf,ypos1,xpos1]])
        if plot_t:
            cm=plt.cm.get_cmap('jet')    
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(slide,origin='lower',cmap=cm,extent=[wave_f[0],wave_f[len(nw)-1],0,lt*dpix],aspect='auto')
            plt.xlim(wave_f[0],wave_f[len(nw)-1])
            plt.ylim(0,lt*dpix) 
            plt.show()
    return slides,wave_f,dpix,vals,hdr


def get_line(reg_dir='./',reg_name='test.reg'):
    """
    Read DS9 line regions from a region file.

    Parameters
    ----------
    reg_dir : str, optional
        Directory containing the region file.
    reg_name : str, optional
        Region file name.

    Returns
    -------
    tuple of ndarray
        Arrays containing line start/end coordinates, colors, and labels.
    """
    ra1=[]
    dec1=[]
    ra2=[]
    dec2=[]
    colr=[]
    namet=[]
    f=open(reg_dir+reg_name,'r')
    ct=1
    for line in f:
        if not 'Region' in line and not 'fk5' in line and not 'global' in line:
            if 'line' in line:
                data=line.replace('\n','').replace('line(','').replace(') # line=0 0 color=',' , ').replace(' width=',' , ').replace(' text={',' , ').replace('}',' ')
                data=data.split(',')
                data=list(filter(None,data))
                #print(data)
                ra1.extend([data[0]])
                dec1.extend([data[1]])
                ra2.extend([data[2]])
                dec2.extend([data[3]])
                colr.extend([data[4].replace(' ','')])
                try:
                    namet.extend([data[5].replace(' ','')])
                except:
                    namet.extend([str(int(ct))])
            ct=ct+1
    ra1=np.array(ra1)
    dec1=np.array(dec1)
    ra2=np.array(ra2)
    dec2=np.array(dec2)
    colr=np.array(colr)
    namet=np.array(namet)
    return ra1,dec1,ra2,dec2,colr,namet


def bpt(wha,niiha,oiiihb,ret=4,agn=3,sf=1,inte=2.5,comp=5,save=False,path='',name='BPT_map',hdr=None):
    """
    Build a BPT classification map.

    Parameters
    ----------
    wha : ndarray
        Halpha equivalent-width map.
    niiha : ndarray
        Log([NII]/Halpha) map.
    oiiihb : ndarray
        Log([OIII]/Hbeta) map.
    ret, agn, sf, inte, comp : float, optional
        Numeric labels assigned to retired, AGN, star-forming,
        intermediate, and composite classes.
    save : bool, optional
        If True, save the classification map to a FITS file.
    path : str, optional
        Output directory.
    name : str, optional
        Output base name.
    hdr : astropy.io.fits.Header, optional
        Header used when saving the FITS file.

    Returns
    -------
    ndarray
        BPT classification map.
    """
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
    """
    Build a WHAN classification map.

    Parameters
    ----------
    wha : ndarray
        Halpha equivalent-width map.
    niiha : ndarray
        Log([NII]/Halpha) map.
    agn, sf, wagn, ret : float, optional
        Numeric labels assigned to strong AGN, star-forming, weak AGN,
        and retired classes.
    save : bool, optional
        If True, save the classification map to a FITS file.
    path : str, optional
        Output directory.
    name : str, optional
        Output base name.
    hdr : astropy.io.fits.Header, optional
        Header used when saving the FITS file.

    Returns
    -------
    ndarray
        WHAN classification map.
    """
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
    """
    Build a WHAD classification map.

    Parameters
    ----------
    logew : ndarray
        Logarithmic equivalent-width map.
    logsig : ndarray
        Logarithmic velocity-dispersion map.
    agn, sf, wagn, ret, unk : float, optional
        Numeric labels assigned to AGN, star-forming, weak AGN, retired,
        and uncertain classes.
    save : bool, optional
        If True, save the classification map to a FITS file.
    path : str, optional
        Output directory.
    name : str, optional
        Output base name.
    hdr : astropy.io.fits.Header, optional
        Header used when saving the FITS file.

    Returns
    -------
    ndarray
        WHAD classification map.
    """
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
    Convert a 2D map into a normalized STL surface.

    Parameters
    ----------
    map : ndarray
        Input 2D map.
    nameid : str, optional
        Output STL base name.
    path_out : str, optional
        Output directory.
    sig : float, optional
        Smoothing width used when ``smoth=True``.
    smoth : bool, optional
        If True, smooth the map before STL generation.
    pval, mval : float, optional
        Linear rescaling parameters applied before STL export.
    border : bool, optional
        If True, zero the border pixels.
    logP : bool, optional
        If True, apply logarithmic scaling before normalization.
    ofsval : float, optional
        Floor value used before smoothing.
    maxval, minval : float, optional
        Manual normalization bounds.

    Returns
    -------
    None
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
    Convert all parameter maps stored in a FITS file into STL surfaces.

    Parameters
    ----------
    file_in : str
        Input FITS file containing parameter maps.
    nameid : str, optional
        Suffix appended to the STL names.
    path_in : str, optional
        Input directory.
    path_out : str, optional
        Output directory.
    sig : float, optional
        Smoothing width used when ``smoth=True``.
    smoth : bool, optional
        If True, smooth the maps before STL generation.
    pval, mval : float, optional
        Linear rescaling parameters.
    border : bool, optional
        If True, zero the border pixels.

    Returns
    -------
    None
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
    """
    Convert a 2D array into an STL triangular mesh.

    Parameters
    ----------
    map : ndarray
        Input 2D map.
    file_out : str
        Output STL base name.
    path_out : str, optional
        Output directory.

    Returns
    -------
    None
    """
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
    """
    Convert a JWST/NIRSpec IFU cube from MJy/sr-like units to
    1e-17 erg/s/cm^2/Angstrom units.

    Parameters
    ----------
    file : str
        Input FITS cube.
    file_out : str
        Output FITS cube.
    zt : float, optional
        Redshift used to shift the spectral axis to rest frame.
    path : str, optional
        Input directory.
    path_out : str, optional
        Output directory.

    Returns
    -------
    None

    Notes
    -----
    The routine updates the output header to use Angstrom in the spectral
    axis and ``E-17erg/s/cm^2/Angstrom`` as brightness unit. :contentReference[oaicite:6]{index=6}
    """
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
    h['BUNIT']='E-17erg/s/cm^2/Angstrom'
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

def A_l(Rv,l):
    """
    Evaluate the Cardelli, Clayton, and Mathis (1989) extinction law.

    Parameters
    ----------
    Rv : float
        Total-to-selective extinction ratio.
    l : array-like
        Wavelength array in Angstrom.

    Returns
    -------
    ndarray
        Extinction curve A(lambda)/A(V).
    """
    #Cardelli, Clayton & Mathis (1989) extintion law
    l=l/10000.; #Amstrongs to Microns
    x=1.0/l
    Arat=np.zeros(len(x))
    for i in range(0, len(x)):
        if x[i] > 1.1 and x[i] <= 3.3:
            y=(x[i]-1.82)
            ax=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
            bx=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
        if x[i] <= 1.1 and x[i] > 0.3:
            ax=0.574*x[i]**1.61
            bx=-0.527*x[i]**1.61
        if x[i] > 3.3 and x[i] <= 8.0:
            if x[i] > 5.9 and x[i] <= 8.0:
                Fa=-0.04473*(x[i]-5.9)**2.0-0.009779*(x[i]-5.9)**3.0
                Fb=0.2130*(x[i]-5.9)**2.0+0.1207*(x[i]-5.9)**3.0
            else:
                Fa=0.0
                Fb=0.0
            ax=1.752-0.316*x[i]-0.104/((x[i]-4.67)**2.0+0.341)+Fa
            bx=-3.090+1.825*x[i]+1.206/((x[i]-4.62)**2.0+0.263)+Fb
        if x[i] > 8.0:
            ax=-1.073-0.628*(x[i]-8.0)+0.137*(x[i]-8.0)**2.0-0.070*(x[i]-8.0)**3.0
            bx=13.670+4.257*(x[i]-8.0)-0.420*(x[i]-8.0)**2.0+0.374*(x[i]-8.0)**3.0
        val=ax+bx/Rv
        if val < 0:
            val=0
        Arat[i]=val
    return Arat

def get_headervals(hdr,keymatch='HaBroad'):
    """
    Extract header keywords whose values match a given component name.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header containing parameter labels.
    keymatch : str, optional
        Substring used to identify the desired component.

    Returns
    -------
    vals : dict
        Dictionary of matching header keywords and values.
    nkeys : list of str
        Matching keyword names.
    """
    keys=list(hdr.keys())
    vals={}
    nkeys=[]
    for i in range(0, len(keys)):
        if keymatch in str(hdr[keys[i]]) and 'VAL_' in keys[i]:
            vals[keys[i]]=hdr[keys[i]]
            nkeys.extend([keys[i]])
    return vals,nkeys


def get_map_component_index(hdr,keymatch='HaBroad'):
    """
    Find the amplitude, velocity, and FWHM indices of a map component.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header containing ``VAL_*`` parameter labels.
    keymatch : str, optional
        Component name to search for.

    Returns
    -------
    indx_amp, indx_vel, indx_fwh : ndarray
        Arrays with indices of amplitude, velocity, and FWHM maps.
    """
    vals,nkeys=get_headervals(hdr,keymatch=keymatch)
    keys=list(vals.keys())
    n_comp=int(len(keys)/3)
    indx_amp=np.zeros(int(n_comp),dtype=int)
    indx_vel=np.zeros(int(n_comp),dtype=int)
    indx_fwh=np.zeros(int(n_comp),dtype=int)
    for i in range(0, n_comp):
        if 'Amplitude' in vals[keys[3*i]]:
            indx_amp[i]=int(keys[3*i].replace('VAL_',''))
        else:
            print('No Amplitude found for component',i)
        if 'Velocity' in vals[keys[3*i+1]]:    
            indx_vel[i]=int(keys[3*i+1].replace('VAL_',''))
        else:
            print('No Velocity found for component',i)
        if 'FWHM' in vals[keys[3*i+2]]:    
            indx_fwh[i]=int(keys[3*i+2].replace('VAL_',''))
        else:
            print('No FWHM found for component',i)
    return indx_amp,indx_vel,indx_fwh

def get_map_param(hdr,keymatch='Noise'):
    """
    Return the index of a single parameter map identified from the header.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header containing ``VAL_*`` labels.
    keymatch : str, optional
        Parameter name to search for.

    Returns
    -------
    int
        Index of the matching parameter map.
    """
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        if keymatch in str(hdr[keys[i]]) and 'VAL_' in keys[i]:    
            indx=int(keys[i].replace('VAL_',''))
    return indx

def fwhm_numeric(wave,flux, dpix=4):
    """
    Estimate the full width at half maximum (FWHM) of a spectral line
    using a numerical interpolation around the half-maximum level.

    The method identifies the region where the flux exceeds half of the
    peak value and performs linear interpolation on both sides of the
    profile to determine the wavelengths corresponding to the half-maximum.
    The resulting width is converted to velocity units assuming a
    Doppler approximation.

    Parameters
    ----------
    wave : ndarray
        1D array of wavelengths. Must be ordered (monotonic increasing
        or decreasing) and in consistent units.
    flux : ndarray
        1D array of flux values corresponding to `wave`. Can contain NaNs,
        which are ignored when computing the maximum flux.
    dpix : int, optional
        Number of pixels around the half-maximum crossing used to define
        the interpolation window on each side of the line. Default is 4.

    Returns
    -------
    fwhm : float
        Estimated FWHM of the line in velocity units (km/s). Returns NaN
        if the FWHM cannot be determined (e.g., insufficient data points
        above half-maximum).
    wave0 : float
        Estimate the central wavelength of the line based on the half-maximum crossings.
    Notes
    -----
    - The FWHM is computed in wavelength units and then converted to
      velocity using:

          FWHM = (Δλ / λ₀) * c

      where λ₀ is the central wavelength and c is the speed of light
      in km/s.
    - The method assumes a single-peaked profile. Complex or multi-peaked
      profiles may yield unreliable results.
    - The interpolation uses `scipy.interpolate.interp1d` with linear
      interpolation.
    - Edge effects are mitigated by limiting the interpolation window
      within array bounds.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes.

    See Also
    --------
    scipy.interpolate.interp1d : Interpolation method used internally.

    """
    ct=299792.458
    max_flux=np.nanmax(flux)
    half_max=max_flux/2.0
    indices=np.where(flux >= half_max)[0]
    if len(indices) < 2:
        print('Warning, Not enough points above half maximum to determine FWHM.')
        return np.nan
    if indices[0]-dpix <0:
        indx0a=0
    else:
        indx0a=indices[0]-dpix
    indx0b=indices[0]+dpix    
    if indices[-1]+dpix >= len(wave):
        indx1b=len(wave)-1
    else:
        indx1b=indices[-1]+dpix
    indx1a=indices[-1]-dpix    
    wave1=interp1d(flux[indx0a:indx0b], wave[indx0a:indx0b], kind='linear',bounds_error=False,fill_value=0.)(half_max)
    wave2=interp1d(flux[indx1a:indx1b], wave[indx1a:indx1b], kind='linear',bounds_error=False,fill_value=0.)(half_max)
    deltawave=wave2-wave1
    wave0=(wave1+wave2)/2.0
    fwhm=deltawave/wave0*ct
    return fwhm,wave0

def get_1D_Totalmodelparam(file,base,path='',keymatch='HaBroad',lam0=6564.632, fwhm_lim=0.0):
    """
    Extract integrated flux, velocity shift, and FWHM from a modeled
    spectral line stored in the MapLine output FITS table.

    This function reads a FITS file containing model components, 
    reconstructs the total model for a given line
    by summing selected components, and computes key physical parameters:
    total flux, velocity offset, and line width (FWHM).

    Parameters
    ----------
    file : str
        Name of the FITS file containing the model output.
    base : str
        Base identifier (currently unused in the function, reserved for
        future extensions or consistency with external interfaces).
    path : str, optional
        Directory path to the FITS file. Default is current directory.
    keymatch : str, optional
        Substring used to identify the relevant model components in the
        FITS table (e.g., 'HaBroad'). Only columns containing both
        `keymatch` and 'Component' will be included in the total model.
    lam0 : float, optional
        Rest-frame wavelength of the emission line (in same units as the
        wavelength array). Default corresponds to Hα (6564.632 Å).
    fwhm_lim : float, optional
        Minimum FWHM (in km/s) required for a component to be included in
        the total model. If set to 0, all components matching `keymatch`
        are included regardless of their width.
    Returns
    -------
    FluxT : float
        Integrated flux of the reconstructed line profile.
    Vel : float
        Velocity shift of the line centroid with respect to `lam0`,
        in km/s (Doppler approximation).
    FWHM : float
        Full width at half maximum of the line in km/s.
    Ampt : float
        The maximum amplitude of the reconstructed line profile.
    Returns None if the required 'Models' extension is not found.

    Notes
    -----
    - The function expects a FITS extension named 'Models' containing:
        - A wavelength column named 'Wave'
        - Multiple model component columns with names including
          both `keymatch` and 'Component'
    - The total model is constructed as the sum of all matching components.
    - The FWHM and centroid are computed using `fwhm_numeric`.
    - The integrated flux is computed using `simpson_r`.
    - Velocity is derived using:

          Vel = ((λ_obs - λ₀) / λ₀) * c

      where c is the speed of light in km/s.
    - No explicit checks are performed for missing columns, NaNs,
      or inconsistent units.

    Raises
    ------
    IOError
        If the FITS file cannot be opened.

    See Also
    --------
    fwhm_numeric : Function used to estimate FWHM and centroid.
    simpson_r : Function used for numerical integration.

    """
    ct=299792.458
    file0=path+'/'+file
    hdu_list = fits.open(file0)
    try:    
        table_hdu = hdu_list['Models']
    except:
        print('Error: No Models found in the FITS file, please provide the model output file')
        return None
    table_data = table_hdu.data
    Wave=table_data.field('Wave')
    keys=list(table_data.names)
    modelT=0
    for keys0 in keys:
        if keymatch in keys0 and 'Component' in keys0:
            if fwhm_lim > 0:
                fwhm,wave0t=fwhm_numeric(Wave,table_data.field(keys0))
                if fwhm > fwhm_lim:
                    modelT=table_data.field(keys0)+modelT
            else:
                modelT=table_data.field(keys0)+modelT
    FWHM,wave0=fwhm_numeric(Wave,modelT)
    FluxT=simpson_r(modelT,Wave,2,len(Wave)-2,typ=0)
    Vel=(wave0-lam0)/lam0*ct
    AmpT=np.nanmax(modelT)
    hdu_list.close()
    return FluxT,Vel,FWHM,AmpT


def rescale_mapmodel(mapT,name,path_out='./',modelbasename='psf_NAME',sigmat=0.2,verbose=False):
    """
    Rescale, smooth, and export a 2D model map as FITS and STL products.

    Parameters
    ----------
    mapT : ndarray
        Input 2D map.
    name : str
        Object name used in the output file names.
    path_out : str, optional
        Output directory.
    modelbasename : str, optional
        Base output name containing the token ``NAME``.
    sigmat : float, optional
        Smoothing width applied to the map.
    verbose : bool, optional
        If True, print normalization diagnostics.

    Returns
    -------
    None
    """
    indx = np.where((mapT == 0) | (np.isfinite(mapT) == False))
    indxt = np.where((np.isfinite(mapT)))
    
    nx,ny=mapT.shape
    mapT[indx]=np.nan
    #mintc=np.nanmin(mapT)# We define the lowest continuum value as the one for which we set the map to NaN, to avoid problems with the logarithm and the normalization. This is because in some cases there are very low continuum values that produce very high flux/continuum ratios, which are not realistic.
    #mapT[np.where(mapT==mintc)]=np.nan
    
    mapT[np.where(np.isfinite(mapT) == False)]=-2
    maxval=np.nanmax(mapT[indxt])
    minval=np.nanmin(mapT[indxt])
    if verbose:
        print(maxval,minval,'Map0')
    mapT=filtNd(mapT,sigma=sigmat)
    maxval=np.nanmax(mapT[indxt])
    minval=0
    if verbose:
        print(maxval,minval,'Map1')
    mapT=(mapT-minval)/(maxval-minval)*1+0
    maxval=np.nanmax(mapT[indxt])
    minval=np.nanmin(mapT[indxt])
    if verbose:
        print(maxval,minval,'Map2')
    mapT[np.where(np.isfinite(mapT) == False)]=minval
    mapT[indx]=minval
    mapT[np.where(mapT < 0)]=-0.2#minval
    nx,ny=mapT.shape
    sycall('mkdir -p '+path_out)
    map_to_stl(mapT*25.34, modelbasename.replace('NAME',name), path_out=path_out+'/')
    mapT[np.where(mapT < 0)]=0
    h1=fits.PrimaryHDU(mapT)
    head_list=[h1]
    hlist=fits.HDUList(head_list)
    hlist.update_extend()
    filet=path_out+'/'+modelbasename.replace('NAME',name)+'.fits'
    hlist.writeto(filet,overwrite=True)
    sycall('gzip -f '+filet)    

def get_mapmodel(name,path_map='./',path_out='./',basename='NAME-2iter_param_V2_HaNII.fits.gz',
    psfmbasename='psf_NAME',sigmat=0.2,lo=6564.632,verbose=False,pow_cr=False,set_am=False,AmpT=2):
    """
    Build a rescaled broad-line model map from fitted parameter products.

    Parameters
    ----------
    name : str
        Object identifier used to replace ``NAME`` in file templates.
    path_map : str, optional
        Directory containing the fitted parameter cubes.
    path_out : str, optional
        Output directory.
    basename : str, optional
        Input FITS file template.
    psfmbasename : str, optional
        Output model base name.
    sigmat : float, optional
        Smoothing width applied to the final map.
    lo : float, optional
        Rest wavelength of the target emission line.
    verbose : bool, optional
        If True, print normalization diagnostics.
    pow_cr : bool, optional
        If True, use the power-law continuum to mask unreliable spaxels.
    set_am : bool, optional
        If True, use the broad-line amplitude threshold to mask spaxels.
    AmpT : float, optional
        Broad-line amplitude threshold used when ``set_am=True``.

    Returns
    -------
    None

    Notes
    -----
    The routine combines component maps identified in the FITS header,
    builds integrated flux maps, masks unreliable spaxels, rescales the
    result, and exports model products. 
    """
    file=path_map+'/'+basename.replace('NAME',name)
    [pdl_cube, hdr]=fits.getdata(file, 0, header=True)
    indx_amp,indx_vel,indx_fwh=get_map_component_index(hdr,keymatch='HaBroad')
    indx_noi=get_map_param(hdr,keymatch='Noise')
    indx_con=get_map_param(hdr,keymatch='Continum')
    nz,nx,ny=pdl_cube.shape
    n_comp=len(indx_amp)
    fluxT=np.zeros((n_comp,nx,ny))
    velT=np.zeros((n_comp,nx,ny))
    sigmaT=np.zeros((n_comp,nx,ny))
    ewT=np.zeros((n_comp,nx,ny))
    for i in range(0, len(indx_amp)):   
        flux,vel,sigma,ew=get_fluxline(basename.replace('NAME',name),path=path_map,ind1=indx_amp[i],ind2=indx_fwh[i],ind3=indx_vel[i],ind4=indx_con,lo=lo,zt=0.0,val0=0)
        fluxT[i,:,:]=flux
        velT[i,:,:]=vel
        sigmaT[i,:,:]=sigma
        ewT[i,:,:]=ew
    cont1=pdl_cube[indx_con,:,:]
    indx = np.where((cont1 == 0) | (np.isfinite(cont1) == False))
    indxt = np.where((np.isfinite(cont1)))
    mapE=pdl_cube[indx_noi,:,:]
    mapT=np.nansum(fluxT,axis=0)
    nx,ny=mapT.shape
    mapT[indx]=np.nan
    cont1[indx]=np.nan
    mintc=np.nanmin(cont1)# We define the lowest continuum value as the one for which we set the map to NaN, to avoid problems with the logarithm and the normalization. This is because in some cases there are very low continuum values that produce very high flux/continuum ratios, which are not realistic.
    mapT[np.where(cont1==mintc)]=np.nan
    if set_am:
        #Use the Amplitude of the briad component to define the usefull spaxels, AmpT is the threshold for the amplitude value defined in MapLine, below which the map is set to NaN. This is because in some cases there are very low amplitude values that produce very high flux/continuum ratios, which are not realistic.
        indx_amp=get_map_param(hdr,keymatch='HaBroad1_Amplitude')
        amp_val=pdl_cube[indx_amp,:,:]
        amp_val=np.round(amp_val,3)
        mapT[np.where(amp_val == AmpT)]=np.nan
    if pow_cr:
        #Use the power line continum to define the usefull spaxels
        try:
            # We define the lowest power line continuum value as the one for which we set the map to NaN, to avoid problems with the logarithm and the normalization. This is because in some cases there are very low continuum values that produce very high flux/continuum ratios, which are not realistic.
            indx_pow=get_map_param(hdr,keymatch='Amp_powerlow')
            amp_pow=pdl_cube[indx_pow,:,:]
            mintc=np.nanmin(amp_pow)
            mapT[np.where(amp_pow==mintc)]=np.nan
        except:
            pass
    #mapT=np.log10(mapT)
    mapT[np.where(np.isfinite(mapT) == False)]=-2
    #map[0:4,0:ny]=0
    #map[nx-4:nx,0:ny]=0
    #map[0:nx,0:4]=0
    #map[0:nx,ny-4:ny]=0
    maxval=np.nanmax(mapT[indxt])
    minval=np.nanmin(mapT[indxt])
    if verbose:
        print(maxval,minval,'Map0')
    #ict=plt.imshow(mapT)
    #cbar=plt.colorbar(ict)
    #plt.show()
    mapT=filtNd(mapT,sigma=sigmat)
    maxval=np.nanmax(mapT[indxt])
    minval=0#np.nanmin(map[indxt])
    if verbose:
        print(maxval,minval,'Map1')
    #ict=plt.imshow(mapT)
    #cbar=plt.colorbar(ict)
    #plt.show()
    mapT=(mapT-minval)/(maxval-minval)*1+0
    maxval=np.nanmax(mapT[indxt])
    minval=np.nanmin(mapT[indxt])
    if verbose:
        print(maxval,minval,'Map2')
    mapT[np.where(np.isfinite(mapT) == False)]=minval
    mapT[indx]=minval
    mapT[np.where(mapT < 0)]=minval
    #ict=plt.imshow(mapT)
    #cbar=plt.colorbar(ict)
    #plt.show()
    nx,ny=mapT.shape
    maxvalE=np.nanmax(mapE[indxt])
    minvalE=np.nanmin(mapE[indxt])
    if verbose:
        print(minvalE,maxvalE,'Err')
    mapE=(mapE-minvalE)/(maxvalE-minvalE)*0.5+0
    mapE=filtNd(mapE,sigma=sigmat)
    mapE[indx]=0.01
    mapE[np.where(np.isfinite(mapE) == False)]=0.01
    #mapE[0:4,0:ny]=0.02
    #mapE[nx-4:nx,0:ny]=0.02
    #mapE[0:nx,0:4]=0.02
    #mapE[0:nx,ny-4:ny]=0.02
    #mapE[np.where(mapE < 0)]=0.02

    #ict=plt.imshow(mapE)
    #cbar=plt.colorbar(ict)
    #plt.show()
    sycall('mkdir -p '+path_out)
    map_to_stl(mapT*25.34, psfmbasename.replace('NAME',name), path_out=path_out+'/')
    keys=list(hdr.keys())
    h1=fits.PrimaryHDU(mapT)
    h2=fits.ImageHDU(mapE)
    h=h1.header
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h[keys[i]]=hdr[keys[i]]
            h.comments[keys[i]]=hdr.comments[keys[i]]
    h.update() 
    head_list=[h1,h2]
    hlist=fits.HDUList(head_list)
    hlist.update_extend()
    filet=path_out+'/'+psfmbasename.replace('NAME',name)+'.fits'
    hlist.writeto(filet,overwrite=True)
    sycall('gzip -f '+filet)    


def numpy_to_tform(arr):
    """
    Convert a NumPy array dtype into a FITS table TFORM code.

    This function maps the dtype and shape of a NumPy array to the
    corresponding FITS table format descriptor used in binary tables.
    The output string can be used as the TFORMn value when defining
    FITS table columns.

    Parameters
    ----------
    arr : array-like
        Input array or object convertible to a NumPy array. The dtype
        determines the FITS format code, while the array dimensionality
        determines whether the column is scalar or vector.

    Returns
    -------
    str
        FITS TFORM code describing the data type and size of the column.
        Examples include 'E', 'D', 'J', 'K', '5E', '20A', etc.

    Notes
    -----
    The following dtype mappings are supported:

    =================  ============
    NumPy dtype        FITS code
    =================  ============
    float32            E
    float64            D
    int16              I
    int32              J
    int64              K
    uint8              B
    bool               L
    =================  ============

    String arrays are converted to FITS ASCII format using the length
    of the string (e.g., '20A').

    For multidimensional arrays:
    - 1D arrays produce a scalar column format (e.g., 'E', 'J').
    - 2D arrays produce a vector column format (e.g., '5E').
    - Arrays with ndim > 2 are not supported by FITS tables.

    Raises
    ------
    ValueError
        If the dtype is not supported or if the array has more than
        two dimensions.

    Examples
    --------
    >>> numpy_to_tform(np.array([1.0, 2.0], dtype=np.float32))
    'E'

    >>> numpy_to_tform(np.ones((10, 5), dtype=np.float32))
    '5E'

    >>> numpy_to_tform(np.array(["abc", "def"]))
    '12A'
    """
    arr = np.asarray(arr)

    # tipo base
    base = arr.dtype

    fits_map = {
        np.dtype('float32'): 'E',
        np.dtype('float64'): 'D',
        np.dtype('int16'):   'I',
        np.dtype('int32'):   'J',
        np.dtype('int64'):   'K',
        np.dtype('uint8'):   'B',
        np.dtype('bool'):    'L',
        np.dtype('>i8'):     'K',
        np.dtype('>f8'):     'D'
    }

    if base.kind in ['U', 'S']:
        # strings
        strlen = arr.dtype.itemsize
        return f'{strlen}A'

    if base not in fits_map:
        raise ValueError(f'Dtype no soportado: {base}')

    code = fits_map[base]

    # escalar o vector
    if arr.ndim == 1:
        return code
    elif arr.ndim == 2:
        return f'{arr.shape[1]}{code}'
    else:
        raise ValueError("FITS no soporta ndim > 2 en tablas")


def simpson_r(f,x,i1,i2,typ=0):
    """
    Compute the numerical integral of a function using Simpson's rule
    over a specified index range.

    This implementation applies the composite Simpson's 1/3 rule on
    uniformly spaced samples between indices `i1` and `i2`. If the number
    of intervals is odd, it is automatically increased by one to satisfy
    Simpson's requirement of an even number of intervals.

    Parameters
    ----------
    f : ndarray
        1D array of function values to be integrated.
    x : ndarray
        1D array of coordinates corresponding to `f`. It is assumed to be
        ordered and approximately uniformly spaced within the integration
        range.
    i1 : int
        Starting index of the integration interval.
    i2 : int
        Ending index of the integration interval (inclusive).
    typ : int, optional
        Output type:
        - 0 : Return the integral value (default).
        - 1 : Return the average value over the interval, i.e.,
              integral divided by (x[i2] - x[i1]).

    Returns
    -------
    float
        Numerical integral of `f` over the interval [x[i1], x[i2]] if
        `typ=0`, or the mean value over the interval if `typ=1`.

    Notes
    -----
    - Simpson's rule requires an even number of intervals. If the number
      of intervals (`i2 - i1`) is odd, the function extends the upper
      limit by one index.
    - The spacing `h` is computed as (x[i2] - x[i1]) / n, assuming
      approximately uniform sampling.
    - No explicit checks are performed to ensure uniform spacing or
      valid index bounds.
    - The function does not handle NaNs or masked values explicitly.

    Raises
    ------
    IndexError
        If `i1` or `i2` are outside the valid range of the input arrays.

    See Also
    --------
    scipy.integrate.simpson : More robust and general Simpson integration.

    """
    n=(i2-i1)*1.0
    if n % 2:
        n=n+1.0
        i2=i2+1
    b=x[i2]
    a=x[i1]
    h=(b-a)/n
    s= f[i1]+f[i2]
    n=int(n)
    dx=b-a
    for i in range(1, n, 2):
        s += 4 * f[i1+i]
    for i in range(2, n-1, 2):
        s += 2 * f[i1+i]
    if typ == 0:
        return s*h/3.0
    if typ == 1:
        return s*h/3.0/dx        