#!/usr/bin/env python
"""
MapLines.tools.models
=====================

Spectral line models used by MapLines.

This module defines the spectral models used to fit emission lines in
astronomical spectra. The models are used by the MCMC sampler to evaluate
the likelihood of the observed data given a set of model parameters.

Two main levels of modeling are implemented:

1. emission_line_model
   Computes individual emission-line profiles (Gaussian, Lorentzian,
   Voigt, or skewed Gaussian).

2. line_model
   Builds the full spectral model combining multiple emission lines,
   optional outflow components, continuum power-law emission, and
   FeII templates.

These models are used in combination with the fitting routines in
``MapLines.tools.line_fit`` and the likelihood functions defined in
``MapLines.tools.priors``.
"""
import MapLines.tools.tools as tol
import numpy as np 

def emission_line_model(x, xo=[5100], A=[1.0], dv=[0.0], fwhm=[200.0], alph=[0.0], gam=[0.0], skew=False, lorentz=False, voigt=False):
    """
    Compute emission-line profiles.

    This function generates emission-line profiles for a set of spectral
    lines using different profile shapes (Gaussian, skewed Gaussian,
    Lorentzian, or Voigt).

    Parameters
    ----------
    x : array-like
        Wavelength grid where the model is evaluated.
    xo : list of float
        Rest-frame wavelengths of the emission lines.
    A : list of float
        Amplitudes of each emission-line component.
    dv : list of float
        Velocity shifts relative to the rest-frame wavelength (km/s).
    fwhm : list of float
        Full width at half maximum (km/s) of each line.
    alph : list of float
        Skewness parameter (used when ``skew=True``).
    gam : list of float
        Lorentzian width parameter for Voigt profiles.
    skew : bool
        Use skewed Gaussian profiles.
    lorentz : bool
        Use Lorentzian profiles.
    voigt : bool
        Use Voigt profiles.

    Returns
    -------
    list of ndarray
        List containing the model profile of each emission line.

    Notes
    -----
    The line width is converted from velocity space (km/s) to wavelength
    dispersion using:

        sigma = FWHM / c * lambda / (2 * sqrt(2 ln 2))

    where ``c`` is the speed of light.
    """
    ct=299792.458
    model_out=[]
    for i in range(len(dv)):
        sigma=fwhm[i]/ct*xo[i]/(2.0*np.sqrt(2.0*np.log(2.0)))
        xm=xo[i]*(1.0+dv[i]/ct)
        #if i > 0:
        #    A1=A/fac[i-1]
        #else:
        A1=A[i]
        if skew:
            alp=alph[i]
            model=tol.gauss_K(x,sigma=sigma,xo=xm,A1=A1,alp=alp)
        else:
            if lorentz:
                model=tol.lorentz(x,sigma=(sigma*(2.0*np.sqrt(2.0*np.log(2.0)))),xo=xm,A1=A1)
            elif voigt:
                if len(gam) > 0:
                    gam1=gam[i]
                    model=tol.voigt(x,sigma=sigma,xo=xm,A1=A1,gam1=gam1)
                else:
                    model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
            else:
                model=tol.gauss_M(x,sigma=sigma,xo=xm,A1=A1)
        model_out.extend([model])
    #if len(model_out) == 1:
    #    return model_out[0]
    #else:
    return model_out
        
def line_model(theta, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0,
    names0, n_lines, vals, x=0, powlaw=False, feii=False, data=None, ret_com=False,
    skew=False, lorentz=False, outflow=False, voigt=False, backend="auto", vectorized=False):        

    """
    Construct the full spectral model for an emission-line complex.

    This function combines multiple emission-line components into a
    single spectral model. The parameters are provided as a vector
    (``theta``) sampled by the MCMC fitting procedure.

    The model may include:

    - Narrow emission lines
    - Optional outflow components
    - Power-law continuum
    - FeII emission templates

    Parameters
    ----------
    theta : array-like
        Parameter vector defining amplitudes, velocities, widths,
        and additional model parameters.
    waves0 : list
        Rest-frame wavelengths of emission lines.
    fac0, facN0 : list
        Scaling relations between line amplitudes.
    velfac0, velfacN0 : list
        Relations between velocity parameters.
    fwhfac0, fwhfacN0 : list
        Relations between line widths.
    names0 : list
        Names of emission lines.
    n_lines : int
        Number of emission lines in the model.
    vals : list
        Names of free parameters.
    x : ndarray
        Wavelength grid where the model is evaluated.

    Optional Components
    -------------------
    powlaw : bool
        Include a power-law continuum component.
    feii : bool
        Include FeII emission templates.
    outflow : bool
        Include outflow components for emission lines.
    skew : bool
        Use skewed Gaussian profiles.
    lorentz : bool
        Use Lorentzian profiles.
    voigt : bool
        Use Voigt profiles.

    Returns
    -------
    ndarray or list
        If ``ret_com=False``, returns the full model spectrum.

        If ``ret_com=True``, returns a list containing:
        - total model spectrum
        - individual emission-line components
        - optional continuum and FeII components

    Notes
    -----
    This function is the core spectral model evaluated during the MCMC
    sampling performed in ``MapLines.tools.mcmc``.
    """
    if vectorized:

        if ret_com:
            raise ValueError(
                "ret_com=True is only supported "
                "for non-vectorized model evaluation."
            )

        return _line_model_vectorized(
            theta, waves0, fac0, facN0,
            velfac0, velfacN0,
            fwhfac0, fwhfacN0,
            names0, n_lines, vals,
            x=x,
            powlaw=powlaw,
            feii=feii,
            data=data,
            skew=skew,
            lorentz=lorentz,
            outflow=outflow,
            voigt=voigt,
            backend=backend
        )
    # A partir de aquí continúa el código original
    # de line_model(), sin ninguna modificación.

    alph=[]
    alphb=[]
    gam=[]
    if skew:
        *f_parm,alp1,alpb=theta 
    else:
        if outflow:
            *f_parm,F1o,dvo,fwhmo,alpho=theta
            A1o=[]
            dvO=[]
            fwhmO=[]
            alphO=[]
        else:
            if voigt:
                *f_parm,gam1=theta
            else:
                f_parm=theta
    if powlaw:
        if feii:
            *f_parm,P0,Pa0,Fes,Fde,FA=theta
        else:
            *f_parm,P0,Pa0=theta
    else:
        if feii:
            *f_parm,Fes,Fde,FA=theta


    A1=[]
    dv=[]
    fwhm=[]
    for myt in range(0,n_lines):            
        inNaM=facN0[myt]
        velinNaM=velfacN0[myt]
        fwhinNaM=fwhfacN0[myt]
        valname='None'
        velvalname='None'
        fwhvalname='None'
        indf=-1
        velindf=-1
        fwhindf=-1
        vt1='AoN'.replace('N',str(myt))
        vt2='dvoN'.replace('N',str(myt))
        vt3='fwhmoN'.replace('N',str(myt))
        for atp in range(0, len(names0)):
            if names0[atp] == inNaM:
                valname='AoN'.replace('N',str(atp))
            if names0[atp] == velinNaM:
                velvalname='dvoN'.replace('N',str(atp))        
            if names0[atp] == fwhinNaM:
                fwhvalname='fwhmoN'.replace('N',str(atp))       
        for atp in range(0, len(vals)):
            if vals[atp] == valname:
                indf=atp
            if vals[atp] == velvalname:
                velindf=atp     
            if vals[atp] == fwhvalname:
                fwhindf=atp     
        if indf >= 0:
            A1.extend([f_parm[indf]/fac0[myt]])
            if outflow:
                A1o.extend([f_parm[indf]/fac0[myt]*F1o])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt1:
                    indfT1=atp
            A1.extend([f_parm[indfT1]])
            if outflow:
                A1o.extend([f_parm[indfT1]*F1o])
        if velindf >= 0:
            dv.extend([f_parm[velindf]*velfac0[myt]])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt2:
                    indfT2=atp  
            dv.extend([f_parm[indfT2]])              
        if fwhindf >= 0:
            fwhm.extend([f_parm[fwhindf]*fwhfac0[myt]])
        else: 
            for atp in range(0, len(vals)):
                if vals[atp] == vt3:
                    indfT3=atp   
            fwhm.extend([f_parm[indfT3]])                 
        if skew:
            alph.extend([alp1])
            alphb.extend([alpb])
        else:
            if outflow:
                dvO.extend([dvo])
                fwhmO.extend([fwhmo])
                alphO.extend([alpho])
            else:
                if voigt:
                    gam.extend([gam1])


    ModA=emission_line_model(x, xo=waves0, A=A1, dv=dv ,fwhm=fwhm, alph=alph, gam=gam, skew=skew, lorentz=lorentz, voigt=voigt)
    if outflow:
        ModAo=emission_line_model(x, xo=waves0, A=A1o, dv=dvO ,fwhm=fwhmO, alph=alphO, skew=True)
        
    if powlaw:
        cont=tol.spow_law(x, A=P0, alpha=Pa0, xo=5100.0)
    else:
        cont=x*0.0
    if feii:
        feiis=tol.opticFeII(x, data ,sigma=Fes, xo=Fde, A1=FA)
    else:
        feiis=x*0.0

    lin=0
    for i in range(len(ModA)):
        if outflow:
            lin=ModA[i]+ModAo[i]+lin
        else:
            lin=ModA[i]+lin
    lin=lin+cont+feiis
    outvect=[]
    outvect.extend([lin])
    for i in range(len(ModA)):
        outvect.extend([ModA[i]])
    if outflow:
        for i in range(len(ModAo)):
            outvect.extend([ModAo[i]])       
    if powlaw:
        outvect.extend([cont])
    if feii:
        outvect.extend([feiis])    
    if ret_com:
        return outvect
    else:
        return lin



def _line_model_vectorized(
    thetas, waves0, fac0, facN0, velfac0, velfacN0,
    fwhfac0, fwhfacN0, names0, n_lines, vals,
    x=0, powlaw=False, feii=False, data=None,
    skew=False, lorentz=False, outflow=False, voigt=False,
    backend="auto"):
    """
    Evaluate complete spectral models for a batch of parameter vectors.

    This routine is the vectorized counterpart of ``line_model`` and is
    intended for posterior evaluations during MCMC sampling. It preserves
    the existing parameter conventions, line dependencies, and spectral
    profile definitions without constructing individual plot components.

    Parameters
    ----------
    thetas : array_like
        Parameter matrix of shape (nwalkers_batch, ndim).
    waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0 : array_like
        Rest wavelengths and amplitude, velocity, and width relations,
        following the conventions of ``line_model``.
    names0 : sequence of str
        Emission-line names used when resolving tied parameters.
    n_lines : int
        Number of emission lines.
    vals : sequence of str
        Names of the free parameters.
    x : array_like, optional
        One-dimensional wavelength grid.
    powlaw, feii, skew, lorentz, outflow, voigt : bool, optional
        Model configuration flags with the same meanings as in
        ``line_model``.
    data : array_like or None, optional
        FeII template with wavelength and flux in columns 0 and 1.
    backend : {'auto', 'cpu', 'cuda'}, optional
        Numerical backend. Default is 'auto'.

    Returns
    -------
    models : numpy.ndarray or cupy.ndarray
        Total modeled flux for each walker with shape
        (nwalkers_batch, nwavelength). The array stays on the selected
        device; no host transfer is performed here.

    Raises
    ------
    ValueError
        If input shapes, names, or parameter indices are invalid.
    NotImplementedError
        If CUDA is selected for Voigt profiles, which require a
        validated GPU equivalent of ``scipy.special.voigt_profile``.

    Notes
    -----
    This function computes only the total spectral model used by the
    likelihood. The original scalar ``line_model`` remains responsible
    for the component spectra returned when ``ret_com=True``.

    FeII smoothing is performed per walker because its width varies
    between parameter vectors. This preserves the original interpolation
    and Gaussian filtering steps but may limit acceleration for FeII fits.
    """
    from MapLines.tools.backend import get_backend

    xp, use_cuda = get_backend(backend)
    theta = xp.asarray(thetas, dtype=xp.float64)
    wave_grid = xp.asarray(x, dtype=xp.float64)
    if theta.ndim != 2:
        raise ValueError("thetas must have shape (batch, ndim)")
    if wave_grid.ndim != 1:
        raise ValueError("x must be a one-dimensional wavelength array")
    if len(waves0) != n_lines:
        raise ValueError("n_lines must equal len(waves0)")

    if skew and outflow:
        raise ValueError("The original model does not support skew=True with outflow=True")

    # MapLine's original parameter extraction: optional continuum/FeII
    # parameters override the trailing free-parameter segment.
    npar = theta.shape[1]
    base_end = npar
    if skew:
        if npar < 2:
            raise ValueError("Missing skew parameters")
        alp1, alpb = theta[:, -2], theta[:, -1]
        base_end = npar - 2
    elif outflow:
        if npar < 4:
            raise ValueError("Missing outflow parameters")
        F1o, dvo, fwhmo, alpho = (
            theta[:, -4], theta[:, -3], theta[:, -2], theta[:, -1]
        )
        base_end = npar - 4
    elif voigt:
        if npar < 1:
            raise ValueError("Missing Voigt parameter")
        gam1 = theta[:, -1]
        base_end = npar - 1

    if powlaw and feii:
        if npar < 5:
            raise ValueError("Missing continuum and FeII parameters")
        P0, Pa0, Fes, Fde, FA = [theta[:, j] for j in range(npar-5, npar)]
        base_end = npar - 5
    elif powlaw:
        if npar < 2:
            raise ValueError("Missing continuum parameters")
        P0, Pa0 = theta[:, -2], theta[:, -1]
        base_end = npar - 2
    elif feii:
        if npar < 3:
            raise ValueError("Missing FeII parameters")
        Fes, Fde, FA = theta[:, -3], theta[:, -2], theta[:, -1]
        base_end = npar - 3

    f_parm = theta[:, :base_end]
    # Match the original loops: when names repeat, the last match wins.
    name_index = {name: i for i, name in enumerate(names0)}
    value_index = {name: i for i, name in enumerate(vals)}

    def linked_parameter(i, names, factors, prefix, divide=False):
        referenced = names[i]
        ref_i = name_index.get(referenced)
        key = f"{prefix}{ref_i}" if ref_i is not None else None
        if key in value_index:
            value = f_parm[:, value_index[key]]
            return value / factors[i] if divide else value * factors[i]
        key = f"{prefix}{i}"
        if key not in value_index:
            raise ValueError(f"Missing free parameter: {key}")
        return f_parm[:, value_index[key]]

    c = 299792.458
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    xrow = wave_grid[None, :]
    model = xp.zeros((theta.shape[0], wave_grid.size), dtype=xp.float64)

    if use_cuda:
        from cupyx.scipy.special import erf
    else:
        from scipy.special import erf, voigt_profile

    def profile(amplitude, velocity, width, restwave,
                shape="gaussian", alpha=None, gamma=None):
        sigma = width * restwave / (c * fwhm_factor)
        center = restwave * (1.0 + velocity / c)
        amp = amplitude[:, None]
        sig = sigma[:, None]
        mu = center[:, None]
        if shape == "skew":
            alp = alpha[:, None]
            delta = alp / xp.sqrt(xp.abs(1.0 + alp**2))
            shifted = mu - sig * delta * xp.sqrt(2.0 / xp.pi)
            omega = xp.sqrt(sig**2 / (1.0 - 2.0*delta**2/xp.pi))
            t = (xrow - shifted) / omega
            return amp * xp.exp(-0.5*t**2) * (1.0 + erf(alp*t/xp.sqrt(2.0)))
        if shape == "lorentz":
            half = width[:, None] * restwave / (2.0*c)
            return amp * half**2 / ((xrow-mu)**2 + half**2)
        if shape == "voigt":
            if use_cuda:
                raise NotImplementedError(
                    "Voigt profiles are not supported by the CUDA "
                    "backend in this implementation."
                )
            g = gamma[:, None]
            return amp * voigt_profile(xrow-mu, sig, g) / voigt_profile(0, sig, g)
        return amp * xp.exp(-0.5*(xrow-mu)**2 / sig**2)

    if skew:
        primary_shape = "skew"
    elif lorentz:
        primary_shape = "lorentz"
    elif voigt and not outflow:
        primary_shape = "voigt"
    else:
        primary_shape = "gaussian"

    for i in range(n_lines):
        amplitude = linked_parameter(i, facN0, fac0, "Ao", divide=True)
        velocity = linked_parameter(i, velfacN0, velfac0, "dvo")
        width = linked_parameter(i, fwhfacN0, fwhfac0, "fwhmo")
        model += profile(
            amplitude, velocity, width, waves0[i],
            shape=primary_shape,
            alpha=alp1 if skew else None,
            gamma=gam1 if (voigt and not skew and not outflow) else None,
        )
        if outflow:
            # Original outflows use independent dv/FWHM and gauss_K.
            model += profile(
                amplitude * F1o, dvo, fwhmo, waves0[i],
                shape="skew", alpha=alpho,
            )

    if powlaw:
        model += P0[:, None] * (xrow / 5100.0)**(-Pa0[:, None])

    if feii:
        if data is None:
            raise ValueError("FeII modeling requires template data")
        from scipy.ndimage import gaussian_filter1d as cpu_filter
        if use_cuda:
            from cupyx.scipy.ndimage import gaussian_filter1d as gpu_filter
        filter1d = gpu_filter if use_cuda else cpu_filter
        template = xp.asarray(data, dtype=xp.float64)
        temp_wave = template[:, 0]
        temp_flux = template[:, 1]
        norm_flux = temp_flux / xp.nanmax(temp_flux)
        for j in range(theta.shape[0]):
            # Interpolation and smoothing order matches opticFeII.
            shifted = temp_wave + Fde[j]
            interp = xp.interp(wave_grid, shifted, norm_flux, left=0.0, right=0.0)
            smoothed = filter1d(interp * FA[j], sigma=float(Fes[j]))
            model[j, :] += smoothed

    return model



"""
def line_model_batch(
    thetas, waves0, fac0, facN0, velfac0, velfacN0,
    fwhfac0, fwhfacN0, names0, n_lines, vals,
    x=0, powlaw=False, feii=False, data=None,
    ret_com=False, skew=False, lorentz=False,
    outflow=False, voigt=False, backend="auto"
):
    ""
    Construct spectral models for multiple parameter vectors.

    This function provides a vectorized implementation of
    ``line_model`` for evaluating multiple MCMC walkers
    simultaneously.

    The spectral models preserve the amplitude, velocity,
    and line-width scaling relations defined by the
    original MapLine implementation.

    The current implementation supports Gaussian emission-line
    profiles. Additional spectral components will be incorporated
    progressively.

    Parameters
    ----------
    thetas : array_like
        Model parameter vectors with shape
        (nwalkers_batch, ndim).

    waves0 : array_like
        Rest-frame wavelengths of emission lines.

    fac0, facN0 : array_like
        Scaling relations between line amplitudes.

    velfac0, velfacN0 : array_like
        Relations between velocity parameters.

    fwhfac0, fwhfacN0 : array_like
        Relations between line widths.

    names0 : list
        Names of emission lines.

    n_lines : int
        Number of emission lines in the model.

    vals : list
        Names of free parameters.

    x : array_like
        Wavelength grid where the models are evaluated.

    powlaw : bool, optional
        Include a power-law continuum component.

    feii : bool, optional
        Include FeII emission templates.

    data : optional
        Additional model data.

    ret_com : bool, optional
        Return individual spectral components.

    skew : bool, optional
        Use skewed Gaussian profiles.

    lorentz : bool, optional
        Use Lorentzian profiles.

    outflow : bool, optional
        Include outflow components.

    voigt : bool, optional
        Use Voigt profiles.

    backend : {'auto', 'cpu', 'cuda'}, optional
        Numerical execution backend.

        'auto' selects CUDA when available.
        'cpu' forces NumPy execution.
        'cuda' requires a functional CUDA device.

        Default is 'auto'.

    Returns
    -------
    models : ndarray
        Spectral models with shape
        (nwalkers_batch, nwavelength).

        The returned array resides in CPU or GPU memory,
        according to the selected backend.

        When `ret_com=True`, returns a list containing
        the total model followed by individual emission-line
        components. Each component has shape
        (nwalkers_batch, nwavelength).

    Raises
    ------
    ValueError
        If the input parameter or wavelength dimensions
        are invalid.

    NotImplementedError
        If a spectral component not yet supported by the
        vectorized implementation is requested.

    Notes
    -----
    This function preserves the original Gaussian model:

        sigma = FWHM / c * lambda / (2 * sqrt(2 ln 2))

        lambda_center = lambda * (1 + dv / c)

    where FWHM and dv are expressed in km/s.

    Amplitude, velocity, and width parameter dependencies
    are constructed using the same relations implemented
    in ``line_model``.

    The original ``line_model`` remains unchanged and
    continues to support all available spectral components.

    Automatic backend selection does not imply support
    for all spectral models.

    Examples
    --------
    Evaluate models on the CPU:

    >>> models = line_model_batch(
    ...     thetas, waves0, fac0, facN0,
    ...     velfac0, velfacN0,
    ...     fwhfac0, fwhfacN0,
    ...     names0, n_lines, vals,
    ...     x=x, backend="cpu"
    ... )
    ""
    from MapLines.tools.backend import get_backend

    # Components not yet implemented in the vectorized model.
    if any((skew, lorentz, outflow, voigt, powlaw, feii)):
        raise NotImplementedError(
            "Vectorized support for the requested "
            "spectral components is not implemented."
        )

    xp, use_cuda = get_backend(backend)

    theta = xp.asarray(thetas)
    x = xp.asarray(x)

    if theta.ndim != 2:
        raise ValueError(
            "thetas must have shape (nwalkers_batch, ndim)"
        )

    if x.ndim != 1:
        raise ValueError(
            "x must be a one-dimensional wavelength array"
        )

    if len(waves0) != n_lines:
        raise ValueError(
            "n_lines must match the length of waves0"
        )

    # Physical constants, matching emission_line_model.
    ct = 299792.458
    fwhm_to_sigma = 2.0 * xp.sqrt(2.0 * xp.log(2.0))

    # Name-to-index relations used by the original model.
    name_index = {
        name: i for i, name in enumerate(names0)
    }

    value_index = {
        name: i for i, name in enumerate(vals)
    }

    def parameter_index(line_index, reference, prefix):
        "
        Resolve the parameter index for a spectral line.

        First search for a parameter tied to the referenced
        line. If no such parameter exists, use the parameter
        associated directly with the current line.
        ""
        ref_index = name_index.get(reference)

        if ref_index is not None:
            ref_name = "{}{}".format(prefix, ref_index)

            if ref_name in value_index:
                return value_index[ref_name], True

        local_name = "{}{}".format(prefix, line_index)

        return value_index[local_name], False

    model_components = []

    for i in range(n_lines):

        # Amplitude dependencies
        idx, tied = parameter_index(
            i, facN0[i], "Ao"
        )

        amp = theta[:, idx]

        if tied:
            amp = amp / fac0[i]

        # Velocity dependencies
        idx, tied = parameter_index(
            i, velfacN0[i], "dvo"
        )

        dv = theta[:, idx]

        if tied:
            dv = dv * velfac0[i]

        # Line-width dependencies
        idx, tied = parameter_index(
            i, fwhfacN0[i], "fwhmo"
        )

        fwhm = theta[:, idx]

        if tied:
            fwhm = fwhm * fwhfac0[i]

        # Gaussian profile, matching the original model.
        wavelength = waves0[i]

        sigma = (
            fwhm / ct * wavelength / fwhm_to_sigma
        )

        center = wavelength * (1.0 + dv / ct)

        model = amp[:, None] * xp.exp(
            -0.5 * (
                (x[None, :] - center[:, None]) /
                sigma[:, None]
            )**2
        )

        model_components.append(model)

    # Sum the individual emission-line components.
    total_model = xp.zeros(
        (theta.shape[0], x.shape[0]),
        dtype=theta.dtype
    )

    for component in model_components:
        total_model = total_model + component

    if ret_com:
        return [total_model] + model_components

    return total_model
"""