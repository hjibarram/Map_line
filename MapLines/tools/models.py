#!/usr/bin/env python
import MapLines.tools.tools as tol
import numpy as np

def emission_line_model(x, xo=[5100], A=[1.0], dv=[0.0], fwhm=[200.0], alph=[0.0], gam=[0.0], skew=False, lorentz=False, voigt=False):
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
        

def line_model(theta, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=0, ret_com=False, skew=False, lorentz=False, outflow=False, voigt=False):
    '''Model for the line complex'''

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
        
    
    lin=0
    for i in range(len(ModA)):
        if outflow:
            lin=ModA[i]+ModAo[i]+lin
        else:
            lin=ModA[i]+lin
    outvect=[]
    outvect.extend([lin])
    for i in range(len(ModA)):
        outvect.extend([ModA[i]])
    if outflow:
        for i in range(len(ModAo)):
            outvect.extend([ModAo[i]])       
    
    if ret_com:
        return outvect
    else:
        return lin