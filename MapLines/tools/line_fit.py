#!/usr/bin/env python
import numpy as np
import MapLines
import MapLines.tools.models as mod
import MapLines.tools.mcmc as mcm
import MapLines.tools.tools as tol
import MapLines.tools.plot_tools as ptol
import MapLines.tools.priors as pri
from astropy.io import fits
import os
import os.path as ptt
import sys
from tqdm import tqdm

def line_fit_single(file1,file_out,file_out2,name_out2,dir_out='',colors=['blue','red','purple','brown','pink'],smoth=False,ker=2,config_lines='line_prop.yml',labplot=True,input_format='TableFits',z=0.05536,lA1=6450.0,lA2=6850.0,verbose=True,outflow=False,powlaw=False,feii=False,res_norm=True,voigt=False,lorentz=False,skew=False,error_c=True,ncpu=10,flux_f=1.0,erft=0.75,cont=False):
    pdl_data,pdl_dataE,wave=tol.get_oneDspectra(file1,flux_f=flux_f,erft=erft,input_format=input_format,error_c=error_c)
    if smoth:
        pdl_data=tol.conv(pdl_data,ke=ker)
    nz=len(pdl_data)
    pdl_data=pdl_data*flux_f     
    wave_f=wave/(1+z)
    nw=np.where((wave_f >= lA1) & (wave_f <= lA2))[0]
    wave_i=wave_f[nw]
    model_all=np.zeros(len(nw))
    model_Inp=np.zeros(len(nw))
    model_InpE=np.zeros(len(nw))
    if outflow:
        model_Outflow=np.zeros(len(nw))
    valsp,n_lines,wavec1,wavec2,Inpvalues,Infvalues,Supvalues,waves0,names0,vals0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,vals,valsL,valsH=tol.get_priorsvalues(config_lines)   
    model_Ind=np.zeros([len(nw),n_lines])  
    if cont:
        oft=2
    else:
        oft=1
    if skew:
        model_param=np.zeros(n_lines*3+2+oft)
    else:
        if outflow:
            model_param=np.zeros(n_lines*3+4+oft)
        else:
            model_param=np.zeros(n_lines*3+oft)
    dataFe=None        
    if powlaw:
        if feii:
            model_param=np.zeros([n_lines*3+5+oft,nx,ny])
            dirFe=os.path.join(MapLines.__path__[0], 'data')+'/'
            dataFe=np.loadtxt(dirFe+'FeII.dat')
        else:
            model_param=np.zeros([n_lines*3+2+oft,nx,ny])
            dataFe=None        
    model_param[:]=np.nan    

    for i in range(0, 1):
        for j in range(0, 1):
            #val=1
            #if val == 1:
            fluxt=pdl_data[nw]
            if error_c:
                fluxtE=pdl_dataE[nw]
            else:
                fluxtE=tol.step_vect(fluxt,sp=50)
            if cont:
                #Defining the continum windows
                nwt=np.where((wave_f[nw] >= wavec1) & (wave_f[nw] <= wavec2))[0]  
                fluxpt=np.nanmean(fluxt[nwt])  
                fluxt=fluxt-fluxpt
            fluxe_t=np.nanmean(fluxtE)
            #Defining the input data for the fitting model
            data = (fluxt, fluxtE, wave_i, Infvalues, Supvalues, valsp, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow)
            nwalkers=240
            niter=1024
            #Defining the initian conditions
            if skew:  
                initial = np.array([*Inpvalues, 0.0, 0.0])
            else:
                if outflow:
                    initial = np.array([*Inpvalues, valsp['f1o'], valsp['dvOo'], valsp['fwhmOo'], valsp['alpOo']])
                else:
                    if voigt:
                        initial = np.array([*Inpvalues, valsp['gam1']])
                    else:
                        initial = np.array([*Inpvalues])

            ndim = len(initial)
            p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
            tim=True #allways allow to print the time stamp
            sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)  
            samples = sampler.flatchain
            theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                
            if skew:
                *f_parm,alph1_f,alphB_f=theta_max
                model,*modsI=mod.line_model(theta_max, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, ret_com=True,  skew=skew)
            else:
                if outflow:
                    *f_parm,F1o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                else:
                    if voigt:
                        *f_parm,gam1_f=theta_max
                    else:
                        gam1_f=0.0
                        f_parm=theta_max
                model,*modsI=mod.line_model(theta_max, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, ret_com=True, skew=skew, lorentz=lorentz, outflow=outflow)
 
            model_all[:]=model
            model_Inp[:]=fluxt
            model_InpE[:]=fluxtE
            for myt in range(0,n_lines):
                model_Ind[:,myt]=modsI[myt]
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
                    model_param[myt*3+0]=f_parm[indf]/fac0[myt]/flux_f
                else: 
                    for atp in range(0, len(vals)):
                        if vals[atp] == vt1:
                            indfT1=atp
                    model_param[myt*3+0]=f_parm[indfT1]/flux_f   
                if velindf >= 0:
                    model_param[myt*3+1]=f_parm[velindf]*velfac0[myt]
                else:      
                    for atp in range(0, len(vals)):
                        if vals[atp] == vt2:
                            indfT2=atp  
                    model_param[myt*3+1]=f_parm[indfT2]        
                if fwhindf >= 0:
                    model_param[myt*3+2]=f_parm[fwhindf]*fwhfac0[myt]
                else: 
                    for atp in range(0, len(vals)):
                        if vals[atp] == vt3:
                            indfT3=atp   
                    model_param[myt*3+2]=f_parm[indfT3]       
            model_param[n_lines*3]=fluxe_t/flux_f
            if cont:
                model_param[n_lines*3+1]=fluxpt/flux_f
                ind=n_lines*3+1
            else:
                ind=n_lines*3
            if skew:
                model_param[ind+1]=alph1_f
                model_param[ind+2]=alphB_f
            if outflow:
                model_param[ind+1]=F1o_f
                model_param[ind+2]=dvO_f
                model_param[ind+3]=fwhmO_f
                model_param[ind+4]=alphaO_f
            # Make plots
            ptol.plot_outputfits(wave_i,fluxt,fluxtE,model,modsI,n_lines,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,names0,vals,valsL,samples,res_norm=res_norm,colors=colors,name_out=name_out2,dir_out=dir_out,labplot=labplot,dataFe=dataFe,lorentz=lorentz,skew=skew,outflow=outflow,powlaw=powlaw,feii=feii)
            if verbose:    
                print('Best fit parameters:')
                linet=''
                for itar in range(0, len(vals)):
                    linet=linet+vals[itar]+'='+str(f_parm[itar])+' '
                if skew:
                    print(linet+'alph1='+str(alph1_f)+' alphB='+str(alphB_f))
                else:
                    if outflow:
                        print(linet+'F1o='+str(F1o_f)+' dvO='+str(dvO_f)+' fwhmO='+str(fwhmO_f)+' alph0='+str(alphaO_f))
                    else:
                        print(linet)  
    hli=[]
    hli.extend([fits.PrimaryHDU(model_all)])
    for myt in range(0,n_lines):
        temp=model_Ind[:,myt]
        hli.extend([fits.ImageHDU(temp)])
    hli.extend([fits.ImageHDU(model_Inp)])    
    hli.extend([fits.ImageHDU(model_InpE)])    
    if outflow:
        hli.extend([fits.ImageHDU(model_Outflow)])    
    h_k=hli[0].header
    h_k['EXTNAME'] ='Model'    
    h_k.update()
    for myt in range(0,n_lines):
        h_t=hli[1+myt].header
        h_t['EXTNAME'] ='N_Component'.replace('N',names0[myt])
        h_t.update()  
    h_y=hli[1+n_lines].header
    h_y['EXTNAME'] ='Input_Component'
    h_y.update()   
    h_y=hli[2+n_lines].header
    h_y['EXTNAME'] ='InputE_Component'
    h_y.update()  
    if outflow:
        h_y=hli[3+n_lines].header
        h_y['EXTNAME'] ='Outflow_Component'
        h_y.update()  
    hlist=fits.HDUList(hli)
    hlist.update_extend()
    hlist.writeto(file_out+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out+'.fits')
    

    h1=fits.PrimaryHDU(model_param)
    h=h1.header
    for i in range(0, len(valsH)):
        h['Val_'+str(i)]=valsH[i] 
    h['Val_'+str(n_lines*3)] ='Noise_Median'
    if cont:
        h['Val_'+str(n_lines*3+1)] ='Continum'
        ind=n_lines*3+1
    else:
        ind=n_lines*3
    if skew:
        h['Val_'+str(ind+1)]='Alpha_Narrow'
        h['Val_'+str(ind+2)]='Alpha_Broad' 
    if outflow: 
        h['Val_'+str(ind+1)]='Amp_Factor_outflow'
        h['Val_'+str(ind+2)]='Vel_outflow' 
        h['Val_'+str(ind+3)]='FWHM_outflow'
        h['Val_'+str(ind+4)]='Alpha_outflow'     
    h.update()        
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    hlist.writeto(file_out2+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out2+'.fits')


def line_fit(file1,file2,file3,file_out,file_out2,name_out2,dir_out='',colors=['blue','red','purple','brown','pink'],z=0.05536,j_t=0,i_t=0,powlaw=False,feii=False,labplot=True,config_lines='line_prop.yml',lA1=6450.0,lA2=6850.0,outflow=False,voigt=False,lorentz=False,skew=False,error_c=True,test=False,plot_f=True,ncpu=10,pgr_bar=True,flux_f=1.0,erft=0,cont=False,res_norm=True):
    pdl_cube,pdl_cubeE,mask,wave,hdr=tol.get_cubespectra(file1,file3,flux_f=flux_f,erft=erft,error_c=error_c)
    nz,nx,ny=pdl_cube.shape
    wave_f=wave/(1+z)
    nw=np.where((wave_f >= lA1) & (wave_f <= lA2))[0]
    wave_i=wave_f[nw]
    hdr["CRVAL3"]=wave_i[0]
    try:
        hdr["CD3_3"]=hdr["CD3_3"]/(1+z)
    except:
        hdr["CDELT3"]=hdr["CDELT3"]/(1+z)
    model_all=np.zeros([len(nw),nx,ny])
    model_Inp=np.zeros([len(nw),nx,ny])
    model_InpE=np.zeros([len(nw),nx,ny])
    if outflow:
        model_Outflow=np.zeros([len(nw),nx,ny])
    if powlaw:
        model_Powerlaw=np.zeros([len(nw),nx,ny])
    valsp,n_lines,wavec1,wavec2,Inpvalues,Infvalues,Supvalues,waves0,names0,vals0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,vals,valsL,valsH=tol.get_priorsvalues(config_lines)   
    model_Ind=np.zeros([len(nw),nx,ny,n_lines])    
    if cont:
        oft=2
    else:
        oft=1
    if skew:
        model_param=np.zeros([n_lines*3+2+oft,nx,ny])
    else:
        if outflow:
            model_param=np.zeros([n_lines*3+4+oft,nx,ny])
        else:
            model_param=np.zeros([n_lines*3+oft,nx,ny])
    dataFe=None        
    if powlaw:
        if feii:
            model_param=np.zeros([n_lines*3+5+oft,nx,ny])
            dirFe=os.path.join(MapLines.__path__[0], 'data')+'/'
            dataFe=np.loadtxt(dirFe+'FeII.dat')
        else:
            model_param=np.zeros([n_lines*3+2+oft,nx,ny])
            dataFe=None
    model_param[:,:,:]=np.nan    
    if pgr_bar:
        pbar=tqdm(total=nx*ny)
    for i in range(0, nx):
        for j in range(0, ny):
            val=mask[i,j]
            if test:
                if j_t*i_t == 0:
                    j_t=int(ny/2)
                    i_t=int(nx/2)
                i=i_t
                j=j_t
                print('testing spaxel '+str(i)+' , '+str(j))
            if val == 1:
                fluxt=pdl_cube[nw,i,j]
                if error_c:
                    fluxtE=pdl_cubeE[nw,i,j]
                else:
                	fluxtE=tol.step_vect(fluxt,sp=50)
                if cont:
                    #Defining the continum windows
                    nwt=np.where((wave_f[nw] >= wavec1) & (wave_f[nw] <= wavec2))[0]  
                    fluxpt=np.nanmean(fluxt[nwt]) 
                    if powlaw == False:
                        fluxt=fluxt-fluxpt
                fluxe_t=np.nanmean(fluxtE)
                #Defining the input data for the fitting model
                data = (fluxt, fluxtE, wave_i, Infvalues, Supvalues, valsp, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, skew, voigt, lorentz, outflow, powlaw, feii, dataFe)
                nwalkers=240
                niter=1024
                #Defining the initian conditions
                if skew:  
                    initial = np.array([*Inpvalues, 0.0, 0.0])
                else:
                    if outflow:
                        initial = np.array([*Inpvalues, valsp['f1o'], valsp['dvOo'], valsp['fwhmOo'], valsp['alpOo']])
                    else:
                        initial = np.array([*Inpvalues])
                if powlaw:
                    if feii:
                        initial = np.array([*Inpvalues, valsp['P1o'], valsp['P2o'], valsp['Fso'], valsp['Fdo'], valsp['Fao']])
                    else:
                        initial = np.array([*Inpvalues, valsp['P1o'], valsp['P2o']])
                #else:
                #    if feii:
                #        initial = np.array([*Inpvalues, valsp['Fso'], valsp['Fdo'], valsp['Fao']])
                
                ndim = len(initial)
                p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
                if plot_f:
                    tim=True
                else:
                    tim=False
                sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)  
                samples = sampler.flatchain
                theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                
                if skew:
                    *f_parm,alph1_f,alphB_f=theta_max
                    model,*modsI=mod.line_model(theta_max, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, ret_com=True,  skew=skew)
                else:
                    if outflow:
                        *f_parm,F1o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                    else:
                        f_parm=theta_max
                    model,*modsI=mod.line_model(theta_max, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, ret_com=True, skew=skew, lorentz=lorentz, outflow=outflow)
                if powlaw:
                    if feii:
                        *f_parm,P1o,P2o,Fso,Fdo,Fao=theta_max
                    else:
                        *f_parm,P1o,P2o=theta_max
                    model,*modsI=mod.line_model(theta_max, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, ret_com=True, powlaw=powlaw, feii=feii, data=dataFe)    
                
                model_all[:,i,j]=model
                model_Inp[:,i,j]=fluxt
                model_InpE[:,i,j]=fluxtE
                for myt in range(0,n_lines):
                    model_Ind[:,i,j,myt]=modsI[myt]
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
                        model_param[myt*3+0,i,j]=f_parm[indf]/fac0[myt]/flux_f
                    else: 
                        for atp in range(0, len(vals)):
                            if vals[atp] == vt1:
                                indfT1=atp
                        model_param[myt*3+0,i,j]=f_parm[indfT1]/flux_f   
                    if velindf >= 0:
                        model_param[myt*3+1,i,j]=f_parm[velindf]*velfac0[myt]
                    else:      
                        for atp in range(0, len(vals)):
                            if vals[atp] == vt2:
                                indfT2=atp  
                        model_param[myt*3+1,i,j]=f_parm[indfT2]        
                    if fwhindf >= 0:
                        model_param[myt*3+2,i,j]=f_parm[fwhindf]*fwhfac0[myt]
                    else: 
                        for atp in range(0, len(vals)):
                            if vals[atp] == vt3:
                                indfT3=atp   
                        model_param[myt*3+2,i,j]=f_parm[indfT3]       
                model_param[n_lines*3,i,j]=fluxe_t/flux_f
                if cont:
                    model_param[n_lines*3+1,i,j]=fluxpt/flux_f
                    ind=n_lines*3+1
                else:
                    ind=n_lines*3
                if skew:
                    model_param[ind+1,i,j]=alph1_f
                    model_param[ind+2,i,j]=alphB_f
                if outflow:
                    model_param[ind+1,i,j]=F1o_f
                    model_param[ind+2,i,j]=dvO_f
                    model_param[ind+3,i,j]=fwhmO_f
                    model_param[ind+4,i,j]=alphaO_f
                if powlaw:
                    model_param[ind+1,i,j]=P1o
                    model_param[ind+2,i,j]=P2o
                if feii:
                    model_param[ind+3,i,j]=Fso
                    model_param[ind+4,i,j]=Fdo
                    model_param[ind+5,i,j]=Fao
                if plot_f:
                    ptol.plot_outputfits(wave_i,fluxt,fluxtE,model,modsI,n_lines,waves0,fac0,facN0,velfac0,velfacN0,fwhfac0,fwhfacN0,names0,vals,valsL,samples,colors=colors,name_out=name_out2,dir_out=dir_out,labplot=labplot,dataFe=dataFe,lorentz=lorentz,skew=skew,outflow=outflow,powlaw=powlaw,feii=feii,res_norm=res_norm)    
                if pgr_bar == False:  
                    linet=''
                    for itar in range(0, len(vals)):
                        linet=linet+vals[itar]+'='+str(f_parm[itar])+' '
                    if skew:
                        print(linet+'alph1='+str(alph1_f)+' alphB='+str(alphB_f))
                    else:
                        if outflow:
                            print(linet+'F1o='+str(F1o_f)+' dvO='+str(dvO_f)+' fwhmO='+str(fwhmO_f)+' alph0='+str(alphaO_f))
                        else:
                            if powlaw:
                                if feii:
                                    print(linet+'P1o='+str(P1o)+' P2o='+str(P2o)+' Fso='+str(Fso)+' Fdo='+str(Fdo)+' Fao='+str(Fao))
                                else:
                                    print(linet+'P1o='+str(P1o)+' P2o='+str(P2o))
                            else:
                                print(linet)
                if test:
                    return        
            if pgr_bar:
                pbar.update(1)
    hli=[]
    hli.extend([fits.PrimaryHDU(model_all)])
    for myt in range(0,n_lines):
        temp=model_Ind[:,:,:,myt]
        hli.extend([fits.ImageHDU(temp)])
    hli.extend([fits.ImageHDU(model_Inp)])    
    hli.extend([fits.ImageHDU(model_InpE)])    
    if outflow:
        hli.extend([fits.ImageHDU(model_Outflow)])
    if powlaw:
        hli.extend([fits.ImageHDU(model_Powerlaw)])        
    h_k=hli[0].header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        try:
            h_k[keys[i]]=hdr[keys[i]]
            h_k.comments[keys[i]]=hdr.comments[keys[i]]
        except:
            continue
    h_k['EXTNAME'] ='Model'    
    h_k.update()
    for myt in range(0,n_lines):
        h_t=hli[1+myt].header
        for i in range(0, len(keys)):
            try:
                h_t[keys[i]]=hdr[keys[i]]
                h_t.comments[keys[i]]=hdr.comments[keys[i]]
            except:
                continue
        h_t['EXTNAME'] ='N_Component'.replace('N',names0[myt])
        h_t.update()  
    h_y=hli[1+n_lines].header
    for i in range(0, len(keys)):
        try:
            h_y[keys[i]]=hdr[keys[i]]
            h_y.comments[keys[i]]=hdr.comments[keys[i]]
        except:
            continue
    h_y['EXTNAME'] ='Input_Component'
    h_y.update()   
    h_y=hli[2+n_lines].header
    for i in range(0, len(keys)):
        try:
            h_y[keys[i]]=hdr[keys[i]]
            h_y.comments[keys[i]]=hdr.comments[keys[i]]
        except:
            continue
    h_y['EXTNAME'] ='InputE_Component'
    h_y.update()  
    if outflow:
        h_y=hli[3+n_lines].header
        for i in range(0, len(keys)):
            try:
                h_y[keys[i]]=hdr[keys[i]]
                h_y.comments[keys[i]]=hdr.comments[keys[i]]
            except:
                continue
        h_y['EXTNAME'] ='Outflow_Component'
        h_y.update()  
    if powlaw:
        h_y=hli[3+n_lines].header
        for i in range(0, len(keys)):
            try:
                h_y[keys[i]]=hdr[keys[i]]
                h_y.comments[keys[i]]=hdr.comments[keys[i]]
            except:
                continue
        h_y['EXTNAME'] ='PowerLaw_Component'
        h_y.update()      
    hlist=fits.HDUList(hli)
    hlist.update_extend()
    hlist.writeto(file_out+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out+'.fits')
    
    h1=fits.PrimaryHDU(model_param)
    h=h1.header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        try:
            if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
                h[keys[i]]=hdr[keys[i]]
                h.comments[keys[i]]=hdr.comments[keys[i]]
        except:
            continue
    for i in range(0, len(valsH)):
        h['Val_'+str(i)]=valsH[i] 
    h['Val_'+str(n_lines*3)] ='Noise_Median'
    if cont:
        h['Val_'+str(n_lines*3+1)] ='Continum'
        ind=n_lines*3+1
    else:
        ind=n_lines*3
    if skew:
        h['Val_'+str(ind+1)]='Alpha_Narrow'
        h['Val_'+str(ind+2)]='Alpha_Broad' 
    if outflow:
        h['Val_'+str(ind+1)]='Amp_Factor_outflow'
        h['Val_'+str(ind+2)]='Vel_outflow' 
        h['Val_'+str(ind+3)]='FWHM_outflow'
        h['Val_'+str(ind+4)]='Alpha_outflow'  
    if powlaw:
        h['Val_'+str(ind+1)]='Amp_powerlow'
        h['Val_'+str(ind+2)]='Alpha_powerlow'    
    if feii:
        h['Val_'+str(ind+3)]='Sigma_FeII'
        h['Val_'+str(ind+4)]='Delta_FeII'
        h['Val_'+str(ind+5)]='Amp_FeII'
    try:    
        del h['CRVAL3']
        del h['CRPIX3']
        del h['CDELT3']    
    except:
        print('No vals')

    h.update()        
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    hlist.writeto(file_out2+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out2+'.fits')