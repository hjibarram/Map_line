#!/usr/bin/env python
import numpy as np
import MapLines.tools.models as mod
import MapLines.tools.mcmc as mcm
import MapLines.tools.tools as tol
import MapLines.tools.priors as pri
from astropy.io import fits
#from progressbar import ProgressBar
from astropy.io import fits
import os.path as ptt
import sys
from tqdm import tqdm
import corner 
import matplotlib.pyplot as plt

def line_fit_single(file1,file_out,file_out2,name_out2,dir_out='',smoth=False,ker=2,config_lines='line_prop.yml',labplot=True,input_format='TableFits',z=0.05536,lA1=6450.0,lA2=6850.0,verbose=True,outflow=False,voigt=False,lorentz=False,skew=False,error_c=True,ncpu=10,flux_f=1.0,erft=0.75,cont=False):
    
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
                wave=table_data.field('wave')
        if error_c:
            try:
                pdl_dataE=table_data.field('ERROR')
            except:
                pdl_dataE=table_data.field('fluxE')
            if erft != 0:
                pdl_dataE=pdl_dataE*flux_f*erft
            else:
                pdl_dataE=pdl_dataE*flux_f
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
        wave=crval+cdelt*(np.arange(nz)+1-crpix)  
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
        return
 
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
       
    data_lines=tol.read_config_file(config_lines)
    if data_lines:
        n_lines=len(data_lines['lines'])
        pac=['AoN','dvoN','fwhmoN']
        pacL=[r'$A_{N}$',r'$\Delta v_{N}$',r'$FWHM_{N}$']
        pacH=['N_Amplitude','N_Velocity','N_FWHM']
        waves0=[]
        names0=[]
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
                return
            try:
                Infvalues.extend([valsp[valt.replace('o','i')]])
            except:
                print('The keyword '+valt.replace('o','i')+' is missing in the line config file')
                return
            try:
                Supvalues.extend([valsp[valt.replace('o','s')]])
            except:
                print('The keyword '+valt.replace('o','s')+' is missing in the line config file')
                return
    else:
        print('No configuration line model file')
        return
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
            colors=['blue','red','purple','brown','pink']
            fig = plt.figure(figsize=(7,5))
            ax1 = fig.add_subplot(1,1,1)
            ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
            ax1.plot(wave_i,fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
            ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
            ax1.plot(wave_i,fluxt-model-np.nanmax(fluxt)*0.25,linewidth=1,color='olive',label=r'Residual')    
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
            fontsize=14
            ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
            ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
            ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
            ax1.legend(fontsize=fontsize)
            plt.tight_layout()
            fig.savefig(dir_out+'spectraFit_NAME.pdf'.replace('NAME',name_out2))
            plt.show()

            if skew:
                labels2 = [*valsL,r'$\alpha_n$',r'$\alpha_b$']
            else:
                if outflow:
                    labels2 = [*valsL,r'$F_{out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$\alpha_{out}$']
                else:
                    labels2 = valsL
            fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
            fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
            fig.savefig(dir_out+'corners_NAME.pdf'.replace('NAME',name_out2))
                                   
            med_model, spread = mcm.sample_walkers(10, samples, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, skew=skew, lorentz=lorentz, outflow=outflow)
                       
            
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
            plt.savefig(dir_out+'spectra_mod_NAME.pdf'.replace('NAME',name_out2))
                
            if verbose:    
                #print Best fit parameters
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


def line_fit(file1,file2,file3,file_out,file_out2,name_out2,dir_out='',colors=['blue','red','purple','brown','pink'],z=0.05536,j_t=0,i_t=0,labplot=True,config_lines='line_prop.yml',lA1=6450.0,lA2=6850.0,outflow=False,voigt=False,lorentz=False,skew=False,error_c=True,test=False,plot_f=True,ncpu=10,pgr_bar=True,flux_f=1.0,erft=0,cont=False):
    try:
        [pdl_cube, hdr]=fits.getdata(file1, 'FLUX', header=True)
    except:
        try:
            [pdl_cube, hdr]=fits.getdata(file1, 'SCI', header=True)
        except:
            [pdl_cube, hdr]=fits.getdata(file1, 0, header=True)
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
                pdl_cubeE =fits.getdata(file1, 1, header=False)    
        if erft != 0:
            pdl_cubeE=pdl_cubeE*flux_f*erft

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
    wave_f=wave/(1+z)
    nw=np.where((wave_f >= lA1) & (wave_f <= lA2))[0]
    wave_i=wave_f[nw]
    model_all=np.zeros([len(nw),nx,ny])
    model_Inp=np.zeros([len(nw),nx,ny])
    model_InpE=np.zeros([len(nw),nx,ny])
    if outflow:
        model_Outflow=np.zeros([len(nw),nx,ny])

    data_lines=tol.read_config_file(config_lines)
    if data_lines:
        n_lines=len(data_lines['lines'])
        pac=['AoN','dvoN','fwhmoN']
        pacL=[r'$A_{N}$',r'$\Delta v_{N}$',r'$FWHM_{N}$']
        pacH=['N_Amplitude','N_Velocity','N_FWHM']
        waves0=[]
        names0=[]
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
                return
            try:
                Infvalues.extend([valsp[valt.replace('o','i')]])
            except:
                print('The keyword '+valt.replace('o','i')+' is missing in the line config file')
                return
            try:
                Supvalues.extend([valsp[valt.replace('o','s')]])
            except:
                print('The keyword '+valt.replace('o','s')+' is missing in the line config file')
                return
    else:
        print('No configuration line model file')
        return
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
    model_param[:,:,:]=np.nan    

    hdr["CRVAL3"]=wave_i[0]
    try:
        hdr["CD3_3"]=cdelt/(1+z)
    except:
        hdr["CDELT3"]=cdelt/(1+z)
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
                        initial = np.array([*Inpvalues])
                
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
                

                if plot_f:
                    fig = plt.figure(figsize=(7,5))
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
                    ax1.plot(wave_i,fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
                    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
                    ax1.plot(wave_i,fluxt-model-np.nanmax(fluxt)*0.25,linewidth=1,color='olive',label=r'Residual')                  
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
                    fontsize=14
                    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
                    ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
                    ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
                    if labplot:
                        ax1.legend(fontsize=fontsize)
                    plt.tight_layout()
                    fig.savefig(dir_out+'spectraFit_NAME.pdf'.replace('NAME',name_out2))
                    plt.show()


                    if skew:
                        labels2 = [*valsL,r'$\alpha_n$',r'$\alpha_b$']
                    else:
                        if outflow:
                            labels2 = [*valsL,r'$F_{out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$\alpha_{out}$']
                        else:
                            labels2 = valsL
                               
                    fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
                    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
                    fig.savefig(dir_out+'corners_NAME.pdf'.replace('NAME',name_out2))
                
                    
                    med_model, spread = mcm.sample_walkers(10, samples, waves0, fac0, facN0, velfac0, velfacN0, fwhfac0, fwhfacN0, names0, n_lines, vals, x=wave_i, skew=skew, lorentz=lorentz, outflow=outflow)
                    
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
                    plt.savefig(dir_out+'spectra_mod_NAME.pdf'.replace('NAME',name_out2))
                
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