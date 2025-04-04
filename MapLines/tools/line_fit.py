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

def line_fit_single(file1,file_out,file_out2,name_out2,config_lines='line_prop.yml',input_format='TableFits',z=0.05536,lA1=6450.0,lA2=6850.0,outflow=False,lorentz=False,broad=True,n_line=False,skew=False,error_c=True,ncpu=10,single=False,flux_f=1.0,erft=0.75,dv1t=200,sim=False,cont=False,hbfit=False):
    
    if input_format == 'TableFits':
        hdu_list = fits.open(file1)
        table_hdu = hdu_list[1]
        table_data = table_hdu.data
        pdl_data=table_data.field('FLUX')
        wave=table_data.field('LAMBDA')
        if error_c:
            pdl_dataE=table_data.field('ERROR')
            pdl_dataE=pdl_dataE*flux_f*erft
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
            pdl_dataE=pdl_dataE*flux_f*erft    
    elif input_format == 'IrafFits':
        [pdl_data, hdr]=fits.getdata(file1, 0, header=True)
        if error_c:
            pdl_dataE =fits.getdata(file1, 1, header=False)
            pdl_dataE=pdl_dataE*flux_f*erft
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
            pdl_dataE=pdl_dataE*flux_f*erft
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
            pdl_dataE=pdl_dataE*flux_f*erft
    else:
        print('Error: input_format not recognized')
        print('Options are: TableFits, IrafFits, CSV, ASCII')
        return
 
    
    nz=len(pdl_data)
    pdl_data=pdl_data*flux_f
    
    
    wave_f=wave/(1+z)
    nw=np.where((wave_f >= lA1) & (wave_f <= lA2))[0]
    wave_i=wave_f[nw]
    model_all=np.zeros(len(nw))
    model_Blue=np.zeros(len(nw))
    model_Red=np.zeros(len(nw))
    model_Broad=np.zeros(len(nw))
    if single:
        if cont:
            if skew:
                model_param=np.zeros(12)
            else:
                if outflow:
                    model_param=np.zeros(16)
                else:
                    model_param=np.zeros(10)
        else:
            if skew:
                model_param=np.zeros(11)
            else:
                if outflow:
                    model_param=np.zeros(15)
                else:
                    model_param=np.zeros(9)
    else:
        if cont:
            if skew:
                model_param=np.zeros(17)
            else:
                model_param=np.zeros(15)
        else:
            if skew:
                model_param=np.zeros(16)
            else:
                model_param=np.zeros(14)
    model_param[:]=np.nan    
    data_lines=tol.read_config_file(config_lines)
    if data_lines:
        n_lines=len(data_lines['lines'])
        L1name=data_lines['lines'][0]['name']
        L1wave=data_lines['lines'][0]['wave']
        lfac12=data_lines['lines'][0]['fac']
        L2wave=data_lines['lines'][1]['wave']
        L2name=data_lines['lines'][1]['name']
        LHwave=data_lines['lines'][2]['wave']
        LHname=data_lines['lines'][2]['name']
        LHBwave=data_lines['lines'][3]['wave']
        LHBname=data_lines['lines'][3]['name']
        region=data_lines['continum'][0]['region']
        wavec1=data_lines['continum'][0]['wave1']
        wavec2=data_lines['continum'][0]['wave2']
        waveb1=data_lines['continum'][0]['waveb1']
        waveb2=data_lines['continum'][0]['waveb2']
        valsp=data_lines['priors']
    else:
        print('No configuration line model file')
        return

    for i in range(0, 1):
        for j in range(0, 1):
            val=1
            if val == 1:
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
                #Defining the Broad continum between lines for the initial condition   
                nwt=np.where((wave_f[nw] >= waveb1) & (wave_f[nw] <= waveb2))[0]
                fluxp=np.nanmean(fluxt[nwt])
                fluxe_t=np.nanmean(fluxtE)
                if fluxp < 0:
                    fluxp=0.0001
                data = (fluxt, fluxtE, wave_i, L2wave, LHwave, L1wave, fluxp, dv1t, sim, lfac12, single, skew, broad, lorentz, valsp, n_line, outflow)
                nwalkers=240
                niter=1024
                if single:
                    if skew:
                        initial = np.array([0.04, 0.09, -20.0, 150.0, 1000.0, fluxp, 0.0, 0.0, 0.0])
                    else:
                        if broad:
                            initial = np.array([0.04, 0.09, -20.0, 150.0, 1000.0, fluxp, 0.0])
                        else:
                            if n_line:
                                initial = np.array([0.04, -20.0, 150.0])
                            else:
                                if outflow:
                                    initial = np.array([0.04, 0.09, -20.0, 150.0, 0.2, 0.2, -100.0, 150.0, 0.0])
                                else:
                                    initial = np.array([0.04, 0.09, -20.0, 150.0])
                else:
                    if skew:
                        initial = np.array([0.04, 0.09, 6.0, -80.0, -500.0, 150.0, 1000.0, fluxp, 0.0, 0.0, 0.0])
                    else:
                        if broad:
                            initial = np.array([0.04, 0.09, 6.0, -80.0, -500.0, 150.0, 1000.0, fluxp, 0.0])
                        else:
                            if n_line:
                                initial = np.array([0.04, 6.0, -80.0, -500.0, 150.0])
                            else:
                                initial = np.array([0.04, 0.09, 6.0, -80.0, -500.0, 150.0])
                ndim = len(initial)
                p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
                tim=True
                sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)  
                samples = sampler.flatchain
                theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                if single:
                    if skew:
                        A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f=theta_max
                        model,m2B,mHB,m1B,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                    else:
                        if broad:
                            A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                            model,m2B,mHB,m1B,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz)
                        else:
                            if n_line:
                                A1_f,dv1_f,fwhm1_f=theta_max
                                model,m2B=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line)
                                A3_f=0
                            else:
                                if outflow:
                                    A1_f,A3_f,dv1_f,fwhm1_f,F1o_f,F3o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                                    model,m2B,mHB,m1B,m2Bo,mHBo,m1Bo=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, outflow=outflow)
                                else:
                                    A1_f,A3_f,dv1_f,fwhm1_f=theta_max
                                    model,m2B,mHB,m1B=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                            A7_f=0
                            fwhm2_f=0
                            dv3_f=0
                    model_all[:]=model
                    if n_line:
                        model_Blue[:]=m2B
                    else:
                        model_Blue[:]=m2B+m1B+mHB
                    if broad:
                        model_Broad[:]=mHBR
                    model_param[0]=A1_f
                    model_param[1]=A1_f/lfac12
                    model_param[2]=A3_f
                    model_param[3]=A7_f
                    model_param[4]=dv1_f
                    model_param[5]=dv3_f
                    model_param[6]=fwhm1_f
                    model_param[7]=fwhm2_f
                    model_param[8]=fluxe_t
                    if cont:
                        model_param[9]=fluxpt
                        ind=9
                    else:
                        ind=8
                    if skew:
                        model_param[ind+1]=alph1_f
                        model_param[ind+2]=alphB_f
                    if outflow:
                        model_param[ind+1]=A1_f/flux_f*F1o_f
                        model_param[ind+2]=A1_f/lfac12/flux_f*F1o_f
                        model_param[ind+3]=A3_f/flux_f*F3o_f
                        model_param[ind+4]=dvO_f
                        model_param[ind+5]=fwhmO_f
                        model_param[ind+6]=alphaO_f    
                else:
                    if skew:
                        A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f=theta_max
                    else:
                        if broad:
                            A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                        else:
                            if n_line:
                                A1_f,fac_f,dv1_f,dv2_f,fwhm1_f=theta_max
                            else:
                                A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f=theta_max
                            A7_f=0
                            fwhm2_f=0
                            dv3_f=0
                    if dv2_f < dv1_f:
                        fac_f=1/fac_f
                        dt=np.copy(dv2_f)
                        dv2_f=np.copy(dv1_f)
                        dv1_f=dt
                        A1_f=A1_f*fac_f
                        A3_f=A3_f*fac_f
                    if skew:
                        theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f
                        model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                    else:
                        if broad:
                            theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f
                            model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz)
                        else:
                            if n_line:
                                theta_max=A1_f,fac_f,dv1_f,dv2_f,fwhm1_f 
                                model,m2B,m2R=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line)
                            else:
                                theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f    
                                model,m2B,m2R,mHB,mHR,m1B,m1R=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)                   
                    model_all[:]=model
                    if n_lines:
                        model_Blue[:]=m2B
                        model_Red[:]=m2R
                    else:
                        model_Blue[:]=m2B+m1B+mHB
                        model_Red[:]=m2R+m1R+mHR
                    if broad:
                        model_Broad[:]=mHBR
                    model_param[0]=A1_f
                    model_param[1]=A1_f/lfac12
                    model_param[2]=A3_f
                    model_param[3]=A7_f
                    model_param[4]=fac_f
                    model_param[5]=A1_f/fac_f
                    model_param[6]=A1_f/fac_f/lfac12
                    model_param[7]=A3_f/fac_f
                    model_param[8]=dv1_f
                    model_param[9]=dv2_f
                    model_param[10]=dv3_f
                    model_param[11]=fwhm1_f
                    model_param[12]=fwhm2_f
                    model_param[13]=fluxe_t
                    if cont:
                        model_param[14]=fluxpt
                        ind=14
                    else:
                        ind=13
                    if skew:
                        model_param[ind+1]=alph1_f
                        model_param[ind+2]=alphB_f
                if True:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(7,5))
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
                    ax1.plot(wave_i,fluxtE,linewidth=0.5,color='grey',label=r'$1\sigma$ Error')
                    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
                    if single:
                        if broad:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'$'+LHBname+'$')
                        ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'$'+L2name+'$')
                        if not n_line:
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'$'+LHname+'$')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'$'+L1name+'$')
                        if outflow:
                            ax1.plot(wave_i,m2Bo,linewidth=1,color='orange')
                            ax1.plot(wave_i,mHBo,linewidth=1,color='orange')
                            ax1.plot(wave_i,m1Bo,linewidth=1,color='orange',label=r'outflow')            
                    else:
                        if broad:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'$'+LHBname+'$')
                        ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'$'+L2name+'_b$')
                        ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'$'+L2name+'_r$')
                        if not n_line:
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'$'+LHname+'_b$')
                            ax1.plot(wave_i,mHR,linewidth=1,color='red',label=r'$'+LHname+'_r$')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'$'+L1name+'_b$')
                            ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'$'+L1name+'_r$')
                    fontsize=14
                    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
                    ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
                    ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
                    ax1.legend(fontsize=fontsize)
                    plt.tight_layout()
                    fig.savefig('spectraFit_NAME.pdf'.replace('NAME',name_out2))
                    plt.show()
                    if single:
                        if skew:
                            labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3", "alph1", "alphB"]
                        else:
                            if broad:
                                labels = ['A1','A3','dv1','FWHM_N']
                            else:
                                if n_line:
                                    labels = ['A1','dv1','FWHM_N']
                                else:
                                    labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3"]
                        if hbfit:
                            if skew:
                                labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        labels2 = ['A1','dv1','FWHM_N']
                                    else:
                                        if outflow:
                                            labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$F_{OIII,out}$',r'$F_{H\beta,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$']
                        else:
                            if skew:
                                labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        labels2 = ['A1','dv1','FWHM_N']
                                    else:
                                        if outflow:
                                            labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$F_{NII,out}$',r'$F_{H\alpha,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$']
                    else:
                        if skew:
                            labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3", "alph1", "alphB"]
                        else:
                            if broad:
                                labels = ['A1','A3','fac','dv1','dv2','FWHM']
                            else:
                                if n_line:
                                    labels = ['A1','fac','dv1','dv2','FWHM']
                                else:
                                    labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                        if hbfit:
                            if skew:
                                labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        labels2 = ['A1','fac','dv1','dv2','FWHM']
                                    else:
                                        labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$']
                        else:
                            if skew:
                                labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        labels2 = ['A1','fac','dv1','dv2','FWHM']
                                    else:
                                        labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$']
                    import corner  
                    fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
                    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
                    fig.savefig('corners_NAME.pdf'.replace('NAME',name_out2))
                
                    
                    med_model, spread = mcm.sample_walkers(10, samples, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, single=single, lfac12=lfac12, skew=skew, broad=broad, lorentz=lorentz, n_line=n_line, outflow=outflow)
                    
                    
                    import matplotlib.pyplot as plt
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
                    plt.savefig('spectra_mod_NAME.pdf'.replace('NAME',name_out2))
                    #plt.show()
                if True:  
                    if single:  
                        if skew:
                            print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f,"alph1=",alph1_f,"alphB=",alphB_f)
                        else:
                            if broad:
                                print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                            else:
                                if n_line:
                                    print("A1=",A1_f,"dv1=",dv1_f,"fwhm=",fwhm1_f)
                                else:
                                    if outflow:
                                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"F1o=",F1o_f,"F3o=",F3o_f,"dvO=",dvO_f,"fwhmO=",fwhmO_f,"alph0=",alphaO_f)
                                    else:
                                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f)
                    else:
                        if skew:
                            print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f,"alph1=",alph1_f,"alphB=",alphB_f)
                        else:
                            if broad:    
                                print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                            else:   
                                if n_line:
                                    print("A1=",A1_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f)
                                else: 
                                    print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f)
               
    
    if single:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h4=fits.ImageHDU(model_Broad)
    else:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h3=fits.ImageHDU(model_Red)
        h4=fits.ImageHDU(model_Broad)
    h_k=h1.header
    h_k['EXTNAME'] ='Model'    
    h_k.update()
    if single:
        h_t=h2.header
        h_t['EXTNAME'] ='Narrow_Component'
        h_t.update()  
    else:
        h_t=h2.header
        h_t['EXTNAME'] ='Blue_Component'
        h_t.update()
        h_r=h3.header
        h_r['EXTNAME'] ='Red_Component'
        h_r.update()    
    h_y=h4.header
    h_y['EXTNAME'] ='Broad_Component'
    h_y.update()   
    if single:
        hlist=fits.HDUList([h1,h2,h4])
    else:
        hlist=fits.HDUList([h1,h2,h3,h4])
    hlist.update_extend()
    hlist.writeto(file_out+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out+'.fits')  
    
    h1=fits.PrimaryHDU(model_param)
    h=h1.header
    if single:
        if hbfit:
            h['Val_0'] ='OIII_5007_Amplitude'
            h['Val_1'] ='OIII_4959Amplitude'
            h['Val_2'] ='H_beta_Amplitude'
            h['Val_3'] ='H_beta_Broad_Amplitude'
            h['Val_4'] ='Narrow_vel'
            h['Val_5'] ='Broad_vel'
            h['Val_6'] ='FWHM_Narrow'
            h['Val_7'] ='FWHM_Broad' 
            h['Val_8'] ='Noise_Median'
        else:
            h['Val_0'] ='NII_6585_Amplitude'
            h['Val_1'] ='NII_6549_Amplitude'
            h['Val_2'] ='H_alpha_Amplitude'
            h['Val_3'] ='H_alpha_Broad_Amplitude'
            h['Val_4'] ='Narrow_vel'
            h['Val_5'] ='Broad_vel'
            h['Val_6'] ='FWHM_Narrow'
            h['Val_7'] ='FWHM_Broad' 
            h['Val_8'] ='Noise_Median'
        if cont:
            h['Val_9'] ='Continum'
            ind=9
        else:
            ind=8
        if skew:
            h['Val_'+str(ind+1)]='Alpha_Narrow'
            h['Val_'+str(ind+2)]='Alpha_Broad' 
        if outflow:
            h['Val_'+str(ind+1)]='FirstLineA_Ampl_outflow'
            h['Val_'+str(ind+2)]='FirstLineB_Ampl_outflow' 
            h['Val_'+str(ind+3)]='SecondLine_Ampl_outflow' 
            h['Val_'+str(ind+4)]='Vel_outflow' 
            h['Val_'+str(ind+5)]='FWHM_outflow'     
    else:
        if hbfit:      
            h['Val_0'] ='OIII_5007_Amplitude_blue'
            h['Val_1'] ='OIII_4959_Amplitude_blue'
            h['Val_2'] ='H_beta_Amplitude_blue'
            h['Val_3'] ='H_beta_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_5007_Amplitude_red'
            h['Val_6'] ='NII_4959_Amplitude_red'
            h['Val_7'] ='H_beta_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'
            h['Val_13'] ='Noise_Median'  
        else:  
            h['Val_0'] ='NII_6585_Amplitude_blue'
            h['Val_1'] ='NII_6549_Amplitude_blue'
            h['Val_2'] ='H_alpha_Amplitude_blue'
            h['Val_3'] ='H_alpha_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_6585_Amplitude_red'
            h['Val_6'] ='NII_6549_Amplitude_red'
            h['Val_7'] ='H_alpha_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'  
            h['Val_13'] ='Noise_Median' 
        if cont:
            h['Val_14'] ='Continum'
            ind=14
        else:
            ind=13  
        if skew:
            h['Val_'+str(ind+1)]='Alpha_Narrow'
            h['Val_'+str(ind+2)]='Alpha_Broad'    
    

    h.update()        
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    hlist.writeto(file_out2+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out2+'.fits')


def line_fit(file1,file2,file3,file_out,file_out2,name_out2,z=0.05536,j_t=0,i_t=0,config_lines='line_prop.yml',lA1=6450.0,lA2=6850.0,outflow=False,lorentz=False,broad=True,n_line=False,skew=False,error_c=True,test=False,plot_f=True,ncpu=10,pgr_bar=True,single=False,flux_f=1.0,erft=0.75,dv1t=200,sim=False,cont=False,hbfit=False):
    try:
        [pdl_cube, hdr]=fits.getdata(file1, 'FLUX', header=True)
    except:
        [pdl_cube, hdr]=fits.getdata(file1, 0, header=True)
    if error_c:
        try:
            pdl_cubeE =fits.getdata(file1, 'ERROR', header=False)
        except:
            try:
                pdl_cubeE =fits.getdata(file1, 'IVAR', header=False)
                pdl_cubeE=1.0/np.sqrt(pdl_cubeE)
            except:
                pdl_cubeE =fits.getdata(file1, 1, header=False)    
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
    model_Blue=np.zeros([len(nw),nx,ny])
    model_Red=np.zeros([len(nw),nx,ny])
    model_Broad=np.zeros([len(nw),nx,ny])
    model_Inp=np.zeros([len(nw),nx,ny])
    model_InpE=np.zeros([len(nw),nx,ny])
    if outflow:
        model_Outflow=np.zeros([len(nw),nx,ny])
    if single:
        if cont:
            if skew:
                model_param=np.zeros([12,nx,ny])
            else:
                if outflow:
                    model_param=np.zeros([16,nx,ny])
                else:
                    model_param=np.zeros([10,nx,ny])
        else:
            if skew:
                model_param=np.zeros([11,nx,ny])
            else:
                if outflow:
                    model_param=np.zeros([15,nx,ny])
                else:
                    model_param=np.zeros([9,nx,ny])
    else:
        if cont:
            if skew:
                model_param=np.zeros([17,nx,ny])
            else:
                if outflow:
                    model_param=np.zeros([21,nx,ny])
                else:
                    model_param=np.zeros([15,nx,ny])
        else:
            if skew:
                model_param=np.zeros([16,nx,ny])
            else:
                if outflow:
                    model_param=np.zeros([20,nx,ny])
                else:
                    model_param=np.zeros([14,nx,ny])
    model_param[:,:,:]=np.nan    

    data_lines=tol.read_config_file(config_lines)
    if data_lines:

        n_lines=len(data_lines['lines'])
        L1name=data_lines['lines'][0]['name']
        L1wave=data_lines['lines'][0]['wave']
        lfac12=data_lines['lines'][0]['fac']
        L2wave=data_lines['lines'][1]['wave']
        L2name=data_lines['lines'][1]['name']
        LHwave=data_lines['lines'][2]['wave']
        LHname=data_lines['lines'][2]['name']
        LHBwave=data_lines['lines'][3]['wave']
        LHBname=data_lines['lines'][3]['name']
        region=data_lines['continum'][0]['region']
        wavec1=data_lines['continum'][0]['wave1']
        wavec2=data_lines['continum'][0]['wave2']
        waveb1=data_lines['continum'][0]['waveb1']
        waveb2=data_lines['continum'][0]['waveb2']
        valsp=data_lines['priors']
    else:
        print('No configuration line model file')
        return

    #Loiii1=L1wave#4960.36 
    #LnrHb=LHwave#4862.68 
    #Loiii2=L2wave#5008.22
    #Lnii2=L2wave#6585.278
    #LnrHa=LHwave#6564.632
    #Lnii1=L1wave#6549.859
    #if hbfit:
    #    lfac12=lfac12#3.0
    #    L1wave=Loiii1
    #    L2wave=Loiii2
    #    LHwave=LnrHb
    #    LHBwave=LnrHb
    #else:
    #    lfac12=lfac12#2.93
    #    L1wave=Lnii1
    #    L2wave=Lnii2
    #    LHwave=LnrHa
    #    LHBwave=LnrHa

    hdr["CRVAL3"]=wave_i[0]
    try:
        hdr["CD3_3"]=cdelt
    except:
        hdr["CDELT3"]=cdelt/(1+z)
    if pgr_bar:
        pbar=tqdm(total=nx*ny)
    for i in range(0, nx):
        for j in range(0, ny):
            val=mask[i,j]
            #val=1
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
                #Defining the Broad continum between lines for the initial condition
                nwt=np.where((wave_f[nw] >= waveb1) & (wave_f[nw] <= waveb2))[0]
                fluxp=np.nanmean(fluxt[nwt])
                fluxe_t=np.nanmean(fluxtE)
                if fluxp < 0:
                    fluxp=0.0001
                data = (fluxt, fluxtE, wave_i, L2wave, LHwave, L1wave, fluxp, dv1t, sim, lfac12, single, skew, broad, lorentz, valsp, n_line, outflow)
                nwalkers=240
                niter=1024
                if single:
                    if skew:  
                        initial = np.array([valsp['a1o'], valsp['a3o'], valsp['dv1o'], valsp['fwhm1o'], valsp['fwhm2o'], fluxp, valsp['dv3o'], 0.0, 0.0])
                    else:
                        if broad:
                            initial = np.array([valsp['a1o'], valsp['a3o'], valsp['dv1o'], valsp['fwhm1o'], valsp['fwhm2o'], fluxp, valsp['dv3o']])
                        else:
                            if n_line:
                                if outflow:
                                    initial = np.array([valsp['a1o'], valsp['dv1o'], valsp['fwhm1o'], valsp['f1o'], valsp['dvOo'], valsp['fwhmOo'], valsp['alpOo']])
                                else:
                                    initial = np.array([valsp['a1o'], valsp['dv1o'], valsp['fwhm1o']])
                            else:
                                if outflow:
                                    initial = np.array([valsp['a1o'], valsp['a3o'], valsp['dv1o'], valsp['fwhm1o'], valsp['f1o'], valsp['f3o'], valsp['dvOo'], valsp['fwhmOo'], valsp['alpOo']])
                                else:
                                    initial = np.array([valsp['a1o'], valsp['a3o'], valsp['dv1o'], valsp['fwhm1o']])
                else:
                    if skew:
                        initial = np.array([valsp['a1o'], valsp['a3o'], valsp['fac12o'], valsp['dv1o'], valsp['dv2o'], valsp['fwhm1o'], valsp['fwhm2o'], fluxp, valsp['dv3o'], 0.0, 0.0])
                    else:
                        if broad:
                            initial = np.array([valsp['a1o'], valsp['a3o'], valsp['fac12o'], valsp['dv1o'], valsp['dv2o'], valsp['fwhm1o'], valsp['fwhm2o'], fluxp, valsp['dv3o']])
                        else:
                            if n_line:
                                if outflow:
                                    initial = np.array([valsp['a1o'], valsp['fac12o'], valsp['dv1o'], valsp['dv2o'], valsp['fwhm1o'], valsp['f1o'], valsp['dvOo'], valsp['fwhmOo'], valsp['alpOo']])
                                else:
                                    initial = np.array([valsp['a1o'], valsp['fac12o'], valsp['dv1o'], valsp['dv2o'], valsp['fwhm1o']])
                            else:
                                initial = np.array([valsp['a1o'], valsp['a3o'], valsp['fac12o'], valsp['dv1o'], valsp['dv2o'], valsp['fwhm1o']])
                ndim = len(initial)
                p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
                if plot_f:
                    tim=True
                else:
                    tim=False
                sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)  
                samples = sampler.flatchain
                theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                if single:
                    if skew:
                        A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f=theta_max
                        model,m2B,mHB,m1B,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                    else:
                        if broad:
                            A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                            model,m2B,mHB,m1B,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz)
                        else:
                            if n_line:
                                if outflow:
                                    A1_f,dv1_f,fwhm1_f,F1o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                                    model,m2B,m2Bo=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line, outflow=outflow)
                                    F3o_f=0
                                else:
                                    A1_f,dv1_f,fwhm1_f=theta_max
                                    model,m2B=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line)
                                A3_f=0
                            else:
                                if outflow:
                                    A1_f,A3_f,dv1_f,fwhm1_f,F1o_f,F3o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                                    model,m2B,mHB,m1B,m2Bo,mHBo,m1Bo=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, outflow=outflow)
                                else:
                                    A1_f,A3_f,dv1_f,fwhm1_f=theta_max
                                    model,m2B,mHB,m1B=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                            A7_f=0
                            fwhm2_f=0
                            dv3_f=0
                    model_all[:,i,j]=model
                    model_Inp[:,i,j]=fluxt
                    model_InpE[:,i,j]=fluxtE
                    if n_line:
                        model_Blue[:,i,j]=m2B
                        if outflow:
                            model_Outflow[:,i,j]=m2Bo
                    else:
                        model_Blue[:,i,j]=m2B+m1B+mHB
                        if outflow:
                            model_Outflow[:,i,j]=m2Bo+m1Bo+mHBo
                    if broad:
                        model_Broad[:,i,j]=mHBR
                    model_param[0,i,j]=A1_f/flux_f
                    model_param[1,i,j]=A1_f/lfac12/flux_f
                    model_param[2,i,j]=A3_f/flux_f
                    model_param[3,i,j]=A7_f/flux_f
                    model_param[4,i,j]=dv1_f
                    model_param[5,i,j]=dv3_f
                    model_param[6,i,j]=fwhm1_f
                    model_param[7,i,j]=fwhm2_f
                    model_param[8,i,j]=fluxe_t/flux_f
                    if cont:
                        model_param[9,i,j]=fluxpt/flux_f
                        ind=9
                    else:
                        ind=8
                    if skew:
                        model_param[ind+1,i,j]=alph1_f
                        model_param[ind+2,i,j]=alphB_f
                    if outflow:
                        model_param[ind+1,i,j]=A1_f/flux_f*F1o_f
                        model_param[ind+2,i,j]=A1_f/lfac12/flux_f*F1o_f
                        model_param[ind+3,i,j]=A3_f/flux_f*F3o_f
                        model_param[ind+4,i,j]=dvO_f
                        model_param[ind+5,i,j]=fwhmO_f
                        model_param[ind+6,i,j]=alphaO_f
                else:
                    if skew:
                        A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f=theta_max
                    else:
                        if broad:
                            A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                        else:
                            if n_line:
                                if outflow:
                                    A1_f,fac_f,dv1_f,dv2_f,fwhm1_f,F1o_f,dvO_f,fwhmO_f,alphaO_f=theta_max
                                    F3o_f=0
                                else:
                                    A1_f,fac_f,dv1_f,dv2_f,fwhm1_f=theta_max
                                A3_f=0
                            else:
                                A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f=theta_max
                            A7_f=0
                            fwhm2_f=0
                            dv3_f=0
                    if dv2_f < dv1_f:
                        fac_f=1/fac_f
                        dt=np.copy(dv2_f)
                        dv2_f=np.copy(dv1_f)
                        dv1_f=dt
                        A1_f=A1_f*fac_f
                        A3_f=A3_f*fac_f
                    if skew:
                        theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f,alph1_f,alphB_f
                        model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)
                    else:
                        if broad:
                            theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f
                            model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz)
                        else:
                            if n_line:
                                if outflow:
                                    theta_max=A1_f,fac_f,dv1_f,dv2_f,fwhm1_f,F1o_f,dvO_f,fwhmO_f,alphaO_f
                                    model,m2B,m2R,m2Bo,m2Ro=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line, outflow=outflow)
                                else:
                                    theta_max=A1_f,fac_f,dv1_f,dv2_f,fwhm1_f    
                                    model,m2B,m2R=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad, n_line=n_line)
                            else:
                                theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f    
                                model,m2B,m2R,mHB,mHR,m1B,m1R=mod.line_model(theta_max, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, ret_com=True, lfac12=lfac12, single=single, skew=skew, broad=broad)                   
                    model_all[:,i,j]=model
                    if n_line:
                        model_Blue[:,i,j]=m2B
                        model_Red[:,i,j]=m2R
                        if outflow:
                            model_Outflow[:,i,j]=m2Bo+m2Ro
                    else:
                        model_Blue[:,i,j]=m2B+m1B+mHB
                        model_Red[:,i,j]=m2R+m1R+mHR
                    if broad:
                        model_Broad[:,i,j]=mHBR    
                    model_param[0,i,j]=A1_f/flux_f
                    model_param[1,i,j]=A1_f/lfac12/flux_f
                    model_param[2,i,j]=A3_f/flux_f
                    model_param[3,i,j]=A7_f/flux_f
                    model_param[4,i,j]=fac_f
                    model_param[5,i,j]=A1_f/fac_f/flux_f
                    model_param[6,i,j]=A1_f/fac_f/lfac12/flux_f
                    model_param[7,i,j]=A3_f/fac_f/flux_f
                    model_param[8,i,j]=dv1_f
                    model_param[9,i,j]=dv2_f
                    model_param[10,i,j]=dv3_f
                    model_param[11,i,j]=fwhm1_f
                    model_param[12,i,j]=fwhm2_f
                    model_param[13,i,j]=fluxe_t/flux_f
                    if cont:
                        model_param[14,i,j]=fluxpt/flux_f
                        ind=14
                    else:
                        ind=13
                    if skew:
                        model_param[ind+1,i,j]=alph1_f
                        model_param[ind+2,i,j]=alphB_f
                    if outflow:
                        model_param[ind+1,i,j]=A1_f/flux_f*F1o_f
                        model_param[ind+2,i,j]=A1_f/lfac12/flux_f*F1o_f
                        model_param[ind+3,i,j]=A3_f/flux_f*F3o_f
                        model_param[ind+4,i,j]=dvO_f
                        model_param[ind+5,i,j]=fwhmO_f
                        model_param[ind+6,i,j]=alphaO_f    
                if plot_f:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(7,5))
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
                    ax1.plot(wave_i,fluxtE,linewidth=1,color='grey',label=r'$1\sigma$ Error')
                    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
                    ax1.plot(wave_i,fluxt-model-np.nanmax(fluxt)*0.25,linewidth=1,color='olive',label=r'Residual')
                    if single:
                        if hbfit:
                            if broad:
                                ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'OIII_2_NR')
                            if not n_line:
                               ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Hb_n_NR')
                               ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'OIII_1_NR')
                        else:
                            if broad:
                                ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_NR')
                            if not n_line:
                                ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Ha_n_NR')
                                ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_NR')
                        if outflow:
                            ax1.plot(wave_i,m2Bo,linewidth=1,color='orange')#,label=r'OIII_2_OR')
                            if not n_line:
                                ax1.plot(wave_i,mHBo,linewidth=1,color='orange')#,label=r'Hb_n_OR')
                                ax1.plot(wave_i,m1Bo,linewidth=1,color='orange',label=r'outflow')        
                    else:
                        if hbfit:
                            if broad:
                                ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'OIII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'OIII_2_r')
                            if not n_line:
                                ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Hb_n_b')
                                ax1.plot(wave_i,mHR,linewidth=1,color='red',label=r'Hb_n_r')
                                ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'OIII_1_b')
                                ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'OIII_1_r')
                        else:
                            if broad:
                                ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Ha_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'NII_2_r')
                            if not n_line:
                                ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Ha_n_b')
                                ax1.plot(wave_i,mHR,linewidth=1,color='red',label=r'Ha_n_r')
                                ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_b')
                                ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'NII_1_r')
                        if outflow:
                            ax1.plot(wave_i,m2Bo,linewidth=1,color='orange')
                            ax1.plot(wave_i,m2Ro,linewidth=1,color='orange',label=r'outflow')    
                    fontsize=14
                    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
                    ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
                    ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
                    ax1.legend(fontsize=fontsize)
                    plt.tight_layout()
                    fig.savefig('spectraFit_NAME.pdf'.replace('NAME',name_out2))
                    plt.show()
                    if single:
                        if skew:
                            labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3", "alph1", "alphB"]
                        else:
                            if broad:
                                labels = ['A1','A3','dv1','FWHM_N']
                            else:
                                if n_line:
                                    labels = ['A1','dv1','FWHM_N',"FWHM_B"]
                                else:
                                    labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3"]
                        if hbfit:
                            if skew:
                                labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        if outflow:
                                            labels2 = ['A1','dv1','FWHM_N',"FWHM_B",r'$F_{OIII,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = ['A1','dv1','FWHM_N',"FWHM_B"]
                                    else:
                                        if outflow:
                                            labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$F_{OIII,out}$',r'$F_{H\beta,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$']
                        else:
                            if skew:
                                labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        if outflow:
                                            labels2 = ['A1','dv1','FWHM_N',"FWHM_B",r'$F_{NII,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = ['A1','dv1','FWHM_N',"FWHM_B"]
                                    else:
                                        if outflow:
                                            labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$F_{NII,out}$',r'$F_{H\alpha,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$']
                    else:
                        if skew:
                            labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3", "alph1", "alphB"]
                        else:
                            if broad:
                                labels = ['A1','A3','fac','dv1','dv2','FWHM']
                            else:
                                if n_line:
                                    labels = ['A1','fac','dv1','dv2','FWHM',"FWHM_B"]
                                else:
                                    labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                        if hbfit:
                            if skew:
                                labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        labels2 = ['A1','fac','dv1','dv2','FWHM',"FWHM_B"]
                                    else:
                                        labels2 = [r'$A_{OIII,b}$',r'$A_{H\beta,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$']
                        else:
                            if skew:
                                labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$',r'$\alpha_n$',r'$\alpha_b$']
                            else:
                                if broad:
                                    labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']
                                else:
                                    if n_line:
                                        if outflow:
                                            labels2 = [r'$A_1$',r'$f_c$',r'$\Delta v_{1b}$',r'$\Delta v_{1r}$',r'$FWHM_n$',r'$F_{1,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = ['A1','fac','dv1','dv2','FWHM',"FWHM_B"]
                                    else:
                                        if outflow:
                                            labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$',r'$F_{NII,out}$',r'$F_{H\alpha,out}$',r'$\Delta v_{out}$',r'$FWHM_{out}$',r'$alpha_{out}$']
                                        else:
                                            labels2 = [r'$A_{NII,b}$',r'$A_{H\alpha,b}$',r'$f_c$',r'$\Delta v_b$',r'$\Delta v_r$',r'$FWHM_n$']
                    import corner  
                    fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
                    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
                    fig.savefig('corners_NAME.pdf'.replace('NAME',name_out2))
                
                    
                    med_model, spread = mcm.sample_walkers(10, samples, x=wave_i, xo1=L2wave, xo2=LHwave, xo3=L1wave, single=single, lfac12=lfac12, skew=skew, broad=broad, lorentz=lorentz, n_line=n_line, outflow=outflow)
                    
                    
                    import matplotlib.pyplot as plt
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
                    plt.savefig('spectra_mod_NAME.pdf'.replace('NAME',name_out2))
                if pgr_bar == False:  
                    if single:  
                        if skew:
                            print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f,"alph1=",alph1_f,"alphB=",alphB_f)
                        else:
                            if broad:
                                print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                            else:
                                if n_line:
                                    print("A1=",A1_f,"dv1=",dv1_f,"fwhm=",fwhm1_f)
                                else:
                                    if outflow:
                                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"F1o=",F1o_f,"F3o=",F3o_f,"dvO=",dvO_f,"fwhmO=",fwhmO_f,"alph0=",alphaO_f)
                                    else:
                                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f)
                    else:
                        if skew:
                            print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f,"alph1=",alph1_f,"alphB=",alphB_f)
                        else:
                            if broad:    
                                print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                            else:   
                                if n_line:
                                    if outflow:
                                        print("A1=",A1_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"F1o=",F1o_f,"dvO=",dvO_f,"fwhmO=",fwhmO_f,"alph0=",alphaO_f)
                                    else:
                                        print("A1=",A1_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f)
                                else: 
                                    if outflow:
                                        print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"F1o=",F1o_f,"F3o=",F3o_f,"dvO=",dvO_f,"fwhmO=",fwhmO_f,"alph0=",alphaO_f)
                                    else:
                                        print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f)
                if test:        
                    #sys.exit()
                    return        
            if pgr_bar:
                pbar.update(1)
    
    if single:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h4=fits.ImageHDU(model_Broad)
        h5=fits.ImageHDU(model_Inp)
        h6=fits.ImageHDU(model_InpE)
        if outflow:
            h7=fits.ImageHDU(model_Outflow)
    else:
        h1=fits.PrimaryHDU(model_all)
        h2=fits.ImageHDU(model_Blue)
        h3=fits.ImageHDU(model_Red)
        h4=fits.ImageHDU(model_Broad)
        h5=fits.ImageHDU(model_Inp)
        h6=fits.ImageHDU(model_InpE)
        if outflow:
            h7=fits.ImageHDU(model_Outflow)
    h_k=h1.header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        h_k[keys[i]]=hdr[keys[i]]
        h_k.comments[keys[i]]=hdr.comments[keys[i]]
    h_k['EXTNAME'] ='Model'    
    h_k.update()
    if single:
        h_t=h2.header
        for i in range(0, len(keys)):
            h_t[keys[i]]=hdr[keys[i]]
            h_t.comments[keys[i]]=hdr.comments[keys[i]]
        h_t['EXTNAME'] ='Narrow_Component'
        h_t.update()  
    else:
        h_t=h2.header
        for i in range(0, len(keys)):
            h_t[keys[i]]=hdr[keys[i]]
            h_t.comments[keys[i]]=hdr.comments[keys[i]]
        h_t['EXTNAME'] ='Blue_Component'
        h_t.update()
        h_r=h3.header
        for i in range(0, len(keys)):
            h_r[keys[i]]=hdr[keys[i]]
            h_r.comments[keys[i]]=hdr.comments[keys[i]]
        h_r['EXTNAME'] ='Red_Component'
        h_r.update()    
    h_y=h4.header
    for i in range(0, len(keys)):
        h_y[keys[i]]=hdr[keys[i]]
        h_y.comments[keys[i]]=hdr.comments[keys[i]]
    h_y['EXTNAME'] ='Broad_Component'
    h_y.update()

    h_y=h5.header
    for i in range(0, len(keys)):
        h_y[keys[i]]=hdr[keys[i]]
        h_y.comments[keys[i]]=hdr.comments[keys[i]]
    h_y['EXTNAME'] ='Input_Component'
    h_y.update()   
    
    h_y=h6.header
    for i in range(0, len(keys)):
        h_y[keys[i]]=hdr[keys[i]]
        h_y.comments[keys[i]]=hdr.comments[keys[i]]
    h_y['EXTNAME'] ='InputE_Component'
    h_y.update()  
    if outflow:
        h_y=h7.header
        for i in range(0, len(keys)):
            h_y[keys[i]]=hdr[keys[i]]
            h_y.comments[keys[i]]=hdr.comments[keys[i]]
        h_y['EXTNAME'] ='Outflow_Component'
        h_y.update()  
    if single:
        if outflow:
            hlist=fits.HDUList([h1,h2,h4,h5,h6,h7])
        else:
            hlist=fits.HDUList([h1,h2,h4,h5,h6])
    else:
        hlist=fits.HDUList([h1,h2,h3,h4,h5,h6])
    hlist.update_extend()
    hlist.writeto(file_out+'.fits', overwrite=True)
    tol.sycall('gzip -f '+file_out+'.fits')
    
    h1=fits.PrimaryHDU(model_param)
    h=h1.header
    keys=list(hdr.keys())
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h[keys[i]]=hdr[keys[i]]
            h.comments[keys[i]]=hdr.comments[keys[i]]
    if single:
        if hbfit:
            h['Val_0'] ='OIII_5007_Amplitude'
            h['Val_1'] ='OIII_4959Amplitude'
            h['Val_2'] ='H_beta_Amplitude'
            h['Val_3'] ='H_beta_Broad_Amplitude'
            h['Val_4'] ='Narrow_vel'
            h['Val_5'] ='Broad_vel'
            h['Val_6'] ='FWHM_Narrow'
            h['Val_7'] ='FWHM_Broad' 
            h['Val_8'] ='Noise_Median'
        else:
            h['Val_0'] ='NII_6585_Amplitude'
            h['Val_1'] ='NII_6549_Amplitude'
            h['Val_2'] ='H_alpha_Amplitude'
            h['Val_3'] ='H_alpha_Broad_Amplitude'
            h['Val_4'] ='Narrow_vel'
            h['Val_5'] ='Broad_vel'
            h['Val_6'] ='FWHM_Narrow'
            h['Val_7'] ='FWHM_Broad' 
            h['Val_8'] ='Noise_Median'
        if cont:
            h['Val_9'] ='Continum'
            ind=9
        else:
            ind=8
        if skew:
            h['Val_'+str(ind+1)]='Alpha_Narrow'
            h['Val_'+str(ind+2)]='Alpha_Broad' 
        if outflow:
            h['Val_'+str(ind+1)]='FirstLineA_Ampl_outflow'
            h['Val_'+str(ind+2)]='FirstLineB_Ampl_outflow' 
            h['Val_'+str(ind+3)]='SecondLine_Ampl_outflow' 
            h['Val_'+str(ind+4)]='Vel_outflow' 
            h['Val_'+str(ind+5)]='FWHM_outflow' 
    else:
        if hbfit:      
            h['Val_0'] ='OIII_5007_Amplitude_blue'
            h['Val_1'] ='OIII_4959_Amplitude_blue'
            h['Val_2'] ='H_beta_Amplitude_blue'
            h['Val_3'] ='H_beta_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_5007_Amplitude_red'
            h['Val_6'] ='NII_4959_Amplitude_red'
            h['Val_7'] ='H_beta_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'
            h['Val_13'] ='Noise_Median'  
        else:  
            h['Val_0'] ='NII_6585_Amplitude_blue'
            h['Val_1'] ='NII_6549_Amplitude_blue'
            h['Val_2'] ='H_alpha_Amplitude_blue'
            h['Val_3'] ='H_alpha_Broad_Amplitude'
            h['Val_4'] ='Blue_Red_Factor'
            h['Val_5'] ='NII_6585_Amplitude_red'
            h['Val_6'] ='NII_6549_Amplitude_red'
            h['Val_7'] ='H_alpha_Amplitude_red'
            h['Val_8'] ='Blue_vel'
            h['Val_9'] ='Red_vel'
            h['Val_10'] ='Broad_vel'
            h['Val_11'] ='FWHM_Narrow'
            h['Val_12'] ='FWHM_Broad'  
            h['Val_13'] ='Noise_Median' 
        if cont:
            h['Val_14'] ='Continum'
            ind=14
        else:
            ind=13  
        if skew:
            h['Val_'+str(ind+1)]='Alpha_Narrow'
            h['Val_'+str(ind+2)]='Alpha_Broad'    
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