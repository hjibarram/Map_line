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

def line_fit(file1,file2,file3,file_out,file_out2,z=0.05536,j_t=0,i_t=0,lA1=6450.0,lA2=6850.0,error_c=True,test=False,plot_f=True,ncpu=10,pgr_bar=True,single=False,flux_f=1.0,erft=0.75,dv1t=200,sim=False,cont=False,hbfit=False):
    [pdl_cube, hdr]=fits.getdata(file1, 0, header=True)
    if error_c:
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
    if single:
        if cont:
            model_param=np.zeros([10,nx,ny])
        else:
            model_param=np.zeros([9,nx,ny])
    else:
        if cont:
            model_param=np.zeros([15,nx,ny])
        else:
            model_param=np.zeros([14,nx,ny])
    model_param[:,:,:]=np.nan    
    Loiii1=4960.36 
    LnrHb=4862.68 
    Loiii2=5008.22
    Lnii2=6585.278
    LnrHa=6564.632
    Lnii1=6549.859
    if hbfit:
        lfac12=3.0
    else:
        lfac12=2.93
    hdr["CRVAL3"]=wave_i[0]
    try:
        hdr["CD3_3"]=cdelt
    except:
        hdr["CDELT3"]=cdelt/(1+z)
    if pgr_bar:
        #progress2 = ProgressBar(maxval=nx).start()
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
                    if hbfit:
                        nwt=np.where((wave_f[nw] >= 5035.0) & (wave_f[nw] <= 5055.0))[0]
                    else:
                        nwt=np.where((wave_f[nw] >= 6380.0) & (wave_f[nw] <= 6400.0))[0]  
                    fluxpt=np.nanmean(fluxt[nwt])  
                    fluxt=fluxt-fluxpt
                if hbfit:
                	nwt=np.where((wave_f[nw] >= 4880.0) & (wave_f[nw] <= 4890.0))[0]
                else:    
                    nwt=np.where((wave_f[nw] >= 6569.0) & (wave_f[nw] <= 6572.0))[0]
                fluxp=np.nanmean(fluxt[nwt])
                fluxe_t=np.nanmean(fluxtE)
                if fluxp < 0:
                    fluxp=0.0001
                #if single:
                if hbfit:
                    data = (fluxt, fluxtE, wave_i, Loiii2, LnrHb, Loiii1, fluxp, dv1t, sim, lfac12)
                else:
                    data = (fluxt, fluxtE, wave_i, Lnii2, LnrHa, Lnii1, fluxp, dv1t, sim, lfac12)
                #else:
                #    if hbfit:
                #        data = (fluxt, fluxtE, wave_i, Loiii2, LnrHb, Loiii1, fluxp, dv1t, sim, lfac12)
                #    else:
                #        data = (fluxt, fluxtE, wave_i, Lnii2, LnrHa, Lnii1, fluxp, dv1t, sim, lfac12)
                nwalkers=240
                niter=1024
                if single:
                    if hbfit:
                        initial = np.array([0.04, 0.09, -20.0, 150.0, 1000.0, fluxp, 0.0])
                    else:
                        initial = np.array([0.04, 0.06, -20.0, 150.0, 1000.0, fluxp, 0.0])
                else:
                    if hbfit:
                        initial = np.array([0.04, 0.09, 0.75, -200.0, 40.0, 150.0, 1000.0, fluxp, 0.0])
                    else:
                        initial = np.array([0.04, 0.06, 0.75, -200.0, 40.0, 150.0, 1000.0, fluxp, 0.0])
                ndim = len(initial)
                p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
                if plot_f:
                    tim=True
                else:
                    tim=False
                if single:
                    sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin_s,data,tim=tim,ncpu=ncpu)
                else:
                    #if hbfit:
                    sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)
                    #else:
                    #    sampler, pos, prob, state = mcm.mcmc(p0,nwalkers,niter,ndim,pri.lnprob_gauss_Lin,data,tim=tim,ncpu=ncpu)    
                samples = sampler.flatchain
                theta_max  = samples[np.argmax(sampler.flatlnprobability)]
                if single:
                    A1_f,A3_f,dv1_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                    if hbfit:
                    	model,m2B,mHB,m1B,mHBR=mod.line_model_s(theta_max, x=wave_i, xo1=Loiii2, xo2=LnrHb, xo3=Loiii1, ret_com=True, lfac12=lfac12)
                    else:
                        model,m2B,mHB,m1B,mHBR=mod.line_model_s(theta_max, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, ret_com=True, lfac12=lfac12)
                    model_all[:,i,j]=model
                    model_Blue[:,i,j]=m2B+m1B+mHB
                    model_Broad[:,i,j]=mHBR
                    model_param[0,i,j]=A1_f
                    model_param[1,i,j]=A1_f/lfac12
                    model_param[2,i,j]=A3_f
                    model_param[3,i,j]=A7_f
                    model_param[4,i,j]=dv1_f
                    model_param[5,i,j]=dv3_f
                    model_param[6,i,j]=fwhm1_f
                    model_param[7,i,j]=fwhm2_f
                    model_param[8,i,j]=fluxe_t
                    if cont:
                        model_param[9,i,j]=fluxpt
                else:
                    A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f=theta_max
                    if dv2_f < dv1_f:
                        fac_f=1/fac_f
                        dt=np.copy(dv2_f)
                        dv2_f=np.copy(dv1_f)
                        dv1_f=dt
                        A1_f=A1_f*fac_f
                        A3_f=A3_f*fac_f
                        theta_max=A1_f,A3_f,fac_f,dv1_f,dv2_f,fwhm1_f,fwhm2_f,A7_f,dv3_f    
                    if hbfit:
                        model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=Loiii2, xo2=LnrHb, xo3=Loiii1, ret_com=True, lfac12=lfac12)
                    else:
                        model,m2B,m2R,mHB,mHR,m1B,m1R,mHBR=mod.line_model(theta_max, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, ret_com=True, lfac12=lfac12)
                    model_all[:,i,j]=model
                    model_Blue[:,i,j]=m2B+m1B+mHB
                    model_Red[:,i,j]=m2R+m1R+mHR
                    model_Broad[:,i,j]=mHBR
                    model_param[0,i,j]=A1_f
                    model_param[1,i,j]=A1_f/lfac12
                    model_param[2,i,j]=A3_f
                    model_param[3,i,j]=A7_f
                    model_param[4,i,j]=fac_f
                    model_param[5,i,j]=A1_f/fac_f
                    model_param[6,i,j]=A1_f/fac_f/lfac12
                    model_param[7,i,j]=A3_f/fac_f
                    model_param[8,i,j]=dv1_f
                    model_param[9,i,j]=dv2_f
                    model_param[10,i,j]=dv3_f
                    model_param[11,i,j]=fwhm1_f
                    model_param[12,i,j]=fwhm2_f
                    model_param[13,i,j]=fluxe_t
                    if cont:
                        model_param[14,i,j]=fluxpt    
                if plot_f:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(7,5))#(22,6))
                    ax1 = fig.add_subplot(1,1,1)
                    ax1.plot(wave_i,fluxt,linewidth=1,color='black',label=r'Spectrum')
                    ax1.plot(wave_i,fluxtE,linewidth=0.5,color='grey',label=r'$1\sigma$ Error')
                    ax1.plot(wave_i,model,linewidth=1,color='green',label=r'Model')
                    #if hbfit:
                    #    ax1.plot(wave_i,mHbBR,linewidth=1,color='red',label=r'Hb_n_BR')
                    #else:
                    #    ax1.plot(wave_i,mHaBR,linewidth=1,color='red',label=r'Ha_n_BR')
                    if single:
                        if hbfit:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'OIII_2_NR')
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Hb_n_NR')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'OIII_1_NR')
                        else:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_NR')
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Ha_n_NR')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_NR')
                    else:
                        if hbfit:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Hb_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'OIII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'OIII_2_r')
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Hb_n_b')
                            ax1.plot(wave_i,mHR,linewidth=1,color='red',label=r'Hb_n_r')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'OIII_1_b')
                            ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'OIII_1_r')
                        else:
                            ax1.plot(wave_i,mHBR,linewidth=1,color='red',label=r'Ha_n_BR')
                            ax1.plot(wave_i,m2B,linewidth=1,color='blue',label=r'NII_2_b')
                            ax1.plot(wave_i,m2R,linewidth=1,color='red',label=r'NII_2_r')
                            ax1.plot(wave_i,mHB,linewidth=1,color='blue',label=r'Ha_n_b')
                            ax1.plot(wave_i,mHR,linewidth=1,color='red',label=r'Ha_n_r')
                            ax1.plot(wave_i,m1B,linewidth=1,color='blue',label=r'NII_1_b')
                            ax1.plot(wave_i,m1R,linewidth=1,color='red',label=r'NII_1_r')
                    fontsize=14
                    ax1.set_title("Observed Spectrum Input",fontsize=fontsize)
                    ax1.set_xlabel(r'$\lambda$ ($\rm{\AA}$)',fontsize=fontsize)
                    ax1.set_ylabel(r'$f_\lambda$ (10$^{-16}$erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)',fontsize=fontsize)
                    ax1.legend(fontsize=fontsize)
                    plt.tight_layout()
                    plt.show()
                    if single:
                        labels = ['A1','A3','dv1','FWHM_N',"FWHM_B","A7","dv3"]
                        if hbfit:
                            labels2 = [r'$A_{OIII}$',r'$A_{H\beta}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']#,"FWHM_B","A7","dv3"]
                        else:
                            labels2 = [r'$A_{NII}$',r'$A_{H\alpha}$',r'$\Delta v$',r'$FWHM_n$',r'$FWHM_b$',r'$A_{b}$',r'$\Delta v_{br}$']#,"FWHM_B","A7","dv3"]
                    else:
                        labels = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                        labels2 = ['A1','A3','fac','dv1','dv2','FWHM',"FWHM_B","A7","dv3"]
                    import corner  
                    #fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
                    #fig.set_size_inches(6.8, 6.8)
                    #plt.show()
                    #print(samples.shape)
                    fig = corner.corner(samples[:,0:len(labels2)],show_titles=True,labels=labels2,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 16})
                    fig.set_size_inches(15.8*len(labels2)/8.0, 15.8*len(labels2)/8.0)    
                    fig.savefig('corners_NAME.pdf')#.replace('NAME',name)
                
                    
                    
                    med_model, spread = mcm.sample_walkers(10, samples, x=wave_i, xo1=Lnii2, xo2=LnrHa, xo3=Lnii1, single=single, lfac12=lfac12)
                    
                    
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6*1.5,3*1.5))
                    ax1 = fig.add_subplot(1,1,1)
                    #if hbfit:
                    #    ax1.set_xlim(4800,5100)
                    #else:
                    #ax1.set_xlim(lA1,lA2)
                    ax1.plot(wave_i,fluxt,label='Input spectrum')
                    ax1.plot(wave_i,model,label='Highest Likelihood Model')
                    plt.ylabel(r'$Flux\ [10^{-16} erg/s/cm^2/\AA]$',fontsize=16)
                    plt.xlabel(r'$Wavelength\ [\AA]$',fontsize=16)
                    ax1.fill_between(wave_i,med_model-spread*50,med_model+spread*50,color='grey',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
                    ax1.legend(fontsize=14)
                    plt.tight_layout()
                    plt.savefig('spectra_mod.pdf')
                    #plt.show()
                if pgr_bar == False:  
                    if single:  
                        print("A1=",A1_f,"A3=",A3_f,"dv1=",dv1_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                    else:
                        print("A1=",A1_f,"A3=",A3_f,"FAC=",fac_f,"dv1=",dv1_f,"dv2=",dv2_f,"fwhm=",fwhm1_f,"fwhm2=",fwhm2_f,"A7=",A7_f,"dv3=",dv3_f)
                if test:        
                    sys.exit()        
            if pgr_bar:
                #progress2.update(i)
                pbar.update(1)
    
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
    if single:
        hlist=fits.HDUList([h1,h2,h4])
    else:
        hlist=fits.HDUList([h1,h2,h3,h4])
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