#!/usr/bin/env python
import numpy as np
import emcee
import MapLines.tools.models as mod

def mcmc(p0,nwalkers,niter,ndim,lnprob,data,verbose=False,multi=True,tim=False,ncpu=10):
    if tim:
        import time
    if multi:
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        ncput=cpu_count()
        if ncpu > ncput:
            ncpu=ncput
        if ncpu == 0:
            ncpu=None
        with Pool(ncpu) as pool:
        #with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,pool=pool)
            if tim:
                start = time.time()
            if verbose:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 1000)
            sampler.reset()
            if verbose:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
            if tim:
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        if tim:
            start = time.time()
        if verbose:
            print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        if verbose:
            print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        if tim:
            end = time.time()
            serial_time = end - start
            print("Serial took {0:.1f} seconds".format(serial_time))
    return sampler, pos, prob, state


def sample_walkers(nsamples,flattened_chain,x=0,xo1=0,xo2=0,xo3=0,single=False, lfac12=2.93, skew=False, broad=True, lorentz=False, n_line=False, outflow=False):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        modt = mod.line_model(i, x=x, xo1=xo1, xo2=xo2, xo3=xo3, lfac12=lfac12, single=single, skew=skew, broad=broad, lorentz=lorentz, n_line=n_line, outflow=outflow)
        models.append(modt)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread