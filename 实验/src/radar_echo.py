"""Versioned shadow maintenance observation with candidate-dependent amplitude.

This is not the frozen-amplitude legacy EFIM. Coherent RCS and reference phase
are known calibration assumptions. 200 MHz LFM is acquired at 250 MHz before
deramping by the coarse track, ideal residual-band filtering and 2 MHz sampling.
Only a common interior pulse gate is retained. The projected-data likelihood,
not the discarded raw waveform likelihood, is evaluated. Noise is proper
complex AWGN; coherent four-pulse averages are modelled exactly.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class EchoConfig:
    fc:float=77e9
    bandwidth:float=200e6
    pulse_duration:float=5e-6
    acquisition_fs:float=250e6
    residual_fs:float=2e6
    pulse_count:int=256
    pri:float=40e-6
    coherent_group:int=4
    fast_samples:int=6
    maintenance_power:float=.05
    rcs:float=10.
    loss:float=2.
    tx_gain:float=4.
    rx_gain:float=4.
    noise_temperature:float=290.
    noise_figure:float=10**.6
    prior_sigma_d:float=.75
    prior_sigma_v:float=.15
    halfwidth_d:float=3.
    halfwidth_v:float=.6
    grid_step_d:float=.05
    grid_step_v:float=.02
    prior_seed:int=47201
    noise_seed_base:int=81000
    phase_rad:float=0.
    model_id:str='shadow_deramped_known_rcs_phase_amplitude_dminus2_v1'

class MaintenanceEcho:
    c=299792458.
    kb=1.380649e-23
    def __init__(self,cfg=EchoConfig()):
        self.cfg=cfg
        self.kappa=cfg.bandwidth/cfg.pulse_duration
        self.u=cfg.pulse_duration/2+(np.arange(cfg.fast_samples)-(cfg.fast_samples-1)/2)/cfg.residual_fs
        g=cfg.coherent_group
        self.t=np.arange(cfg.pulse_count//g)*g*cfg.pri
        self.within=np.arange(g)*cfg.pri
        self.noise_psd=self.kb*cfg.noise_temperature*cfg.noise_figure
        self.noise_variance=self.noise_psd*cfg.residual_fs/g
        self.amplitude_const=np.sqrt(cfg.maintenance_power*cfg.tx_gain*cfg.rx_gain*(self.c/cfg.fc)**2*cfg.rcs/((4*np.pi)**3*cfg.loss))
        # Acceptance bounds are fixed before any noisy data are evaluated.
        if cfg.acquisition_fs<cfg.bandwidth:raise ValueError('Raw acquisition undersamples nominal LFM')
        max_beat=2*self.kappa*cfg.halfwidth_d/self.c+2*cfg.fc*5/self.c
        if max_beat>=cfg.residual_fs/2:raise ValueError('Residual beat aliases within candidate window')
        if np.min(self.u)-2*cfg.halfwidth_d/self.c<=0 or np.max(self.u)+2*cfg.halfwidth_d/self.c>=cfg.pulse_duration:raise ValueError('Candidate leaves common pulse gate')
    def grid(self,prior):
        c=self.cfg
        ds=prior[0]+np.arange(round(2*c.halfwidth_d/c.grid_step_d)+1)*c.grid_step_d-c.halfwidth_d
        vs=prior[1]+np.arange(round(2*c.halfwidth_v/c.grid_step_v)+1)*c.grid_step_v-c.halfwidth_v
        if ds.min()<=0:raise ValueError('Invalid range prior')
        return ds,vs
    def factors(self,ds,vs,prior):
        dtau=2*(np.asarray(ds)-prior[0])/self.c
        delay=np.exp(1j*(np.pi*self.kappa*dtau[:,None]**2-2*np.pi*self.kappa*dtau[:,None]*(self.u-self.cfg.pulse_duration/2)))
        doppler=2*np.asarray(vs)*self.cfg.fc/self.c
        fast=np.exp(1j*2*np.pi*doppler[:,None]*(self.u+2*prior[0]/self.c))
        avg=np.mean(np.exp(1j*2*np.pi*doppler[:,None]*self.within),axis=1)
        slow=np.exp(1j*2*np.pi*doppler[:,None]*self.t)*avg[:,None]
        amp=self.amplitude_const/np.asarray(ds)**2*np.exp(1j*self.cfg.phase_rad)
        return amp,delay,fast,slow
    def mean(self,d,v,prior):
        a,u,f,t=self.factors([d],[v],prior)
        return a[0]*u[0,:,None]*f[0,:,None]*t[0,None,:]
    def simulate(self,truth,prior,rng):
        mu=self.mean(*truth,prior)
        noise=np.sqrt(self.noise_variance/2)*(rng.standard_normal(mu.shape)+1j*rng.standard_normal(mu.shape))
        return mu+noise
    def likelihood(self,y,prior,ds=None,vs=None):
        if ds is None:ds,vs=self.grid(prior)
        amp,u,fast,slow=self.factors(ds,vs,prior)
        # Exact squared norm for all Cartesian candidates, without making
        # 7381 copies of the measured complex observation.
        correlation=u.conj()@((y@slow.conj().T)*fast.conj().T)
        meansq=abs(amp[:,None])**2*len(self.u)*np.sum(abs(slow)**2,axis=1)[None,:]
        nll=(np.sum(abs(y)**2)+meansq-2*np.real(amp.conj()[:,None]*correlation))/self.noise_variance
        return np.maximum(nll,0.)
    def candidate_loss(self,y,prior,ds,vs,indices):
        # Selected candidates only. Online DNN+ML must call this exact evaluator.
        ids=np.asarray(indices,dtype=int);i,j=np.unravel_index(ids,(len(ds),len(vs)))
        amp,u,fast,slow=self.factors(ds[i],vs[j],prior)
        mu=amp[:,None,None]*u[:,:,None]*fast[:,:,None]*slow[:,None,:]
        return np.sum(abs(y[None,:,:]-mu)**2,axis=(1,2))/self.noise_variance
    def selected_likelihood(self,y,prior,ds,vs,indices):
        """Exact same loss on selected states, with separable factors only."""
        indices=np.asarray(indices,dtype=int)
        di,vi=np.unravel_index(indices,(len(ds),len(vs)))
        unique_v,inverse=np.unique(vi,return_inverse=True)
        a,u,f,t=self.factors(ds,vs[unique_v],prior)
        projection=y@t.conj().T
        cross=np.sum(u[di].conj()*f[inverse].conj()*projection[:,inverse].T,axis=1)
        energy=abs(a[di])**2*len(self.u)*np.sum(abs(t[inverse])**2,axis=1)
        return np.maximum((np.sum(abs(y)**2)+energy-2*np.real(a[di].conj()*cross))/self.noise_variance,0.)
    def fim(self,d,v,prior):
        eps=[1e-4,1e-5]
        dd=(self.mean(d+eps[0],v,prior)-self.mean(d-eps[0],v,prior))/(2*eps[0])
        dv=(self.mean(d,v+eps[1],prior)-self.mean(d,v-eps[1],prior))/(2*eps[1])
        deriv=np.stack([dd.ravel(),dv.ravel()],axis=1)
        j=2*np.real(deriv.conj().T@deriv)/self.noise_variance
        return j,np.linalg.inv(j)
