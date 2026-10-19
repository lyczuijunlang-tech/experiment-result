"""Independent analytic checks of the dechirped maintenance observation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src')]
import numpy as np
from radar_echo import MaintenanceEcho

def main():
    echo=MaintenanceEcho();c=echo.cfg;prior=np.array([95.,.6]);d=95.4;v=.9
    tau=2*d/echo.c;tp=2*prior[0]/echo.c;u=tp+echo.u
    pulses=np.arange(c.pulse_count)*c.pri
    original=echo.amplitude_const/d**2*np.exp(1j*np.pi*echo.kappa*(u[:,None]-tau-c.pulse_duration/2)**2)*np.exp(1j*4*np.pi*c.fc/echo.c*v*(u[:,None]+pulses))
    deramped=original*np.exp(-1j*np.pi*echo.kappa*(echo.u[:,None]-c.pulse_duration/2)**2)
    direct=deramped.reshape(c.fast_samples,-1,c.coherent_group).mean(axis=2)
    mu=echo.mean(d,v,prior);identity=np.linalg.norm(mu-direct)/np.linalg.norm(direct)
    dtau=2*(d-prior[0])/echo.c
    dd=mu*(-2/d+1j*4*np.pi*echo.kappa/echo.c*(dtau-(echo.u[:,None]-c.pulse_duration/2)))
    f=2*v*c.fc/echo.c;average_deriv=np.mean(1j*4*np.pi*c.fc/echo.c*echo.within*np.exp(1j*2*np.pi*f*echo.within))
    amp,ur,fast,slow=echo.factors([d],[v],prior)
    avg=np.mean(np.exp(1j*2*np.pi*f*echo.within))
    dv=mu*1j*4*np.pi*c.fc/echo.c*(u[:,None]+echo.t[None,:])+mu/avg*average_deriv
    grad=np.stack([dd.ravel(),dv.ravel()],axis=1)
    analytic=2*np.real(grad.conj().T@grad)/echo.noise_variance
    numerical,crb=echo.fim(d,v,prior)
    relative=np.linalg.norm(analytic-numerical)/np.linalg.norm(analytic)
    rng=np.random.default_rng(771);y=echo.simulate([d,v],prior,rng);ds,vs=echo.grid(prior);ids=rng.choice(7381,100,replace=False)
    candidate_error=float(np.max(abs(echo.selected_likelihood(y,prior,ds,vs,ids)-echo.candidate_loss(y,prior,ds,vs,ids))))
    raw_t=np.arange(round(c.pulse_duration*c.acquisition_fs))/c.acquisition_fs
    raw_waveform=np.exp(1j*np.pi*echo.kappa*(raw_t-c.pulse_duration/2)**2)
    np.savez_compressed(ROOT.parent/'结果'/'full_ml'/'maintenance_waveform_samples.npz',time_s=raw_t,unit_power_lfm=raw_waveform)
    r={'raw_lfm_samples':len(raw_t),'raw_fs_hz':c.acquisition_fs,'raw_bandwidth_hz':c.bandwidth,
       'noiseless_deramp_identity_relative_error':identity,'analytic_vs_numerical_fim_relative_error':relative,
       'selected_vs_bruteforce_nll_max_abs_error':candidate_error,'FIM':analytic.tolist(),'CRB':crb.tolist(),
       'noise_generation':'Analytic ideal post-deramp white-noise model; raw high-rate ADC noise/filter transients are not numerically propagated',
       'no_FIM_nuisance_projection':True,'amplitude_derivative_included':True,
       'passed':bool(identity<1e-9 and relative<1e-6 and candidate_error<1e-8)}
    (ROOT.parent/'结果'/'full_ml'/'waveform_fim_validation.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r),flush=True)
    if not r['passed']:raise RuntimeError('Waveform validation failed; no DNN integration')
if __name__=='__main__':main()
