"""Gate3: independent shadow echoes, full-grid ML and saved likelihood maps."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,json,time
from dataclasses import asdict
import numpy as np
from radar_echo import EchoConfig,MaintenanceEcho
sys.path.insert(0,str(ROOT/'legacy'))
import main as legacy

OUT=ROOT.parent/'结果'/'full_ml'
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')
def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def evaluate_cases(rows,seeds,tag):
    echo=MaintenanceEcho();c=echo.cfg
    records=[];maps=[];ys=[];priors=[];truths=[];fims=[];dslist=[];vslist=[]
    for n,r in enumerate(rows):
        cid=int(r['cycle_index']);target=list(legacy.FIXED_IDS).index(r['slot'])
        truth=np.array([float(r['distance_m']),float(r['closing_speed_mps'])])
        for seed in seeds:
            # This is a simulated noisy pre-existing tracker measurement.
            # Truth is only accessed by the OFFLINE sensor generator and scorer.
            # No estimator accepts truth or a teacher optimum as input.
            rng_prior=np.random.default_rng(np.random.SeedSequence([c.prior_seed,seed,cid,target]))
            prior=truth+rng_prior.normal(size=2)*[c.prior_sigma_d,c.prior_sigma_v]
            rng_noise=np.random.default_rng(np.random.SeedSequence([c.noise_seed_base,seed,cid,target]))
            start=time.perf_counter();y=echo.simulate(truth,prior,rng_noise);elapsed_generation=time.perf_counter()-start
            ds,vs=echo.grid(prior)
            start=time.perf_counter();loss=echo.likelihood(y,prior,ds,vs);flat=int(np.argmin(loss));ix,iv=np.unravel_index(flat,loss.shape);elapsed=time.perf_counter()-start
            j,crb=echo.fim(*truth,prior)
            # Brute-force reconstruction checks the factorized physical likelihood.
            probes=np.array([0,loss.size//2,flat,loss.size-1]);ref=echo.candidate_loss(y,prior,ds,vs,probes)
            error=float(np.max(abs(ref-loss.ravel()[probes])))
            noiseless=echo.mean(*truth,prior);zero=echo.likelihood(noiseless,prior,ds,vs);zi,zv=np.unravel_index(np.argmin(zero),zero.shape)
            inside=ds[0]<=truth[0]<=ds[-1] and vs[0]<=truth[1]<=vs[-1]
            records.append({'case':tag,'cycle_index':cid,'time_s':float(r['time_s']),'target_name':f'k{target+1}','slot':r['slot'],'noise_seed':seed,
              'true_distance_m':truth[0],'true_velocity_mps':truth[1],'coarse_distance_m':prior[0],'coarse_velocity_mps':prior[1],
              'estimated_distance_m':ds[ix],'estimated_velocity_mps':vs[iv],'distance_error_m':ds[ix]-truth[0],'velocity_error_mps':vs[iv]-truth[1],
              'grid_d_min':ds[0],'grid_d_max':ds[-1],'grid_v_min':vs[0],'grid_v_max':vs[-1],
              'candidate_count':loss.size,'likelihood_evaluations':loss.size,'best_flat_index':flat,'best_d_index':ix,'best_v_index':iv,
              'minimum_nll':loss[ix,iv],'full_ml_seconds':elapsed,'generation_seconds':elapsed_generation,
              'truth_in_grid':int(inside),'boundary_optimum':int(ix in [0,len(ds)-1] or iv in [0,len(vs)-1]),
              'brute_force_loss_max_abs_error':error,'noiseless_distance_error_m':ds[zi]-truth[0],'noiseless_velocity_error_mps':vs[zv]-truth[1],
              'fim_min_eigenvalue':float(np.linalg.eigvalsh(j).min()),'CRBd':crb[0,0],'CRBv':crb[1,1],
              'receive_energy_snr':float(np.sum(abs(noiseless)**2)/echo.noise_variance)})
            maps.append(loss.astype('float32'));ys.append(y);priors.append(prior);truths.append(truth);dslist.append(ds);vslist.append(vs);fims.append(j)
        if n%80==0:print(f'{tag}: {n+1}/{len(rows)} traffic targets',flush=True)
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    legacy.write_csv(folder/'full_ml_estimates.csv',records)
    legacy.write_csv(folder/'full_ml_runtime.csv',[{k:r[k] for k in ['cycle_index','time_s','target_name','noise_seed','candidate_count','likelihood_evaluations','full_ml_seconds','generation_seconds']} for r in records])
    np.savez_compressed(folder/'full_ml_likelihood_maps.npz',nll=np.asarray(maps),observation=np.asarray(ys),prior=np.asarray(priors),truth=np.asarray(truths),distance_grid=np.asarray(dslist),velocity_grid=np.asarray(vslist),fim=np.asarray(fims))
    aggregate=[]
    for target in ['all','k1','k2','k3','k4']:
        rr=[r for r in records if target=='all' or r['target_name']==target]
        aggregate.append({'target':target,'n':len(rr),'distance_rmse_m':float(np.sqrt(np.mean([r['distance_error_m']**2 for r in rr]))),
          'velocity_rmse_mps':float(np.sqrt(np.mean([r['velocity_error_mps']**2 for r in rr]))),
          'prior_distance_rmse_m':float(np.sqrt(np.mean([(r['coarse_distance_m']-r['true_distance_m'])**2 for r in rr]))),
          'prior_velocity_rmse_mps':float(np.sqrt(np.mean([(r['coarse_velocity_mps']-r['true_velocity_mps'])**2 for r in rr]))),
          'mean_full_ml_seconds':float(np.mean([r['full_ml_seconds'] for r in rr])),
          'boundary_count':sum(r['boundary_optimum'] for r in rr),'truth_outside_count':sum(not r['truth_in_grid'] for r in rr)})
    legacy.write_csv(folder/'full_ml_rmse_summary.csv',aggregate)
    a=aggregate[0]
    checks={'likelihood_matches_brute_force':max(r['brute_force_loss_max_abs_error'] for r in records)<1e-8,
      'positive_fim':min(r['fim_min_eigenvalue'] for r in records)>0,
      'noiseless_grid_errors_within_one_bin':max(abs(r['noiseless_distance_error_m']) for r in records)<=c.grid_step_d and max(abs(r['noiseless_velocity_error_mps']) for r in records)<=c.grid_step_v,
      'distance_rmse_below_1m':a['distance_rmse_m']<1.,'velocity_rmse_below_025':a['velocity_rmse_mps']<.25,
      'truth_outside_count':a['truth_outside_count'],'model_id':c.model_id,'scope':'offline shadow, not charged to legacy resource budget',
      'actual_metrics':a,'acceptance_thresholds_fixed_in_source':True,'contains_full_likelihood_maps':True}
    checks['passed']=all(checks[k] for k in ['likelihood_matches_brute_force','positive_fim','noiseless_grid_errors_within_one_bin','distance_rmse_below_1m','velocity_rmse_below_025']) and a['truth_outside_count']==0
    dump(folder/'gate3_checks.json',checks)
    return checks

def figures(tag):
    import matplotlib.pyplot as plt
    folder=OUT/tag;z=np.load(folder/'full_ml_likelihood_maps.npz')
    # Four independent single-panel examples avoid squeezing 12 heatmaps.
    for k in range(4):
        i=k*16 if tag=='snapshots' else k
        ds=z['distance_grid'][i];vs=z['velocity_grid'][i];loss=z['nll'][i];ix,iv=np.unravel_index(np.argmin(loss),loss.shape)
        fig,ax=plt.subplots(figsize=(6.25,5));im=ax.pcolormesh(ds,vs,(loss-loss.min()).T,shading='auto',cmap='viridis',vmin=0,vmax=20)
        fig.colorbar(im,ax=ax,label='Negative log likelihood difference')
        ax.scatter(*z['truth'][i],marker='x',c='white',s=65,label='True state')
        ax.scatter(*z['prior'][i],marker='+',c='orange',s=70,label='Coarse track')
        ax.scatter(ds[ix],vs[iv],facecolors='none',edgecolors='red',s=70,label='Full-grid ML')
        ax.set(xlabel='Distance (m)',ylabel='Radial closing speed (m/s)',xlim=(ds[0],ds[-1]),ylim=(vs[0],vs[-1]));ax.legend(fontsize=8,loc='best')
        legacy.save_figure(fig,folder/('full_ml_examples' if k==0 else f'full_ml_example_k{k+1}'))

def main(stage):
    gate2=ROOT.parent/'结果'/'legacy_reproduction'/'gate2_checks.json'
    if not gate2.exists() or not json.loads(gate2.read_text(encoding='utf-8'))['passed']:raise RuntimeError('Gate2 must pass before Gate3')
    OUT.mkdir(parents=True,exist_ok=True)
    from validate_echo import main as validate_observation
    validate_observation()
    c=EchoConfig()
    config=asdict(c);config.update({'full_grid_candidates':121*61,'known_parameters':'RCS=10 m2, phase=0, gains=4/4, L=2; fixed during a case',
      'noise':'proper complex Gaussian, E|n|^2=N0*residual_fs/coherent_group',
      'timing':'prior track and maintenance echo exist before next resource decision; shadow observation only',
      'budget':'10.24 ms wall observation per target, 1.28 ms transmit-on time, 64 microjoule per target; excluded from legacy frame',
      'sampling':'250 MHz raw LFM -> coarse-delay deramp -> ideal 1 MHz lowpass -> 2 MHz residual samples; common 3 us interior gate; exact four-pulse coherent average',
      'low_dimensional_likelihood':'Full-grid search of retained projected observations; not claimed lossless for discarded raw samples',
      'prior':'independent simulated pre-existing tracker measurement; sigma_d=0.75m,sigma_v=0.15m/s; truth only in offline generator and scoring',
      'latest_FIM':'2 Re(Dmu^H Dmu)/noise_variance including d^-2 amplitude derivative and all cross terms; no nuisance-amplitude projection',
      'raw_waveform':'unit-power LFM with rectangular 5us support, interior gate avoids nonregular pulse edges; stop-and-hop with continuous Doppler',
      'seeds':{'snapshots':list(range(16)),'episode':[20,21,22]},'model_version':'independent shadow ML; not substituted into original figure7/8/9'})
    dump(OUT/'maintenance_echo_config.json',config)
    allrows=read(ROOT.parent/'结果'/'legacy_reproduction'/'rerun'/'targets.csv')
    if stage in ['snapshots','all']:
        r=evaluate_cases([r for r in allrows if float(r['time_s']) in [0.,20.,40.]],range(16),'snapshots');figures('snapshots')
        print(json.dumps(r,default=lambda x:x.item()),flush=True)
        if not r['passed']:raise RuntimeError('Snapshot Gate3 failed: do not run episode or train DNN')
    if stage in ['episode','all']:
        if not json.loads((OUT/'snapshots'/'gate3_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Snapshots must pass')
        r=evaluate_cases(allrows,[20,21,22],'episode');print(json.dumps(r,default=lambda x:x.item()),flush=True)
        if not r['passed']:raise RuntimeError('Episode Gate3 failed: do not train DNN')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','snapshots','episode'],default='all');main(p.parse_args().stage)
