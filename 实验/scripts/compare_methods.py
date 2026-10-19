"""Gate4 small CNN study, with scene-disjoint and seed-disjoint evaluation.

An offline teacher is never part of online features. Physical full-grid and
selected-grid likelihood evaluators are exactly the same observation model.
All timing includes feature generation, inference, selection and exact ML.
"""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
sys.path[:0]=[str(ROOT/'legacy'),str(ROOT/'src')]
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,json,time,xml.etree.ElementTree as ET
from dataclasses import asdict
import numpy as np
import main as legacy
from model import SimulationConfig,compute_ttc_weights
from radar_echo import MaintenanceEcho,EchoConfig

OUT=ROOT.parent/'结果'/'dnn_ml'
SUMO=Path(os.environ.get('SUMO_HOME',r'C:\Program Files (x86)\Eclipse\Sumo'))
SCENES=[
 {'id':0,'split':'train','gap_delta':-12,'speed_delta':-.08},
 {'id':1,'split':'train','gap_delta':12,'speed_delta':.08},
 {'id':2,'split':'train','gap_delta':-6,'speed_delta':.04},
 {'id':3,'split':'train','gap_delta':6,'speed_delta':-.04},
 {'id':4,'split':'validation','gap_delta':-15,'speed_delta':-.05},
 {'id':5,'split':'validation','gap_delta':15,'speed_delta':.05},
 {'id':6,'split':'test','gap_delta':-18,'speed_delta':-.12},
 {'id':7,'split':'test','gap_delta':18,'speed_delta':.12},
]
def dump(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')
def blocks():
    ds=np.array_split(np.arange(121),11);vs=np.array_split(np.arange(61),7)
    return [np.array([i*61+j for i in a for j in b]) for a in ds for b in vs]
BLOCKS=blocks();BLOCK_OF=np.empty(7381,dtype=int)
for bi,indices in enumerate(BLOCKS):BLOCK_OF[indices]=bi

def simulate_scene(scene):
    folder=OUT/'scenarios'/str(scene['id']);folder.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(ROOT/'legacy'/'sumo'/'p2_fixed.rou.xml')
    # Each independent scene shifts target type speed limits by the same
    # declared speed offset, preserving the historical safety margin.
    for vtype in tree.getroot().findall('vType'):
        if vtype.get('id')!='egoType':vtype.set('maxSpeed',str(float(vtype.get('maxSpeed'))+scene['speed_delta']))
    for vehicle in tree.getroot().findall('vehicle'):
        k=vehicle.get('id')
        if k!='ego':
            pos=float(vehicle.get('departPos'));sign=1 if pos>500 else -1
            vehicle.set('departPos',str(pos+sign*scene['gap_delta']))
            vehicle.set('departSpeed',str(float(vehicle.get('departSpeed'))+scene['speed_delta']))
    tree.write(folder/'scene.rou.xml',encoding='utf-8',xml_declaration=True)
    conn_name=f'dnn_scene_{os.getpid()}_{scene["id"]}';traci=legacy.load_matching_traci(SUMO)
    command=[str(SUMO/'bin'/'sumo.exe'),'-n',str(legacy.SUMO_DIR/'p2_fixed.net.xml'),'-r',str(folder/'scene.rou.xml'),
        '--step-length','.1','--end','42.1','--seed',str(100+scene['id']),'--collision.action','warn','--time-to-teleport','-1','--no-step-log','true','--duration-log.disable','true']
    traci.start(command,label=conn_name,numRetries=1);conn=traci.getConnection(conn_name);previous={};configured=set();rows=[];collisions=set();ncycle=0
    try:
        while conn.simulation.getTime()<42.1-1e-9:
            conn.simulationStep();t=round(float(conn.simulation.getTime())-2.1,10)
            states=legacy.vehicle_states_from_traci(conn,t,previous);previous={s.vehicle_id:s for s in states}
            for s in states:
                if s.vehicle_id not in configured:
                    conn.vehicle.setSpeedMode(s.vehicle_id,31);conn.vehicle.setLaneChangeMode(s.vehicle_id,512);configured.add(s.vehicle_id)
            for k,u in legacy.smooth_vehicle_speed_targets(t).items():
                if k in previous:conn.vehicle.setSpeed(k,float(u+scene['speed_delta']))
            collisions.update(conn.simulation.getCollidingVehiclesIDList())
            if t<0:continue
            ego,observations,missing=legacy.fixed_target_observations(states,'ego')
            if missing:raise RuntimeError('Scene loses targets')
            ncycle+=1
            if t%5==0:
                for i,x in enumerate(observations):rows.append({'time_s':t,'target':i,'distance_m':x.distance_m,'velocity_mps':x.closing_speed_mps})
    finally:conn.close()
    if collisions or ncycle!=401:raise RuntimeError(f'Invalid independent scene {scene}: {collisions} {ncycle}')
    legacy.write_csv(folder/'sampled_states.csv',rows);dump(folder/'scene_config.json',{**scene,'cycles':ncycle,'collisions':sorted(collisions),'seed':100+scene['id']})
    return rows

def feature(y,prior,echo):
    normalized=y/np.sqrt(echo.noise_variance)
    a=np.stack([normalized.real,normalized.imag]).astype('float32')
    # Only received samples and noisy prior are used. SNR is measured from y.
    snr=max(float(np.mean(abs(normalized)**2))-1.,0.)
    meta=np.array([prior[0]/100,prior[1]/2,np.log1p(snr)],dtype='float32')
    return a,meta

def dataset():
    if not json.loads((ROOT.parent/'结果'/'full_ml'/'episode'/'gate3_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Gate3 must pass')
    if not json.loads((ROOT.parent/'结果'/'full_ml'/'waveform_fim_validation.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Waveform/FIM must pass')
    OUT.mkdir(parents=True,exist_ok=True);echo=MaintenanceEcho()
    xs=[];ms=[];labels=[];reg=[];ys=[];prior_list=[];truth_list=[];splits=[];scenes=[];seedlist=[];bestlist=[];lossmin=[]
    for scene in SCENES:
        rows=simulate_scene(scene)
        seeds=range({'train':1000,'validation':2000,'test':3000}[scene['split']],{'train':1000,'validation':2000,'test':3000}[scene['split']]+16)
        for r in rows:
            truth=np.array([r['distance_m'],r['velocity_mps']])
            for seed in seeds:
                rng=np.random.default_rng(np.random.SeedSequence([seed,scene['id'],int(r['time_s']*10),r['target']]))
                prior=truth+rng.normal(size=2)*[echo.cfg.prior_sigma_d,echo.cfg.prior_sigma_v]
                y=echo.simulate(truth,prior,rng);d,v=echo.grid(prior);nll=echo.likelihood(y,prior,d,v);best=int(np.argmin(nll));i,j=np.unravel_index(best,nll.shape)
                x,m=feature(y,prior,echo);xs.append(x);ms.append(m);labels.append(BLOCK_OF[best]);reg.append([(d[i]-prior[0])/3,(v[j]-prior[1])/.6])
                ys.append(y);prior_list.append(prior);truth_list.append(truth);splits.append(scene['split']);scenes.append(scene['id']);seedlist.append(seed);bestlist.append(best);lossmin.append(float(nll.min()))
        print('dataset scene',scene['id'],scene['split'],'cumulative samples',len(xs),flush=True)
    np.savez_compressed(OUT/'candidate_dataset.npz',x=np.asarray(xs),meta=np.asarray(ms),label=np.asarray(labels),reg=np.asarray(reg,dtype='float32'),
        observation=np.asarray(ys),prior=np.asarray(prior_list),truth=np.asarray(truth_list),split=np.asarray(splits),scene=np.asarray(scenes),seed=np.asarray(seedlist),teacher_best=np.asarray(bestlist),teacher_nll=np.asarray(lossmin))
    dump(OUT/'dataset_config.json',{'scenes':SCENES,'sample_times_s':list(range(0,41,5)),'independent_seeds':{'train':[1000,1015],'validation':[2000,2015],'test':[3000,3015]},
        'feature_input':'current complex maintenance echo, noisy coarse distance/velocity, observed SNR; no truth, future or teacher features',
        'teacher_use':'offline labels only','regions':77,'full_grid':7381,'blocks':'11 distance blocks x 7 velocity blocks, disjoint partition',
        'threshold_selection':'validation only; select highest score regions until cumulative mass threshold; allow multi-peak; fallback if budget>40%, boundary or standardized residual anomalous',
        'training_seed':66771,'epochs':40,'batch_size':128})

def train_and_compare(retrain=True):
    if not json.loads((ROOT.parent/'结果'/'full_ml'/'episode'/'gate3_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Gate3 must pass before training/evaluation')
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(66771);np.random.seed(66771)
    class CandidateCNN(nn.Module):
        def __init__(self):
            super().__init__();self.conv=nn.Sequential(nn.Conv2d(2,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,8)),nn.Flatten())
            self.fc=nn.Sequential(nn.Linear(16*3*8+3,96),nn.ReLU());self.classifier=nn.Linear(96,77);self.regressor=nn.Linear(96,2)
        def forward(self,x,m):
            z=self.fc(torch.cat([self.conv(x),m],dim=1));return self.classifier(z),torch.tanh(self.regressor(z))
    z=np.load(OUT/'candidate_dataset.npz');x=torch.from_numpy(z['x']);m=torch.from_numpy(z['meta']);y=torch.from_numpy(z['label']).long();reg=torch.from_numpy(z['reg'])
    train=np.where(z['split']=='train')[0];val=np.where(z['split']=='validation')[0];test=np.where(z['split']=='test')[0]
    for a,b in [(train,val),(train,test),(val,test)]:
        if set(z['scene'][a])&set(z['scene'][b]) or set(z['seed'][a])&set(z['seed'][b]):raise RuntimeError('Scene/seed leakage between dataset splits')
    net=CandidateCNN();optimizer=torch.optim.Adam(net.parameters(),lr=.001);log=[];bestval=float('inf');beststate=None
    rng=np.random.default_rng(66771)
    for epoch in range(40 if retrain else 0):
        net.train();total=0.
        for ids in np.array_split(rng.permutation(train),int(np.ceil(len(train)/128))):
            optimizer.zero_grad();pred,direct=net(x[ids],m[ids]);loss=nn.functional.cross_entropy(pred,y[ids])+.5*nn.functional.mse_loss(direct,reg[ids]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        net.eval()
        with torch.no_grad():pred,direct=net(x[val],m[val]);vl=float(nn.functional.cross_entropy(pred,y[val])+.5*nn.functional.mse_loss(direct,reg[val]))
        if vl<bestval:bestval=vl;beststate={k:v.clone() for k,v in net.state_dict().items()}
        log.append({'epoch':epoch+1,'train_loss':total/len(train),'validation_loss':vl})
        if epoch%10==0:print('training',epoch+1,total/len(train),vl,flush=True)
    if not retrain:beststate=torch.load(OUT/'candidate_cnn.pt',weights_only=False)['state_dict']
    net.load_state_dict(beststate);net.eval()
    if retrain:
        torch.save({'state_dict':beststate,'torch_version':str(torch.__version__),'regions':77},OUT/'candidate_cnn.pt');legacy.write_csv(OUT/'training_log.csv',log)
    echo=MaintenanceEcho()
    with torch.inference_mode():vp=torch.softmax(net(x[val],m[val])[0],dim=1).numpy()
    choices=[]
    for threshold in [.95,.99,.995,.999]:
        hits=[];budget=[]
        for row,idx in zip(vp,val):
            order=np.argsort(row)[::-1];k=np.searchsorted(np.cumsum(row[order]),threshold)+1;sel=order[:k]
            hits.append(int(z['label'][idx] in sel));budget.append(sum(len(BLOCKS[b]) for b in sel)/7381)
        choices.append({'threshold':threshold,'validation_recall':float(np.mean(hits)),'mean_budget':float(np.mean(budget))})
    acceptable=[c for c in choices if c['validation_recall']>=.99]
    chosen=min(acceptable,key=lambda c:c['mean_budget']) if acceptable else choices[-1]
    dump(OUT/'validation_thresholds.json',{'choices':choices,'chosen':chosen,'test_used_for_selection':False})
    # Warm all execution paths. No cached teacher outputs participate below.
    with torch.inference_mode():net(x[val[:1]],m[val[:1]])
    rows=[]
    for it,idx in enumerate(test):
        observation=z['observation'][idx];prior=z['prior'][idx];truth=z['truth'][idx];ds,vs=echo.grid(prior)
        for method in ['Full-Grid ML','Coarse-to-Fine ML','DNN Direct Regression','DNN-Adaptive-ML']:
            start=time.perf_counter();fallback=False;pre_hit=None;retained_indices=None
            if method=='Full-Grid ML':
                nl=echo.likelihood(observation,prior,ds,vs);best=int(np.argmin(nl));di,vi=np.unravel_index(best,nl.shape);est=np.array([ds[di],vs[vi]]);minimum=float(nl.min());count=nl.size;hits=True
            elif method=='Coarse-to-Fine ML':
                di=np.unique(np.r_[np.arange(0,len(ds),4),len(ds)-1]);vi=np.unique(np.r_[np.arange(0,len(vs),4),len(vs)-1]);coarse=echo.likelihood(observation,prior,ds[di],vs[vi]);a,b=np.unravel_index(np.argmin(coarse),coarse.shape)
                fine_d=np.arange(max(0,di[a]-4),min(len(ds),di[a]+5));fine_v=np.arange(max(0,vi[b]-4),min(len(vs),vi[b]+5));nl=echo.likelihood(observation,prior,ds[fine_d],vs[fine_v]);a,b=np.unravel_index(np.argmin(nl),nl.shape);best=int(fine_d[a]*61+fine_v[b]);est=np.array([ds[fine_d[a]],vs[fine_v[b]]]);minimum=float(nl.min());count=len(di)*len(vi)+nl.size;retained_indices=(fine_d[:,None]*61+fine_v[None,:]).ravel()
            else:
                fx,fm=feature(observation,prior,echo)
                with torch.inference_mode():scores,direct=net(torch.from_numpy(fx[None]),torch.from_numpy(fm[None]));probs=torch.softmax(scores,dim=1)[0].numpy();delta=direct[0].numpy()
                if method=='DNN Direct Regression':
                    est=prior+delta*[3.,.6];minimum=float(np.sum(abs(observation-echo.mean(*est,prior))**2)/echo.noise_variance);count=1;best=-1;hits=False
                else:
                    order=np.argsort(probs)[::-1];k=np.searchsorted(np.cumsum(probs[order]),chosen['threshold'])+1;selected=np.concatenate([BLOCKS[b] for b in order[:k]])
                    retained_indices=selected;count=len(selected)
                    # Selected physical likelihood is evaluated before fallback.
                    nl=echo.selected_likelihood(observation,prior,ds,vs,selected);best=int(selected[np.argmin(nl)]);minimum=float(nl.min());a,b=np.unravel_index(best,(121,61))
                    residual=minimum/observation.size
                    fallback=len(selected)>.4*7381 or a in [0,120] or b in [0,60] or residual>1.3
                    if fallback:
                        nl=echo.likelihood(observation,prior,ds,vs);count+=nl.size;best=int(np.argmin(nl));minimum=float(nl.min())
                    a,b=np.unravel_index(best,(121,61));est=np.array([ds[a],vs[b]])
            elapsed=time.perf_counter()-start
            # Offline teacher checks are strictly outside online execution and
            # outside the measured preprocessing/inference/ML interval.
            if retained_indices is not None:
                pre_hit=bool(np.any(retained_indices==int(z['teacher_best'][idx])))
                hits=pre_hit or fallback
            rows.append({'case_index':int(idx),'scene':int(z['scene'][idx]),'noise_seed':int(z['seed'][idx]),'method':method,
                'estimated_d':est[0],'estimated_v':est[1],'distance_error_m':est[0]-truth[0],'velocity_error_mps':est[1]-truth[1],
                'seconds':elapsed,'likelihood_evaluations':count,'retained_fraction':count/7381,'candidate_recall':int(hits),
                'pre_fallback_recall':int(pre_hit) if pre_hit is not None else '', 'fallback':int(fallback),
                'minimum_nll':minimum,'nll_gap_full_ml':minimum-float(z['teacher_nll'][idx]),'best_flat_index':best})
        if it%200==0:print('test comparisons',it+1,len(test),flush=True)
    legacy.write_csv(OUT/'comparison_test.csv',rows)
    sums=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==method]
        sums.append({'method':method,'n':len(rr),'distance_rmse_m':float(np.sqrt(np.mean([r['distance_error_m']**2 for r in rr]))),
            'velocity_rmse_mps':float(np.sqrt(np.mean([r['velocity_error_mps']**2 for r in rr]))),
            'mean_seconds':float(np.mean([r['seconds'] for r in rr])),'median_seconds':float(np.median([r['seconds'] for r in rr])),
            'candidate_recall':float(np.mean([r['candidate_recall'] for r in rr])),'fallback_frequency':float(np.mean([r['fallback'] for r in rr])),
            'mean_likelihood_evaluations':float(np.mean([r['likelihood_evaluations'] for r in rr])),
            'max_nll_gap':max(r['nll_gap_full_ml'] for r in rr)})
    base=sums[0]
    for r in sums:r['end_to_end_speedup']=base['mean_seconds']/r['mean_seconds']
    legacy.write_csv(OUT/'comparison_summary.csv',sums)
    dl=sums[-1]
    checks={'candidate_recall_ge_099':dl['candidate_recall']>=.99,
       'distance_rmse_not_worse_5pct':dl['distance_rmse_m']<=1.05*base['distance_rmse_m'],
       'velocity_rmse_not_worse_5pct':dl['velocity_rmse_mps']<=1.05*base['velocity_rmse_mps'],
       'speedup_above_1':dl['end_to_end_speedup']>1.,'separate_scenes_and_seeds':True,
       'test_cases':len(test),'torch_version':torch.__version__,'full_ml_summary':base,'dnn_ml_summary':dl,
       'integration_into_P1_P2':False}
    checks['passed']=all(checks[k] for k in ['candidate_recall_ge_099','distance_rmse_not_worse_5pct','velocity_rmse_not_worse_5pct','speedup_above_1','separate_scenes_and_seeds'])
    dump(OUT/'gate4_checks.json',checks)
    dump(OUT/'failures.json',[r for r in rows if r['method']=='DNN-Adaptive-ML' and (r['candidate_recall']==0 or r['nll_gap_full_ml']>1e-7)])
    print(json.dumps(checks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['data','train','evaluate','all'],default='all');a=p.parse_args()
    if a.stage in ['data','all']:dataset()
    if a.stage in ['train','all']:train_and_compare()
    if a.stage=='evaluate':train_and_compare(retrain=False)
