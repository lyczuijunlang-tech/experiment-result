"""PyCharm entry point: immutable legacy SUMO and resource reproduction gates."""
from pathlib import Path

# Resolve the single stored copy after repository artifact deduplication.
import sys
for _parent in Path(__file__).resolve().parents:
    if (_parent/'实验/scripts/artifact_paths.py').exists():
        sys.path.insert(0,str(_parent/'实验/scripts'));break
try:
    from artifact_paths import resolve_artifact, figure_output, materialize_directory
except ModuleNotFoundError:
    def resolve_artifact(path):return Path(path)
    def figure_output(bundle,name):return Path(bundle)/name
    def materialize_directory(source,destination):
        import shutil
        shutil.copytree(source,destination,dirs_exist_ok=True)
import sys, os
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.deps').exists(): sys.path.insert(0,str(ROOT/'.deps'))
sys.path.insert(0,str(ROOT/'legacy'))
import argparse, csv, json, hashlib, importlib.metadata, importlib.util, subprocess
from dataclasses import asdict
import numpy as np
import main as legacy
import model

HISTORY=ROOT/'reference'/'legacy_results'
RESULT=ROOT.parent/'结果'/'legacy_reproduction'
SUMO=Path(os.environ.get('SUMO_HOME',r'C:\Program Files (x86)\Eclipse\Sumo'))

def read(path):
    with resolve_artifact(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(resolve_artifact(path).read_bytes()).hexdigest()
def gate0():
    import scipy,matplotlib
    versions={'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,
              'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
              'pytorch':importlib.metadata.version('torch') if importlib.util.find_spec('torch') else None,
              'SUMO':subprocess.check_output([str(SUMO/'bin'/'sumo.exe'),'--version'],text=True).splitlines()[0],
              'seeds':{'traffic':17,'maintenance_noise':'configured separately before shadow experiment'},
              'paper_naming':{'P1':'system time and power AO (legacy alternating_optimize_system)',
                              'P2':'target dwell allocation (legacy solve_p1)'},'source_hashes':{}}
    source_manifest=json.loads((ROOT/'reference'/'source_sha256.json').read_text(encoding='utf-8'))
    for name in ['main.py','model.py','sumo/p2_fixed.net.xml','sumo/p2_fixed.rou.xml','sumo/p2_fixed.sumocfg']:
        a=ROOT/'legacy'/name
        if sha(a)!=source_manifest[name]:raise RuntimeError('Legacy source not identical: '+name)
        versions['source_hashes'][name]=sha(a)
    prior=json.loads((HISTORY/'run_summary.json').read_text(encoding='utf-8'))
    versions['history_network_hash_match']=all(sha(ROOT/'legacy'/'sumo'/k)==v for k,v in prior['input_sha256'].items())
    if not versions['history_network_hash_match']:raise RuntimeError('History SUMO hashes differ')
    dump(RESULT/'environment_versions.json',versions)
    print('GATE0 PASS same SUMO, exact legacy source and scenario',flush=True)

def compare_rows(actual,expected,keyfields,columns,abs_tol,rel_tol=0):
    aa={tuple(str(r[k]) for k in keyfields):r for r in actual};bb={tuple(str(r[k]) for k in keyfields):r for r in expected}
    if aa.keys()!=bb.keys():return {'passed':False,'reason':'row keys differ','actual':len(aa),'expected':len(bb)}
    out={}
    for col in columns:
        errors=[];relative=[];bad=0
        for key,a in aa.items():
            x=float(a[col]);y=float(bb[key][col])
            if np.isinf(x) and x==y:continue
            diff=abs(x-y);errors.append(diff);relative.append(diff/max(abs(y),1e-30))
            bad+=int(not np.isfinite(diff) or diff>abs_tol+rel_tol*abs(y))
        out[col]={'max_abs':max(errors,default=0),'max_rel':max(relative,default=0),'mismatches':bad}
    return {'passed':all(v['mismatches']==0 for v in out.values()),'rows':len(aa),'abs_tol':abs_tol,'rel_tol':rel_tol,'columns':out}

def gate1():
    gate0();RESULT.mkdir(parents=True,exist_ok=True)
    sim=model.SimulationConfig();traci=legacy.load_matching_traci(SUMO)
    command=[str(SUMO/'bin'/'sumo.exe'),'-c',str(legacy.SUMO_DIR/'p2_fixed.sumocfg'),
             '--step-length',str(sim.step_s),'--end',str(sim.warmup_s+sim.duration_s+sim.step_s),
             '--seed',str(sim.seed),'--collision.action','warn','--time-to-teleport','-1',
             '--no-step-log','true','--duration-log.disable','true','--log',str(RESULT/'traffic_sumo.log')]
    label=f'reproduction_traffic_{os.getpid()}';traci.start(command,label=label,numRetries=1)
    conn=traci.getConnection(label);previous={};configured=set();traffic=[];targets=[];combined=[];collisions=set();ids_by_cycle=[]
    try:
        while conn.simulation.getTime()<sim.warmup_s+sim.duration_s+sim.step_s-1e-9:
            conn.simulationStep();st=float(conn.simulation.getTime());t=round(st-sim.warmup_s-sim.step_s,10)
            states=legacy.vehicle_states_from_traci(conn,t,previous);previous={s.vehicle_id:s for s in states}
            for s in states:
                if s.vehicle_id not in configured:
                    conn.vehicle.setSpeedMode(s.vehicle_id,31);conn.vehicle.setLaneChangeMode(s.vehicle_id,512);configured.add(s.vehicle_id)
            active=set(conn.vehicle.getIDList())
            for k,u in legacy.smooth_vehicle_speed_targets(t).items():
                if k in active:conn.vehicle.setSpeed(k,float(u))
            collisions.update(conn.simulation.getCollidingVehiclesIDList())
            if t< -1e-9:continue
            index=len(ids_by_cycle);ego,obs,missing=legacy.fixed_target_observations(states,'ego')
            if missing:raise RuntimeError('Missing target: '+str(missing))
            ids_by_cycle.append([x.vehicle_id for x in obs]);risk=model.compute_ttc_weights(obs)
            details={x.vehicle_id:(x,i) for i,x in enumerate(obs)}
            for s in states:
                traffic.append({'time_s':t,'sumo_time_s':st,**asdict(s)})
                if s.vehicle_id in details:
                    x,i=details[s.vehicle_id];d=x.distance_m;v=x.closing_speed_mps;az=x.azimuth_rad;ttc=risk.ttc_s[i];w=risk.omega[i]
                    targets.append({'cycle_index':index,'time_s':t,'slot':x.slot,'vehicle_id':s.vehicle_id,'distance_m':d,
                        'closing_speed_mps':v,'azimuth_rad':az,'TTC_s':ttc,'omega':w})
                else:d=v=az=0.;ttc=float('inf');w=0.
                combined.append({'cycle_index':index,'time_s':t,'vehicle_id':s.vehicle_id,'x':s.x_m,'y':s.y_m,'vx':s.vx_mps,'vy':s.vy_mps,
                    'distance_m':d,'closing_speed_mps':v,'azimuth_rad':az,'ttc_s':ttc,'risk_weight':w,'lane_id':s.lane_id})
    finally:conn.close()
    legacy.write_csv(RESULT/'traffic.csv',traffic);legacy.write_csv(RESULT/'traffic_targets.csv',targets);legacy.write_csv(RESULT/'traffic_states.csv',combined)
    checks={'cycles':len(ids_by_cycle),'stable_ids':len({tuple(x) for x in ids_by_cycle})==1,'collisions':sorted(collisions),'cutin_executed':False,
       'trajectory':compare_rows(traffic,read(HISTORY/'traffic.csv'),['time_s','vehicle_id'],['x_m','y_m','vx_mps','vy_mps','speed_mps'],1e-7),
       'physical_state':compare_rows(targets,read(HISTORY/'targets.csv'),['cycle_index','slot'],['distance_m','closing_speed_mps','azimuth_rad','TTC_s','omega'],1e-7)}
    checks['passed']=checks['cycles']==401 and checks['stable_ids'] and not collisions and checks['trajectory']['passed'] and checks['physical_state']['passed']
    dump(RESULT/'gate1_checks.json',checks)
    dump(RESULT/'traffic_config.json',{'simulation':asdict(sim),'fixed_ids':legacy.FIXED_IDS,'speed_reference_source_sha256':sha(ROOT/'legacy'/'main.py'),
        'configuration_xml_hashes':{p.name:sha(p) for p in (ROOT/'legacy'/'sumo').glob('*')},'maintenance_or_DNN_enabled':False})
    if not checks['passed']:raise RuntimeError('Gate1 failed; downstream gates blocked')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42})
    for filename,col,label in [('distance_vs_time','distance_m','Relative distance (m)'),('radial_velocity_vs_time','closing_speed_mps','Radial closing speed (m/s)'),('ttc_vs_time','TTC_s','TTC (s)'),('risk_weights_vs_time','omega','Physical TTC weight')]:
        fig,ax=plt.subplots(figsize=(6.25,5))
        for j,slot in enumerate(model.SLOT_ORDER):
            rs=[r for r in targets if r['slot']==slot];ax.plot([r['time_s'] for r in rs],[r[col] for r in rs],label=f'k{j+1}')
        ax.set(xlabel='SUMO time (s)',ylabel=label,xlim=(0,40));ax.legend();ax.grid(alpha=.25);legacy.save_figure(fig,RESULT/filename)
    fig,ax=plt.subplots(figsize=(6.25,5))
    for k in ['ego',*legacy.FIXED_IDS.values()]:
        rs=[r for r in traffic if r['vehicle_id']==k];ax.plot([r['time_s'] for r in rs],[r['x_m'] for r in rs],label=k.replace('target','k').replace('_front_same','').replace('_rear_same','').replace('_front_adj','').replace('_rear_adj',''))
    ax.set(xlabel='SUMO time (s)',ylabel='Longitudinal position (m)',xlim=(0,40));ax.legend();ax.grid(alpha=.25);legacy.save_figure(fig,RESULT/'traffic_trajectory')
    print('GATE1 PASS 401/401, trajectory and TTC agree with history',flush=True)

def gate2():
    if not json.loads((RESULT/'gate1_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Gate1 not passed')
    out=RESULT/'rerun'
    meta=legacy.run_experiment(out,SUMO,model.SimulationConfig(),model.RadarConfig(),model.SystemConfig())
    checks={}
    specs={
      'cycles.csv':(['cycle_index'],['weighted_total_crb','ordinary_total_crb','Ts_star_s','Ps_star_w','sensing_energy_j','time_upper_margin_s','energy_margin_j','qos_rate_margin_bps']),
      'targets.csv':(['cycle_index','slot'],['distance_m','closing_speed_mps','omega','dwell_s','scalar_CRB','weighted_scalar_CRB','tx_beam_gain_linear','rx_beam_gain_linear']),
      'fig7_multi_scenario_ao_trace.csv':(['scenario','ao_iteration'],['total_wcrb','Ts_ms','Ps_dbm']),
      'fig8a_time_wcrb.csv':(['cycle_index'],['ttc_total_wcrb','no_ttc_total_wcrb']),
      'fig8bc_physical_sensitivity.csv':(['sweep','point_index'],['x_value','ttc_total_wcrb','no_ttc_total_wcrb']),
      'fig9_ttc_aware_target_wcrb.csv':(['cycle_index','slot'],['omega','dwell_ms','scalar_CRB','target_wcrb'])}
    for file,(keys,cols) in specs.items():
        # Numerical acceptance fixed before the run; no fitted rescaling.
        checks[file]=compare_rows(read(out/file),read(HISTORY/file),keys,cols,1e-9,1e-7)
    a=read(out/'fig8a_time_wcrb.csv');ttc=np.mean([float(r['ttc_total_wcrb']) for r in a]);no=np.mean([float(r['no_ttc_total_wcrb']) for r in a])
    checks['summary']={'ttc_average_wcrb':ttc,'no_ttc_average_wcrb':no,'relative_improvement_percent':100*(no-ttc)/no,'cycles':len(a),
        'model':'legacy unknown-complex-amplitude-projected diagonal EFIM','paper_P1':'system AO','paper_P2':'target dwell','old_identifiers_unchanged':True}
    checks['passed']=all(c['passed'] for name,c in checks.items() if name!='summary') and meta['summary']['status']=='passed'
    dump(RESULT/'gate2_checks.json',checks)
    print(json.dumps({'gate2_passed':checks['passed'],**checks['summary']},ensure_ascii=False),flush=True)
    if not checks['passed']:raise RuntimeError('Gate2 failed; full-ML disabled')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['gate0','gate1','gate2','all'],default='all');args=parser.parse_args()
    if args.stage=='all':gate1();gate2()
    else:globals()[args.stage]()
