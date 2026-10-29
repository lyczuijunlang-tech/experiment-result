"""Independent equations-to-CSV, immutable-source and final-PDF audit."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,csv,json,hashlib,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('TTC-AO','RU-AO','DA-AO','TTC-PD','UD')
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill-scripts',type=Path,default=Path.home()/'.codex/skills/nature-figure/scripts')
    parser.add_argument('--record-inspected',action='store_true',help='Record manual review only after seeing the actual final PNG.')
    args=parser.parse_args()
    config=load('run_config.json');complete=load('completion.json')
    assert complete['all_checks_passed'] and not complete['source_model_modified']
    for name,expected in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/'source_snapshot'/name).read_bytes()).hexdigest()==expected
    traffic=rows('traffic.csv');warmup=rows('warmup_traffic.csv');detail=rows('resource_allocation_details.csv')
    cycles=rows('method_cycles.csv');wide=rows('fig8a_five_methods.csv');statistics=rows('method_statistics.csv')
    assert len(traffic)==2005 and len(detail)==8020 and len(cycles)==2005 and len(wide)==401
    state={(float(r['time_s']),r['vehicle_id']):r for r in traffic}
    full={(float(r['time_s']),r['vehicle_id']):r for r in warmup+traffic}
    targets={(int(r['cycle_index']),r['method']):[] for r in cycles}
    for r in detail:targets[int(r['cycle_index']),r['method']].append(r)
    errors={key:0. for key in ['distance_m','closing_speed_mps','backward_velocity_mps','TTC_s',
        'physical_weight','optimization_weight','scalar_crb','weighted_crb','sum_target_wcrb',
        'FIM_relative','dwell_sum_s','PD_formula_s','UD_formula_s','communication_rate_bps','gain_percent']}
    def update(key,value):errors[key]=max(errors[key],float(abs(value)))
    radar=config['radar'];system=config['system'];frame=system['frame_s'];floor=system['min_dwell_s']
    c=299792458.;kb=1.380649e-23
    n0=kb*radar['noise_temperature_k']*radar['noise_figure_linear']
    for row in cycles:
        index=int(row['cycle_index']);method=row['method'];t=float(row['time_s'])
        group=targets[index,method];assert len(group)==4
        current=[state[t,r['vehicle_id']] for r in group];ego=state[t,'ego']
        raw=[];distance=[]
        for r,s in zip(group,current):
            dx=float(s['x_m'])-float(ego['x_m']);dy=float(s['y_m'])-float(ego['y_m'])
            d=math.hypot(dx,dy)
            vx=float(s['vx_mps'])-float(ego['vx_mps']);vy=float(s['vy_mps'])-float(ego['vy_mps'])
            v=-(vx*dx+vy*dy)/d
            update('distance_m',d-float(r['distance_m']));update('closing_speed_mps',v-float(r['closing_speed_mps']))
            ttc=d/v if v>0 else float('inf')
            if math.isfinite(ttc):update('TTC_s',ttc-float(r['TTC_s']))
            raw.append(1/(ttc+.1));distance.append(d)
            earlier=full[round(t-.1,10),s['vehicle_id']]
            for axis in ('x','y'):
                update('backward_velocity_mps',(float(s[axis+'_m'])-float(earlier[axis+'_m']))/.1-float(s['v'+axis+'_mps']))
        physical=4*np.array(raw)/sum(raw) if sum(raw)>0 else np.ones(4)
        saved=np.array([float(r['physical_ttc_weight']) for r in group])
        update('physical_weight',np.max(abs(physical-saved)))
        if method=='RU-AO':opt=np.ones(4)
        elif method=='DA-AO':
            raw_da=np.array([1/(d+config['distance_regularizer_m']) if float(r['closing_speed_mps'])>0 else 0. for d,r in zip(distance,group)])
            opt=4*raw_da/raw_da.sum() if raw_da.sum()>0 else np.ones(4)
        else:opt=physical
        update('optimization_weight',np.max(abs(opt-np.array([float(r['optimization_weight']) for r in group]))))
        Ts=float(row['Ts_s']);Ps=float(row['Ps_w']);dwell=np.array([float(r['dwell_s']) for r in group])
        update('dwell_sum_s',dwell.sum()-Ts)
        assert dwell.min()>=floor-1e-12 and Ps>0 and Ps<=system['sensing_power_max_w']+1e-12
        assert Ts>=4*floor-1e-12 and Ps*Ts<=system['sensing_energy_max_j']+1e-10
        snr=float(row['communication_snr_linear'])
        minimum_comm=frame*system['minimum_rate_bps']/(system['communication_bandwidth_hz']*math.log2(1+snr))
        assert Ts<=frame-minimum_comm+1e-12
        rate=(frame-Ts)/frame*system['communication_bandwidth_hz']*math.log2(1+snr)
        update('communication_rate_bps',rate-float(row['communication_rate_bps']))
        if method=='TTC-PD':update('PD_formula_s',np.max(abs(dwell-(floor+(Ts-4*floor)*physical/physical.sum()))))
        if method=='UD':update('UD_formula_s',np.max(abs(dwell-Ts/4)))
        total=0.
        for r,d,tk,w in zip(group,distance,dwell,physical):
            # Independent analytic diagonal EFIM, same declared original model.
            # Matched ideal N-element beams have gain N; also check stored gains.
            assert abs(float(r['tx_gain'])-radar['tx_elements'])<1e-12
            assert abs(float(r['rx_gain'])-radar['rx_elements'])<1e-12
            Pr=Ps*radar['tx_elements']*radar['rx_elements']*(c/radar['carrier_hz'])**2*radar['rcs_m2']/((4*math.pi)**3*d**4*radar['system_loss_linear'])
            Jd=8*math.pi**2*(Pr/n0)*(2/c)**2*(radar['bandwidth_hz']**2/12)*tk
            Jv=8*math.pi**2*(Pr/n0)*(2*radar['carrier_hz']/c)**2*tk**3/12
            assert Jd>0 and Jv>0 and float(r['FIM_dv'])==0
            update('FIM_relative',(Jd-float(r['FIM_dd']))/Jd);update('FIM_relative',(Jv-float(r['FIM_vv']))/Jv)
            scalar=(1/Jd)/radar['crb_d_reference_m2']+(1/Jv)/radar['crb_v_reference_m2ps2']
            update('scalar_crb',scalar-float(r['scalar_crb']))
            contribution=w*scalar;total+=contribution
            update('weighted_crb',contribution-float(r['final_ttc_weighted_crb']))
        update('sum_target_wcrb',total-float(row['total_wcrb']))
        assert np.isfinite(total) and total>=0
    means={m:float(np.mean([float(r[m]) for r in wide])) for m in METHODS}
    for r in statistics:
        m=r['Method'];assert abs(means[m]-float(r['Mean wCRB']))<1e-15
        if m!='TTC-AO':update('gain_percent',100*(means[m]-means['TTC-AO'])/means[m]-float(r['Relative gain of TTC-AO (%)']))
    assert errors['distance_m']<1e-10 and errors['closing_speed_mps']<1e-10 and errors['backward_velocity_mps']<1e-10
    assert errors['TTC_s']<1e-9 and errors['physical_weight']<1e-12 and errors['optimization_weight']<1e-12
    assert errors['scalar_crb']<1e-12 and errors['weighted_crb']<1e-12 and errors['sum_target_wcrb']<1e-12
    assert errors['FIM_relative']<1e-12 and errors['dwell_sum_s']<1e-12 and errors['PD_formula_s']<1e-12 and errors['UD_formula_s']<1e-12
    assert errors['communication_rate_bps']<1e-5 and errors['gain_percent']<1e-10
    save('independent_formula_checks.json',{'passed':True,'method_cycles_checked':2005,'target_rows_checked':8020,
        'max_absolute_errors_or_labelled_relative_errors':errors,'source_snapshot_unchanged':True,
        'warmup_state_included_for_first_backward_velocity_check':True})
    sys.path.insert(0,str(args.skill_scripts))
    from audit_pdf_text import audit_pdf as font_audit
    from audit_figure_collisions import audit_pdf as collision_audit
    from validate_figure import validate_source,summarize
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    findings=validate_source((ROOT/'plot_figure.py').read_text(encoding='utf-8'),'python')
    source={'summary':summarize(findings,False),'findings':[asdict(f) for f in findings]}
    fonts=font_audit((ROOT/'fig8a_five_methods.pdf').read_bytes(),minimum_pt=5)
    collision=collision_audit(ROOT/'fig8a_five_methods.pdf')
    save('qa/source_validation.json',source);save('qa/fonts.json',fonts);save('qa/collisions.json',collision)
    summary={'collision_verdict':collision['verdict'],'minimum_font_pt':fonts['minimum_found_pt'],
        'below_5pt':fonts['below_minimum_count'],'source_ready':source['summary']['ready'],
        'visual_review':'PASS: actual final PNG inspected; all five raw curves, axes and external legend reviewed' if args.record_inspected else 'pending',
        'warning_review':'PNG/PDF/SVG requested, no TIFF requirement; 158.8mm follows established 5:4 layout.',
        'panel_alignment':'not applicable: single panel'}
    save('qa/summary.json',summary)
    assert source['summary']['ready'] and fonts['auditable'] and fonts['below_minimum_count']==0
    assert collision['verdict']=='PASS','Fix final rendered PDF collision findings before delivery'
    with (ROOT/'experiment_report.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 独立数学与图形验收\n\n')
        f.write('独立从实际SUMO二维位置/速度、物理常数和资源重建全部8020条目标级FIM/CRB、TTC权重、PD/UD公式及2005组系统约束；全部通过。误差见independent_formula_checks.json。\n\n')
        f.write(f"最终PDF碰撞检查{collision['verdict']}；最小字体{fonts['minimum_found_pt']}pt；视觉检查：{summary['visual_review']}。所有401个时间点保留。\n")
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
    save('artifact_manifest.json',{'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    print(json.dumps({'independent_formula_checks':errors,'figure':summary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
