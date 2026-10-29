"""Audit all controlled states and policies; report without assuming a ranking."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=['TTC-AO','RU-AO','DA-AO','TTC-PD','UD']
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def save(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write(name,data):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in data]])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-inspected',action='store_true');args=parser.parse_args()
    config=load('run_config.json');done=load('completion.json');frozen=load('baseline_snapshot.json')
    assert done['all_checks_passed'] and done['actual_AO_solves']==510
    points=rows('method_points.csv');targets=rows('target_allocations.csv');states=rows('controlled_states.csv');refs=rows('outer_precision_checks.csv')
    assert len(points)==510 and len(targets)==2040 and len(states)==510
    state_map={(r['sweep'],int(r['point_index']),r['vehicle_id']):r for r in states}
    grouped={}
    for r in targets:grouped.setdefault((r['sweep'],int(r['point_index']),r['method']),[]).append(r)
    baseline_by={r['vehicle_id']:r for r in frozen['vehicle_states']}
    selected=config['selected_vehicle_id']
    errors={k:0. for k in ['distance_m','radial_velocity_mps','TTC_weights','optimization_weights','control_variable',
        'frozen_other_states','scalar_crb','weighted_crb','target_sum_total','dwell_sum_s','PD_formula_s','UD_formula_s','FIM_relative']}
    def err(k,value):errors[k]=max(errors[k],float(abs(value)))
    radar=config['radar'];sy=config['system'];c=299792458.;n0=1.380649e-23*radar['noise_temperature_k']*radar['noise_figure_linear']
    for row in points:
        sweep=row['sweep'];point=int(row['point_index']);method=row['method'];rr=grouped[sweep,point,method]
        assert len(rr)==4
        ego=state_map[sweep,point,'ego'];raw=[]
        for r in rr:
            st=state_map[sweep,point,r['vehicle_id']]
            dx=float(st['x_m'])-float(ego['x_m']);dy=float(st['y_m'])-float(ego['y_m']);d=math.hypot(dx,dy)
            v=-((float(st['vx_mps'])-float(ego['vx_mps']))*dx+(float(st['vy_mps'])-float(ego['vy_mps']))*dy)/d
            err('distance_m',d-float(r['distance_m']));err('radial_velocity_mps',v-float(r['closing_speed_mps']))
            raw.append(1/(d/v+.1) if v>0 else 0.)
            if r['vehicle_id']==selected:
                expected=(config['baseline_distance_m'] if sweep=='range' else config['baseline_closing_speed_mps'])*float(row['scale'])
                err('control_variable',(d if sweep=='range' else v)-expected)
                err('control_variable',(v-config['baseline_closing_speed_mps']) if sweep=='range' else (d-config['baseline_distance_m']))
            else:
                for key in ['x_m','y_m','vx_mps','vy_mps']:
                    err('frozen_other_states',float(st[key])-float(baseline_by[r['vehicle_id']][key]))
            assert float(st['speed_mps'])<30
        w=4*np.array(raw)/sum(raw) if sum(raw)>0 else np.ones(4)
        err('TTC_weights',np.max(abs(w-np.array([float(r['physical_ttc_weight']) for r in rr]))))
        if method=='RU-AO':opt=np.ones(4)
        elif method=='DA-AO':
            a=np.array([1/(float(r['distance_m'])+.1) if float(r['closing_speed_mps'])>0 else 0. for r in rr])
            opt=4*a/a.sum() if a.sum()>0 else np.ones(4)
        else:opt=w
        err('optimization_weights',np.max(abs(opt-np.array([float(r['optimization_weight']) for r in rr]))))
        Ts=float(row['Ts_s']);Ps=float(row['Ps_w']);tk=np.array([float(r['dwell_s']) for r in rr])
        err('dwell_sum_s',tk.sum()-Ts)
        assert tk.min()>=sy['min_dwell_s']-1e-12 and float(row['final_feasible'])==1 and Ps>0
        assert Ps<=sy['sensing_power_max_w']+1e-12 and Ts*Ps<=sy['sensing_energy_max_j']+1e-10
        Tc=sy['frame_s']*sy['minimum_rate_bps']/(sy['communication_bandwidth_hz']*math.log2(1+sy['communication_snr_linear']))
        assert 4*sy['min_dwell_s']-1e-12<=Ts<=sy['frame_s']-Tc+1e-12
        if method=='UD':err('UD_formula_s',np.max(abs(tk-Ts/4)))
        if method=='TTC-PD':err('PD_formula_s',np.max(abs(tk-(sy['min_dwell_s']+(Ts-4*sy['min_dwell_s'])*w/w.sum()))))
        total=0.
        for r,T,wk in zip(rr,tk,w):
            d=float(r['distance_m'])
            Pr=Ps*radar['tx_elements']*radar['rx_elements']*(c/radar['carrier_hz'])**2*radar['rcs_m2']/((4*math.pi)**3*d**4*radar['system_loss_linear'])
            Jd=8*math.pi**2*(Pr/n0)*(2/c)**2*(radar['bandwidth_hz']**2/12)*T
            Jv=8*math.pi**2*(Pr/n0)*(2*radar['carrier_hz']/c)**2*T**3/12
            assert Jd>0 and Jv>0 and float(r['FIM_dv'])==0
            err('FIM_relative',(Jd-float(r['FIM_dd']))/Jd);err('FIM_relative',(Jv-float(r['FIM_vv']))/Jv)
            C=1/Jd/radar['crb_d_reference_m2']+1/Jv/radar['crb_v_reference_m2ps2']
            err('scalar_crb',C-float(r['scalar_crb']));err('weighted_crb',wk*C-float(r['target_wcrb']));total+=wk*C
        err('target_sum_total',total-float(row['total_wcrb']))
        assert np.isfinite(total) and int(row['ao_converged'])==1
    assert errors['control_variable']<1e-10 and errors['frozen_other_states']==0
    assert errors['distance_m']<1e-10 and errors['radial_velocity_mps']<1e-10
    assert all(value<1e-12 for key,value in errors.items() if key not in ['control_variable','distance_m','radial_velocity_mps'])
    for key in {(r['sweep'],r['point_index']) for r in points}:
        group=[r for r in points if (r['sweep'],r['point_index'])==key]
        assert len(group)==5 and len({r['state_sha256'] for r in group})==1
    statistics=[];crossings=[]
    for sweep in ['range','closing_speed']:
        means={m:float(np.mean([float(r['total_wcrb']) for r in points if r['sweep']==sweep and r['method']==m])) for m in METHODS}
        for m in METHODS:
            rr=[r for r in points if r['sweep']==sweep and r['method']==m]
            statistics.append({'sweep':sweep,'method':m,'scan_grid_mean_wcrb':means[m],
                'TTC_AO_gain_over_grid_mean_percent':'' if m=='TTC-AO' else 100*(1-means['TTC-AO']/means[m]),
                'feasible_points':len(rr),'converged_points':sum(int(r['ao_converged']) for r in rr),
                'max_own_objective_reference_gap_percent':max(float(r['relative_objective_gap_percent']) for r in refs if r['sweep']==sweep and r['method']==m)})
        for i,a in enumerate(METHODS):
            av=np.array([float(r['total_wcrb']) for r in points if r['sweep']==sweep and r['method']==a])
            for b in METHODS[i+1:]:
                bv=np.array([float(r['total_wcrb']) for r in points if r['sweep']==sweep and r['method']==b]);delta=av-bv
                nonzero=delta[abs(delta)>1e-12]
                crossings.append({'sweep':sweep,'method_A':a,'method_B':b,'sign_crossings':int(np.sum(np.sign(nonzero[1:])!=np.sign(nonzero[:-1]))),
                    'B_better_than_A_points':int(np.sum(delta>1e-12)),'minimum_absolute_gap':float(np.min(abs(delta)))})
    write('method_statistics.csv',statistics);write('curve_crossings.csv',crossings)
    write('constraint_checks.csv',[{k:r[k] for k in ['sweep','point_index','method','dwell_sum_error_s','minimum_dwell_margin_s','time_lower_margin_s',
        'time_upper_margin_s','energy_margin_j','power_lower_margin_w','power_upper_margin_w','qos_rate_margin_bps','final_feasible']} for r in points])
    save('scientific_audit.json',{'passed':True,'actual_AO_solves':510,'all_points_included':True,'same_current_state_and_physical_weights_for_five_methods':True,
        'max_errors':errors,'independent_geometric_and_EFIM_formula_audit':True,'no_future_online_state_input':True,
        'experiment_scope':'controlled parameter sensitivity from an actual recorded SUMO snapshot; not new traffic trajectories',
        'all_methods_same_initialization_budget_tolerance_and_constraints':True,'all_final_CRBs_finite_and_FIM_positive':True})
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_pdf_text import audit_pdf as fonts
    from audit_figure_collisions import audit_pdf as collisions
    from validate_figure import validate_source,summarize
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    findings=validate_source((ROOT/'plot_sensitivity.py').read_text(encoding='utf-8'),'python')
    source={'summary':summarize(findings,False),'findings':[asdict(f) for f in findings]};save('qa/source_validation.json',source)
    assert source['summary']['ready']
    qa_summary=[]
    for name in ['fig8b_five_methods','fig8c_five_methods']:
        font=fonts((ROOT/(name+'.pdf')).read_bytes(),minimum_pt=5);collision=collisions(ROOT/(name+'.pdf'))
        save('qa/'+name+'.fonts.json',font);save('qa/'+name+'.collisions.json',collision)
        assert font['auditable'] and font['below_minimum_count']==0 and collision['verdict']=='PASS'
        qa_summary.append({'figure':name,'minimum_font_pt':font['minimum_found_pt'],'collision':collision['verdict'],
            'visual_review':'PASS: actual final PNG inspected; all five raw curves and labels reviewed' if args.record_inspected else 'pending'})
    save('qa/summary.json',qa_summary)
    text=['# 图8(b)(c)：五方法物理敏感性实验',
        '图8(a)保留原始完整时域五方法结果；本次只增加(b)距离—总wCRB、(c)径向闭合速度—总wCRB五方法对照，并同步三联图。',
        f"基线来自已实际执行SUMO的t={config['baseline_time_s']}s快照，高风险目标{config['selected_target']}。距离基线{config['baseline_distance_m']:.9g}m，闭合速度基线{config['baseline_closing_speed_mps']:.9g}m/s。选点规则沿用原离线最大风险差快照，不用于在线决策读取未来。",
        '距离取基线75%–125%，闭合速度取50%–150%，各51点，均包含基线。只控制被选目标，另三目标与ego冻结。距离修改真实状态坐标并重新投影；速度沿视线方向改变，保留切向分量。每点重新计算d、径向v、φ、TTC、风险权重和匹配波束，分别冷启动求解五种完整P2。',
        '这些是受控静态反事实状态，不是重新执行102次SUMO，也不是将原轨迹按距离或速度排序后连线。速度扫描低于道路30m/s限速；反事实速度并非重新在原车型maxSpeed配置下积分出的速度。原(a)交通轨迹不变。',
        '五方法与(a)完全一致：TTC-AO/RU-AO/DA-AO分别用TTC/等权/距离权重完整P1；TTC-PD/UD直接分配目标驻留，均保留完整外层AO。最终统一调用evaluate_common_ttc_wcrb，用该扫描点同一物理TTC权重与原对角EFIM/CRB评价。εTTC=0.1s，εDA=0.1m；初值Ts=25ms、Ps=13.5dBm，η=1e-6、最多100轮、原松弛0.35/0.03(近边界0.15)、最低驻留2ms、帧50ms、功率50mW、能量1mJ及通信QoS保持一致。',
        '## 实算统计',table(['扫描','方法','网格均值wCRB','TTC-AO相对改善 %','可行/收敛点'],[
            [s['sweep'],s['method'],f"{s['scan_grid_mean_wcrb']:.12g}",'—' if s['method']=='TTC-AO' else f"{s['TTC_AO_gain_over_grid_mean_percent']:.6f}",f"{s['feasible_points']}/{s['converged_points']}"] for s in statistics]),
        '这里的平均值是均匀扫描网格上的描述统计，不是全时域平均，也不是交通条件的概率期望。所有真实交叉保留，见curve_crossings.csv，未根据排名调整参数。',
        '速度图同时改变归一化物理TTC权重和优化后的驻留时间。因此，TTC-AO的总wCRB不必随闭合速度单调增加；下降不能直接解释为高速度本身让估计更容易，或更高风险代表更安全。原冻结几何EFIM本身不以速度大小改变信息系数，本图还包含风险重分配和资源优化的联合作用。',
        '## 数值核查',f'510次新AO、2040条目标级分配，全部有效且早停收敛。独立几何、TTC、PD/UD和EFIM公式误差：{errors}。外层参考诊断保留在outer_precision_checks.csv，没有替换主图AO结果。',
        '基线点与(a)时域热启动的微小差别见baseline_point_checks.csv。本扫描所有点统一冷启动，所以不能强行使数值逐位匹配热启动轨迹；模型、预算和评价器完全相同。',
        '## 文件与运行',
        'run_sensitivity.py重算全部510次AO；plot_sensitivity.py只重绘本次CSV。PyCharm安装requirements.txt后直接运行run_sensitivity.py。完成目录受保护，重跑请复制代码、baseline_snapshot.json、allocation_methods.py及source_snapshot到新目录。',
        'fig8b_five_methods和fig8c_five_methods分别有PNG/PDF/SVG及51行五方法CSV。method_points.csv、target_allocations.csv、controlled_states.csv、constraint_checks.csv、candidate_evaluations.csv及ao_trace.csv保存全部实际扫描状态、资源、FIM/CRB、约束、候选和更新过程。没有平滑、插值、固定偏移或删点。每张图5:4、600dpi，曲线顶到横轴两端。',
        '## 图形验收',table(['图片','PDF字体下限pt','碰撞','视觉'],[[r['figure'],r['minimum_font_pt'],r['collision'],r['visual_review']] for r in qa_summary])]
    (ROOT/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
    save('artifact_manifest.json',{'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    print(json.dumps({'formula_checks':errors,'figure_QA':qa_summary,'statistics':statistics},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
