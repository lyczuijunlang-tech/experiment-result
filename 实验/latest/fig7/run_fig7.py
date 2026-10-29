"""Run five traffic/initialization/relaxation sensitivity cases with original solver.

Python/PyCharm entry point. No SUMO episode is rerun: this is the independent
Figure 7 experiment on the recorded, actually executed SUMO snapshot.
"""
from pathlib import Path
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import contextlib
import csv
import hashlib
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
# This is a solver-configuration sensitivity experiment, not a comparison of
# five traffic conditions under an identical numerical solver configuration.
# All resource constraints, objective/FIM definitions and update formulas agree.
SETTINGS = {
    'scenario_1': (.039, .021, .08, .06),
    'scenario_2': (.032, .021, .05, .03),
    'scenario_3': (.035, .021, .08, .04),
    'scenario_4': (.034, .021, .06, .03),
    'scenario_5': (.033, .021, .04, .02),
}
BASELINE = ROOT.parent/'fig7_fig8_fig9_rerun'
SCENARIOS = [
    ('scenario_1', 'Scenario 1', .975, 1.015, 'Distance -2.5%; slower AO relaxation 0.08/0.06'),
    ('scenario_2', 'Scenario 2', 1.025, 1., 'Distance +2.5%; slower AO relaxation 0.05/0.03'),
    ('scenario_3', 'Scenario 3', 1.008, .985, 'Original mild perturbation'),
    ('scenario_4', 'Scenario 4', .95, 1., 'Distance -5%; closing speed unchanged'),
    ('scenario_5', 'Scenario 5', 1.05, 1., 'Distance +5%; closing speed unchanged'),
]

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_inputs():
    inputs = ROOT/'inputs'
    inputs.mkdir(exist_ok=True)
    if not (ROOT/'source_snapshot').exists():
        shutil.copytree(BASELINE/'source_snapshot', ROOT/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    if not (inputs/'snapshot.json').exists():
        previous = read_csv(BASELINE/'fig7_convergence.csv')
        sample = previous[0]
        cycle, timestamp = int(sample['cycle_index']), float(sample['time_s'])
        states = [r for r in read_csv(BASELINE/'raw_run/traffic.csv') if float(r['time_s']) == timestamp]
        scopes = json.loads((BASELINE/'cycle_and_scenario_parameters.json').read_text(encoding='utf-8'))
        scope = next(s for s in scopes if s['scope']=='episode' and int(s['cycle_index'])==cycle)
        parameters = json.loads((BASELINE/'parameters_and_solver.json').read_text(encoding='utf-8'))
        frozen = {'cycle_index':cycle, 'time_s':timestamp, 'states':states,
                  'base_system':scope['system'], 'radar':parameters['radar'],
                  'source':'Actual preceding SUMO run; same offline snapshot as original Figure 7',
                  'baseline_traffic_sha256':digest(BASELINE/'raw_run/traffic.csv'),
                  'source_hashes':{n:digest(ROOT/'source_snapshot'/n) for n in ['main.py','model.py']}}
        write_json(inputs/'snapshot.json', frozen)
        shutil.copy2(BASELINE/'fig7_convergence.csv', inputs/'original_three_scenarios.csv')

class Tee:
    def __init__(self, *files): self.files=files
    def write(self, value):
        for file in self.files: file.write(value); file.flush()
    def flush(self):
        for file in self.files: file.flush()

def main():
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    prepare_inputs()
    sys.path.insert(0, str(ROOT/'source_snapshot'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import main as original
    import model

    frozen=json.loads((ROOT/'inputs/snapshot.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT/'source_snapshot'/n)==h for n,h in frozen['source_hashes'].items())
    state_map={}
    strings={'vehicle_id','road_id','lane_id','velocity_source'}
    for row in frozen['states']:
        values={field.name:(row[field.name] if field.name in strings else int(row[field.name]) if field.name=='lane_index' else float(row[field.name])) for field in fields(model.VehicleState)}
        state_map[row['vehicle_id']]=model.VehicleState(**values)
    observations=tuple(model.relative_observation(slot, state_map['ego'], state_map[original.FIXED_IDS[slot]]) for slot in model.SLOT_ORDER)
    base_system=model.SystemConfig(**frozen['base_system'])
    system=replace(base_system, sensing_energy_max_j=.001, sensing_power_max_w=.05,
                   minimum_rate_bps=10e6, initial_sensing_time_s=.037, initial_sensing_power_w=.021,
                   ao_time_relaxation=.08, ao_power_relaxation=.04, ao_power_relaxation_near_time_bound=.04,
                   ao_eta=0., ao_max_rounds=240)
    radar=model.RadarConfig(**frozen['radar'])
    write_json(ROOT/'parameters.json', {'started_at_utc':started, 'system':asdict(system), 'radar':asdict(radar),
                'scenarios':[dict(zip(['id','label','distance_scale','closing_speed_scale','meaning'],s)) for s in SCENARIOS],
                'scenario_solver_settings': {key:dict(zip(['initial_Ts_s','initial_Ps_w','time_relaxation','power_relaxation'],value)) for key,value in SETTINGS.items()},
                'comparison':'Joint traffic and AO initialization/relaxation configuration sensitivity; not a traffic-only or algorithm-performance comparison.',
                'snapshot_cycle':frozen['cycle_index'], 'snapshot_time_s':frozen['time_s'], 'source_hashes':frozen['source_hashes']})
    all_rows=[]; updates=[]; summaries=[]; physical=[]; counter=0; scope=''
    original_p1=model.solve_p1
    with (ROOT/'run.log').open('w', encoding='utf-8') as logfile, (ROOT/'p1_call_audit.csv').open('w', encoding='utf-8-sig',newline='') as pfile, contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
        writer=None
        def audited_p1(targets, beams, weights, Ts, Ps, Tmin, radar):
            nonlocal counter, writer
            result=original_p1(targets, beams, weights, Ts, Ps, Tmin, radar)
            counter+=1
            matrices=np.asarray([r.fim for r in result.crb_results])
            eigen=float(np.linalg.eigvalsh(matrices).min())
            positive=Ps>0
            assert np.isfinite(matrices).all()
            assert not positive or (eigen>0 and np.isfinite(result.per_target_scalar_crb).all())
            error=abs(float(np.sum(result.dwell_s))-Ts)
            floor=float(np.min(result.dwell_s)-Tmin)
            assert error<1e-12 and floor>=-1e-12
            row={'call':counter,'scenario':scope,'Ts_s':Ts,'Ps_w':Ps,'weighted_crb':result.weighted_total_crb,
                 'sum_dwell_error_s':error,'min_dwell_margin_s':floor,'min_FIM_eigenvalue':eigen,
                 'expected_zero_power_degeneracy':int(not positive),'kkt_residual':result.kkt_residual}
            if writer is None: writer=csv.DictWriter(pfile, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            return result
        model.solve_p1=audited_p1
        print('Start:',started, '; independent five-scenario AO, original solver and EFIM')
        for order,(scope,label,ds,vs,meaning) in enumerate(SCENARIOS):
            ts0,ps0,alpha_t,alpha_p=SETTINGS[scope]
            system=replace(system,initial_sensing_time_s=ts0,initial_sensing_power_w=ps0,
                           ao_time_relaxation=alpha_t,ao_power_relaxation=alpha_p,
                           ao_power_relaxation_near_time_bound=alpha_p)
            # Same observation perturbation rule as the original independent
            # Figure 7. These are controlled LOS snapshots, not five new SUMO runs.
            targets=tuple(replace(t,distance_m=t.distance_m*ds,closing_speed_mps=t.closing_speed_mps*vs) for t in observations)
            risk=model.compute_ttc_weights(targets, epsilon_s=.1)
            weights=risk.omega
            beams=tuple(model.closed_form_beams(t.azimuth_rad,radar) for t in targets)
            initial=model.constraint_margins(system.initial_sensing_time_s,system.initial_sensing_power_w,4,system)
            assert min(initial.values())>=-system.feasibility_tolerance
            before=counter
            ao=model.alternating_optimize_system(frozen['cycle_index'],targets,beams,weights,radar,system)
            assert counter-before==ao.p1_calls
            rr=original.round_end_rows(list(ao.ao_trace))
            assert len(rr)==241 and ao.ao_rounds==240
            relative=abs(rr[-1]['weighted_crb']-rr[-2]['weighted_crb'])/abs(rr[-2]['weighted_crb'])
            assert relative<1e-6
            assert abs(float(np.sum(weights))-4)<1e-12
            for row in ao.ao_trace:
                margins=model.constraint_margins(float(row['Ts_s']),float(row['Ps_w']),4,system)
                assert min(margins.values())>=-system.feasibility_tolerance
                updates.append({'scenario':scope,**row,**margins})
            values=np.array([r['weighted_crb'] for r in rr],dtype=float)
            relative_steps=np.abs(np.diff(values))/np.abs(values[:-1])
            tail_max=np.maximum.accumulate(relative_steps[::-1])[::-1]
            stable_round=int(np.flatnonzero(tail_max<1e-6)[0])+1
            assert np.all(np.isfinite(values)) and np.all(np.diff(values)<=1e-12)
            final=ao.final
            independent=sum(float(w)*model.fim_and_crb(t,b,float(dwell),final.sensing_power_w,radar).scalar_crb for t,b,w,dwell in zip(targets,beams,weights,final.p1.dwell_s))
            assert abs(independent-final.physical_objective)<1e-12
            for row in rr:
                all_rows.append({'scenario_order':order,'scenario':scope,'scenario_label':label,'cycle_index':frozen['cycle_index'],
                    'time_s':frozen['time_s'],'ao_iteration':int(row['ao_iteration']),'total_wcrb':float(row['weighted_crb']),
                    'Ts_ms':1000*float(row['Ts_s']),'Ps_dbm':model.dbm(float(row['Ps_w'])),
                    'distance_scale':ds,'closing_speed_scale':vs,'initial_Ts_ms':1000*ts0,'initial_Ps_mW':1000*ps0,
                    'time_relaxation':alpha_t,'power_relaxation':alpha_p,'stability_round':stable_round,'ao_rounds':ao.ao_rounds,'p1_calls':ao.p1_calls,
                    'final_relative_objective_change':relative,'final_Ts_ms':1000*final.total_sensing_s,'final_Ps_mW':1000*final.sensing_power_w})
            for k,(target,weight) in enumerate(zip(targets,weights)):
                physical.append({'scenario':scope,'target':f'k{k+1}','distance_m':target.distance_m,
                    'closing_speed_mps':target.closing_speed_mps,'azimuth_rad':target.azimuth_rad,
                    'TTC_s':target.distance_m/target.closing_speed_mps if target.closing_speed_mps>0 else 'inf','omega':float(weight)})
            summaries.append({'scenario':scope,'meaning':meaning,'initial_wcrb':float(values[0]),'final_wcrb':float(values[-1]),
                'stability_round':stable_round,'initial_Ts_ms':1000*ts0,'initial_Ps_mW':1000*ps0,
                'time_relaxation':alpha_t,'power_relaxation':alpha_p,'reduction_percent':100*(1-values[-1]/values[0]),'last_relative_change':relative,'p1_calls':ao.p1_calls,
                'minimum_trace_constraint_margin':min(min(model.constraint_margins(float(r['Ts_s']),float(r['Ps_w']),4,system).values()) for r in ao.ao_trace),
                'max_final_wcrb_recompute_error':abs(independent-final.physical_objective),
                'solver_early_stop_flag':ao.converged,'fixed_budget_stability_pass':bool(relative<1e-6),
                'power_monotonic_theory':ao.power_monotonic_theory,'power_monotonic_numeric':ao.power_monotonic_numeric})
            print(label,meaning,'final wCRB=',values[-1],'relative change=',relative,'P1 calls=',ao.p1_calls,'sustained stability round=',stable_round)
        model.solve_p1=original_p1
    write_csv(ROOT/'fig7_convergence.csv',all_rows)
    write_csv(ROOT/'all_ao_updates.csv',updates)
    write_csv(ROOT/'scenario_target_parameters.csv',physical)
    # Relative convergence speeds are measured, never enforced by editing data.
    previous=read_csv(ROOT/'inputs/previous_five_scenarios.csv')
    unchanged={}
    for scenario in ['scenario_3','scenario_4','scenario_5']:
        old=[r for r in previous if r['scenario']==scenario]
        new=[r for r in all_rows if r['scenario']==scenario]
        assert len(old)==len(new)==241
        error=max(abs(float(a[key])-float(b[key])) for a,b in zip(old,new) for key in ['total_wcrb','Ts_ms','Ps_dbm'])
        assert error<1e-12
        unchanged[scenario]=error
    reference=summaries[1]['final_wcrb']
    for row in summaries: row['final_change_vs_scenario2_percent']=100*(row['final_wcrb']/reference-1)
    settings_rows=[{'scenario':row['scenario'],'distance_scale':SCENARIOS[i][2],
                   'closing_speed_scale':SCENARIOS[i][3],
                   **{key:row[key] for key in ['initial_Ts_ms','initial_Ps_mW','time_relaxation',
                       'power_relaxation','stability_round','initial_wcrb','final_wcrb','p1_calls']}}
                  for i,row in enumerate(summaries)]
    write_csv(ROOT/'scenario_settings.csv',settings_rows)
    write_json(ROOT/'constraint_checks.json',{'passed':True,'all_five_scenarios':summaries,'total_p1_calls':counter,
        'experiment_type':'Joint scenario and solver-configuration sensitivity',
        'unchanged_scenarios_max_absolute_difference':unchanged,
        'stability_criterion':'First round after which all subsequent round-end relative changes stay below 1e-6','positive_power_FIM_and_CRB_finite':True,
        'minimum_dwell_and_sum_constraint_pass':True,'zero_power_note':'Ps=0 endpoints have zero FIM and infinite CRB, as in the original solver.',
        'global_optimality_claim':False})
    # One single-panel quantitative comparison; no smoothing/interpolation,
    # row exclusion, or per-curve normalization. Every round is plotted.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
        'axes.labelsize':11,'legend.fontsize':8.5,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
        'pdf.fonttype':42,'svg.fonttype':'none'})
    styles=[('#0072B2','-','o'),('#D55E00','--','s'),('#009E73','-.','^'),('#CC79A7',':','D'),('#5B5B5B',(0,(5,1,1,1)),'v')]
    fig,ax=plt.subplots(figsize=(6.25,5.0))
    for order,(_,label,_,_,_) in enumerate(SCENARIOS):
        rows=[r for r in all_rows if r['scenario_order']==order]
        color,style,marker=styles[order]
        ax.plot([r['ao_iteration'] for r in rows],[r['total_wcrb'] for r in rows],label=label,
            color=color,linestyle=style,linewidth=1.6,marker=marker,markevery=24,markersize=3.,markerfacecolor='white',markeredgewidth=.7)
    ax.set(xlim=(0,240),xlabel='Alternating-optimization iteration',ylabel='Total TTC-weighted wCRB')
    ax.set_xticks(np.arange(0,241,40))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    ax.grid(color='.88',linewidth=.45)
    handles,labels=ax.get_legend_handles_labels()
    legend_order=[0,3,1,4,2]  # Row reading order 1,2,3 then 4,5.
    ax.legend([handles[i] for i in legend_order],[labels[i] for i in legend_order],loc='lower center',bbox_to_anchor=(.5,1.09),ncol=3,frameon=False,columnspacing=1.4,handlelength=2.8)
    fig.tight_layout();fig.canvas.draw()
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    write_json(qa/'layout.json',{'panel_count':1,'alignment_gate':'not applicable one panel','canvas_inches':[6.25,5.],
        'axes_position':list(ax.get_position().bounds),'x_limits':list(ax.get_xlim()),'y_limits':list(ax.get_ylim()),
        'data_rows':len(all_rows),'excluded_rows':0,'smoothing':False})
    fig.savefig(ROOT/'fig7_convergence.png',dpi=600)
    fig.savefig(ROOT/'fig7_convergence.pdf')
    fig.savefig(ROOT/'fig7_convergence.svg')
    plt.close(fig)
    table='\n'.join(f"| {SCENARIOS[i][1]} | {SCENARIOS[i][2]:.3f} | {SCENARIOS[i][3]:.3f} | {r['initial_Ts_ms']:.0f} | {r['initial_Ps_mW']:.0f} | {r['time_relaxation']:.2f} | {r['power_relaxation']:.2f} | {r['stability_round']} | {r['final_wcrb']:.10g} |" for i,r in enumerate(summaries))
    readme='''# 图7：五场景及求解配置敏感性对照

本次按用户要求减缓蓝色和橙色曲线下降：只将场景1时间/功率松弛系数从0.24/0.20减为0.08/0.06，场景2从0.12/0.06减为0.05/0.03。五组的交通状态、初始化、物理评价、资源约束和240轮预算保持不变；场景3、4、5逐点与上一版核对。没有平移横轴、平滑或手工修改曲线值。它是**场景与求解配置的联合敏感性实验**，不能将不同下降速度全部归因于交通状态，也不是五种算法的公平性能排名。

| 曲线 | 距离比例 | 闭合速度比例 | Ts初值 ms | Ps初值 mW | 时间松弛 | 功率松弛 | 持续稳定首轮 | 最终wCRB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
'''+table+'''

五组都使用同一真实SUMO t=40秒基准快照，依原图7规则作受控视距扰动；并非重新运行五次SUMO。TTC、权重、匹配波束和P1/P2均重新计算。原对角EFIM、CRB定义、通信QoS、1mJ能量和50mW功率上限、2ms最低驻留、AO更新公式及块候选检查不变。

全部曲线从第0轮开始真实下降，没有设置等待轮次、横轴平移、人工分段、平滑或插值。不同松弛系数对应不同更新步幅，因此进入稳定区先后有别。曲线在图中使用同一物理wCRB纵轴；未逐曲线归一化。

为给最保守的配置足够收敛预算，统一由140轮延长为240轮，eta=0关闭早停。表中的持续稳定首轮是首次达到“本轮及之后所有轮次相对目标变化均小于1e-6”的轮次，与原求解器的早停标志分开记录。稳定不等于全局最优，能量耦合下仍存在坐标锁定可能；不同步幅还可能到达略有差异的稳定点。

`fig7_convergence.csv`包含1205条原始轮次数据；`all_ao_updates.csv`包括分块资源更新和约束裕量；`p1_call_audit.csv`逐次记录完整P1；`scenario_target_parameters.csv`记录距离、闭合速度、TTC、风险权重；`parameters.json`记录逐场景初始化和松弛系数；`constraint_checks.json`记录真实持续稳定轮次和所有数值检查。

PyCharm安装`requirements.txt`后直接运行`run_fig7.py`即可重算。已冻结的`inputs/`与`source_snapshot/`足以独立运行，无需再次启动SUMO。运行只覆盖本目录输出，之前的五场景同设置对照和图8/9保持原样。

导出5:4、600dpi PNG、PDF及SVG。所有241个轮次点均绘制；仅标记符号稀疏显示。字体和排版检查记录于`qa/`。
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    write_json(ROOT/'completion.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'wall_seconds':time.perf_counter()-timer,'scenario_count':5,'trace_rows':len(all_rows),'p1_calls':counter,
        'checks_passed':True,'original_three_unchanged':False,'change_reason':'Only AO relaxation of blue/orange changed; traffic states, initialization, constraints and scenarios 3/4/5 unchanged','figure8_9_modified':False})
    print('Five-scenario figure and audits written:',ROOT)

if __name__=='__main__':
    main()
