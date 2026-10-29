"""Fresh nominal SUMO episode; five independently optimized system policies."""
from pathlib import Path
from dataclasses import asdict,replace
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,sys,time,contextlib,traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np
import model
import main as original
from allocation_methods import (METHODS,LABELS,EPSILON_D_M,CurrentState,optimization_weights,
    evaluate_common_ttc_wcrb,make_ao_solver,outer_reference)

def jsave(path,obj):
    Path(path).write_text(json.dumps(original.json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))

class CSVStream:
    def __init__(self,path):self.file=Path(path).open('w',encoding='utf-8-sig',newline='');self.writer=None
    def append(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):self.file.close()

class Tee:
    def __init__(self,*files):self.files=files
    def write(self,value):
        for file in self.files:file.write(value);file.flush()
    def flush(self):
        for file in self.files:file.flush()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sumo-home',type=Path,default=Path(r'C:\Program Files (x86)\Eclipse\Sumo'))
    args=parser.parse_args()
    if (ROOT/'completion.json').exists():raise RuntimeError('Completed result protected. Copy this experiment to a new directory before rerunning.')
    import scipy,matplotlib
    sim=model.SimulationConfig();radar=model.RadarConfig();system=model.SystemConfig()
    original.validate_configs(sim,radar,system)
    started=datetime.now(timezone.utc).isoformat();begin=time.perf_counter()
    source_files=[p for p in (ROOT/'source_snapshot').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    jsave(ROOT/'run_config.json',{'started_at_utc':started,'simulation':asdict(sim),'radar':asdict(radar),'system':asdict(system),
        'methods':METHODS,'labels':LABELS,'distance_regularizer_m':EPSILON_D_M,
        'risk_regularizer_s':.1,'RU_weight':1.,'original_RU_weight':.25,
        'same_initialization_tolerance_budget_and_update_parameters_for_all':True,
        'same_final_evaluator':'evaluate_common_ttc_wcrb','common_final_weights':'Physical TTC weights with sum K',
        'DA_eligibility':'same v>0 branch as TTC; speed magnitude not used in DA priority',
        'P2_scope':'Each method independently optimizes Ts/Ps with its own previous-cycle initial resources',
        'PD_UD_inner_scope':'Direct target dwell policies, no optimal P1 solve; retain original AO outer engine',
        'random_traffic_perturbations':False,'robustness_shading':False,'nominal_episode_count':1,
        'source_sha256':{p.relative_to(ROOT/'source_snapshot').as_posix():sha(p) for p in source_files},
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
        'sumo_home':str(args.sumo_home),'sumo_binary_sha256':sha(args.sumo_home/'bin/sumo.exe'),
        'outer_precision_diagnostic':'All policies checked against model-scaling resource boundary; PD uses independent 81-point grid and bounded refinement. Diagnostic does not alter reported AO.',
        'zero_power_policy':'Original candidate endpoint Ps=0 has zero FIM/+infinite objective. Expected degeneracy logged separately; all final positive-power outputs must be finite.'})
    metrics={m:{'inner_evaluations':0,'complete_p1_solves':0,'direct_allocations':0,'expected_zero_power_endpoints':0,
        'max_dwell_sum_error_s':0.,'minimum_dwell_margin_s':float('inf'),'maximum_p1_kkt_residual':0.} for m in METHODS}
    previous_states={};previous_resources={m:None for m in METHODS};configured=set();collisions=set()
    traffic=[];warmup=[];method_rows=[];details=[];checks=[];wide=[];references=[];scaling_checks=[];states_saved=[]
    candidate_stream=CSVStream(ROOT/'candidate_evaluations.csv');trace_stream=CSVStream(ROOT/'ao_trace.csv')
    with (ROOT/'run_log.txt').open('w',encoding='utf-8') as logfile, contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
        print('Fresh five-method run:',started,flush=True)
        traci=original.load_matching_traci(args.sumo_home)
        command=[str(args.sumo_home/'bin/sumo.exe'),'-c',str(ROOT/'source_snapshot/sumo/p2_fixed.sumocfg'),
            '--step-length',str(sim.step_s),'--end',str(sim.warmup_s+sim.duration_s+sim.step_s),'--seed',str(sim.seed),
            '--collision.action','warn','--time-to-teleport','-1','--no-step-log','true','--duration-log.disable','true',
            '--log',str(ROOT/'sumo.log')]
        label=f'five_methods_{os.getpid()}'
        traci.start(command,label=label,numRetries=3);conn=traci.getConnection(label)
        jsave(ROOT/'sumo_execution.json',{'command':command,'version':list(conn.getVersion()),'PID':os.getpid()})
        try:
            while conn.simulation.getTime()<sim.warmup_s+sim.duration_s+sim.step_s-1e-9:
                conn.simulationStep();sumo_t=float(conn.simulation.getTime())
                t=round(sumo_t-sim.warmup_s-sim.step_s,10)
                states=original.vehicle_states_from_traci(conn,t,previous_states)
                previous_states={s.vehicle_id:s for s in states}
                for vehicle in states:
                    if vehicle.vehicle_id not in configured:
                        conn.vehicle.setSpeedMode(vehicle.vehicle_id,31);conn.vehicle.setLaneChangeMode(vehicle.vehicle_id,512)
                        configured.add(vehicle.vehicle_id)
                    row={'sumo_time_s':sumo_t,**asdict(vehicle)}
                    (traffic if t>=-1e-9 else warmup).append(row)
                active=set(conn.vehicle.getIDList())
                for vehicle,speed in original.smooth_vehicle_speed_targets(t).items():
                    if vehicle in active:conn.vehicle.setSpeed(vehicle,float(speed))
                collisions.update(conn.simulation.getCollidingVehiclesIDList())
                if t< -1e-9:continue
                cycle=len(wide)
                ego,observations,missing=original.fixed_target_observations(states,sim.ego_id)
                if missing:raise RuntimeError(f'Missing fixed targets: cycle {cycle}, {missing}')
                assert all(s.time_s==t for s in states)
                snr,rsu_distance=original.communication_snr_from_rsu(ego)
                sy=replace(system,communication_snr_linear=snr)
                risk=model.compute_ttc_weights(observations,epsilon_s=.1)
                beams=tuple(model.closed_form_beams(target.azimuth_rad,radar) for target in observations)
                state=CurrentState(tuple(observations),beams,radar,tuple(float(w) for w in risk.omega))
                state_hash=hashlib.sha256(json.dumps([asdict(s) for s in states],sort_keys=True).encode()).hexdigest()
                states_saved.append({'cycle_index':cycle,'time_s':t,'state_sha256':state_hash,'communication_snr_linear':snr,
                    'current_state_time_s':max(s.time_s for s in states),'physical_ttc_weights':list(state.physical_ttc_weights)})
                cycle_row={'cycle_index':cycle,'time_s':t}
                for method in METHODS:
                    weights=optimization_weights(method,state)
                    before=metrics[method]['inner_evaluations']
                    solver=make_ao_solver(method,state,metrics[method])
                    assert solver.__code__ is model.alternating_optimize_system.__code__
                    prior=previous_resources[method]
                    ao=solver(cycle,observations,beams,weights,radar,sy,
                        None if prior is None else prior[0],None if prior is None else prior[1])
                    final=ao.final;inner=final.p1
                    assert inner is not None and metrics[method]['inner_evaluations']-before==ao.p1_calls
                    previous_resources[method]=(final.total_sensing_s,final.sensing_power_w)
                    common=evaluate_common_ttc_wcrb(state,inner.dwell_s,final.sensing_power_w,state.physical_ttc_weights)
                    assert np.isfinite(common.total_wcrb) and common.total_wcrb>=0
                    margins=model.constraint_margins(final.total_sensing_s,final.sensing_power_w,4,sy)
                    feasible=min(margins.values())>=-sy.feasibility_tolerance and inner.minimum_dwell_margin_s>=-1e-12 and inner.dwell_sum_error_s<1e-12
                    assert feasible and ao.power_monotonic_theory and ao.power_monotonic_numeric
                    row={'cycle_index':cycle,'time_s':t,'method':method,'method_label':LABELS[method],'state_sha256':state_hash,
                        'total_wcrb':common.total_wcrb,'allocation_objective':final.optimizer_objective,
                        'Ts_s':final.total_sensing_s,'Ps_w':final.sensing_power_w,'Ps_dbm':model.dbm(final.sensing_power_w),
                        'energy_j':final.total_sensing_s*final.sensing_power_w,'ordinary_total_crb':float(common.scalar_crb.sum()),
                        'communication_snr_linear':snr,'communication_rate_bps':model.communication_rate_bps(final.total_sensing_s,sy),
                        'ao_rounds':ao.ao_rounds,'ao_converged':int(ao.converged),'ao_last_relative_change':ao.last_relative_change,
                        'inner_evaluations':ao.p1_calls,'complete_p1_solves':ao.p1_calls if method in METHODS[:3] else 0,
                        'direct_allocations':ao.p1_calls if method in METHODS[3:] else 0,'inner_status':inner.status,
                        'p1_kkt_residual':inner.kkt_residual if inner.kkt_residual is not None else '',
                        'dwell_sum_error_s':inner.dwell_sum_error_s,'minimum_dwell_margin_s':inner.minimum_dwell_margin_s,
                        'final_feasible':int(feasible),'power_monotonic_numeric':int(ao.power_monotonic_numeric)}
                    row.update({f'T{k+1}_s':float(v) for k,v in enumerate(inner.dwell_s)})
                    method_rows.append(row);cycle_row[method]=common.total_wcrb
                    check={'cycle_index':cycle,'time_s':t,'method':method,'state_sha256':state_hash,
                        'dwell_sum_error_s':inner.dwell_sum_error_s,'minimum_dwell_margin_s':inner.minimum_dwell_margin_s,
                        'final_fim_min_eigenvalue':min(float(np.linalg.eigvalsh(r.fim).min()) for r in common.crb_results),
                        'weight_sum_error':abs(sum(state.physical_ttc_weights)-4),
                        'target_sum_vs_total_error':abs(float(common.target_wcrb.sum())-common.total_wcrb),
                        'all_final_values_finite':int(np.isfinite(common.scalar_crb).all()),'feasible':int(feasible),**margins}
                    checks.append(check)
                    for k,(target,w,opt_w,crb,scalar,contribution,dwell) in enumerate(zip(observations,state.physical_ttc_weights,
                            weights,common.crb_results,common.scalar_crb,common.target_wcrb,inner.dwell_s)):
                        details.append({'cycle_index':cycle,'time_s':t,'method':method,'state_sha256':state_hash,
                            'target':f'k{k+1}','slot':target.slot,'vehicle_id':target.vehicle_id,'distance_m':target.distance_m,
                            'closing_speed_mps':target.closing_speed_mps,'azimuth_rad':target.azimuth_rad,'TTC_s':float(risk.ttc_s[k]),
                            'physical_ttc_weight':w,'optimization_weight':float(opt_w),
                            'Ts_s':final.total_sensing_s,'Ps_w':final.sensing_power_w,'dwell_s':float(dwell),
                            'CRB_d_m2':crb.crb_d_m2,'CRB_v_m2ps2':crb.crb_v_m2ps2,'scalar_crb':float(scalar),
                            'final_ttc_weighted_crb':float(contribution),'FIM_dd':float(crb.fim[0,0]),
                            'FIM_dv':float(crb.fim[0,1]),'FIM_vv':float(crb.fim[1,1]),
                            'tx_gain':beams[k].tx_gain_linear,'rx_gain':beams[k].rx_gain_linear})
                    for record in ao.candidate_records:
                        # Preserve expected singular endpoints as explicit text;
                        # blank means infeasible/not evaluated, not a zero score.
                        value=record['physical_weighted_crb']
                        objective=(value if np.isfinite(value) else 'infinity_expected_zero_power' if record['Ps_w']==0 and record['feasible'] else '')
                        candidate_stream.append({'cycle_index':cycle,'time_s':t,'method':method,
                            'candidate_index':record['candidate_index'],'ao_round':record['ao_round'],'stage':record['stage'],
                            'Ts_s':record['Ts_s'],'Ps_w':record['Ps_w'],'feasible':record['feasible'],
                            'inner_evaluated':record['p1_called'],'inner_is_optimal_P1':int(method in METHODS[:3]),
                            'allocation_objective':objective,'expected_zero_power':int(record['Ps_w']==0 and record['feasible']),
                            **{key:record[key] for key in margins}})
                    for record in ao.ao_trace:
                        trace_stream.append({'cycle_index':cycle,'time_s':t,'method':method,'ao_round':record['ao_iteration'],
                            'stage':record['stage'],'Ts_s':record['Ts_s'],'Ps_w':record['Ps_w'],
                            'allocation_objective':record['weighted_crb']})
                    reference=outer_reference(method,state,weights,sy)
                    references.append({'cycle_index':cycle,'time_s':t,'method':method,'ao_allocation_objective':final.optimizer_objective,
                        'ao_common_ttc_wcrb':common.total_wcrb,**reference,
                        'allocation_objective_gap_percent':100*(final.optimizer_objective/reference['reference_allocation_objective']-1)})
                    if method=='RU-AO' and cycle in (0,100,200,300,400):
                        # Diagnostic only; it never changes the chosen RU result.
                        scaled=model.alternating_optimize_system(cycle,observations,beams,np.full(4,.25),radar,sy,
                            None if prior is None else prior[0],None if prior is None else prior[1])
                        evaluated=evaluate_common_ttc_wcrb(state,scaled.final.p1.dwell_s,scaled.final.sensing_power_w,state.physical_ttc_weights)
                        scaling_checks.append({'cycle_index':cycle,'time_s':t,'max_dwell_difference_s':float(np.max(abs(scaled.final.p1.dwell_s-inner.dwell_s))),
                            'Ts_difference_s':abs(scaled.final.total_sensing_s-final.total_sensing_s),
                            'Ps_difference_w':abs(scaled.final.sensing_power_w-final.sensing_power_w),
                            'common_wcrb_difference':abs(evaluated.total_wcrb-common.total_wcrb),
                            'ao_round_difference':scaled.ao_rounds-ao.ao_rounds})
                wide.append(cycle_row)
                if cycle%25==0:
                    print(f'Fresh SUMO cycle {cycle:03d}, t={t:.1f}s; all five policies evaluated; inner evaluations={sum(m["inner_evaluations"] for m in metrics.values())}',flush=True)
                    jsave(ROOT/'progress.json',{'completed_cycles':len(wide),'time_s':t,'elapsed_s':time.perf_counter()-begin})
        except BaseException:
            traceback.print_exc();raise
        finally:
            conn.close();candidate_stream.close();trace_stream.close()
        assert len(wide)==401 and len(method_rows)==2005 and len(details)==8020 and not collisions
        for name,rows in [('fig8a_five_methods',wide),('method_cycles',method_rows),('resource_allocation_details',details),
            ('constraint_checks',checks),('traffic',traffic),('warmup_traffic',warmup),('outer_precision_checks',references),('RU_scale_checks',scaling_checks)]:
            original.write_csv(ROOT/(name+'.csv'),rows)
        jsave(ROOT/'current_state_records.json',states_saved)
        jsave(ROOT/'candidate_metrics.json',metrics)
        jsave(ROOT/'completion.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'wall_seconds':time.perf_counter()-begin,'cycles':401,'method_cycles':2005,'target_method_rows':8020,
            'all_checks_passed':True,'collisions':list(collisions),'future_state_used':False,
            'figure7_or9_modified':False,'source_model_modified':False})
        print('Actual five-method simulation completed. Summarizing and plotting.',flush=True)
    from summarize_results import summarize
    summarize()
    from plot_figure import draw
    draw()

if __name__=='__main__':main()
