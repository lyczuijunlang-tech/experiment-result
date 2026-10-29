"""Fresh five-policy AO solves on two controlled physical-state sweeps.

An actual recorded SUMO snapshot is the reference. Counterfactual sweep points
are explicitly labelled controlled states, not new SUMO trajectory samples.
"""
from pathlib import Path
from dataclasses import fields,replace,asdict
from datetime import datetime,timezone
import contextlib,csv,hashlib,json,math,sys,time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np
import model
import main as original
from allocation_methods import (METHODS,LABELS,EPSILON_D_M,CurrentState,optimization_weights,
    make_ao_solver,evaluate_common_ttc_wcrb,outer_reference)

def save(path,value):Path(path).write_text(json.dumps(original.json_safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):original.write_csv(ROOT/name,rows)
class Stream:
    def __init__(self,path):self.file=Path(path).open('w',encoding='utf-8-sig',newline='');self.writer=None
    def add(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):self.file.close()
class Tee:
    def __init__(self,*files):self.files=files
    def write(self,t):
        for f in self.files:f.write(t);f.flush()
    def flush(self):
        for f in self.files:f.flush()

def baseline():
    frozen=json.loads((ROOT/'baseline_snapshot.json').read_text(encoding='utf-8'))
    strings={'vehicle_id','road_id','lane_id','velocity_source'};states={}
    for r in frozen['vehicle_states']:
        values={f.name:(r[f.name] if f.name in strings else int(r[f.name]) if f.name=='lane_index' else float(r[f.name])) for f in fields(model.VehicleState)}
        states[r['vehicle_id']]=model.VehicleState(**values)
    return frozen,states

def controlled_states(base,selected_id,sweep,scale):
    states=dict(base);ego=base['ego'];vehicle=base[selected_id]
    vector=np.array([vehicle.x_m-ego.x_m,vehicle.y_m-ego.y_m]);los=vector/np.linalg.norm(vector)
    if sweep=='range':
        x,y=np.array([ego.x_m,ego.y_m])+scale*vector
        dx=float(x)-vehicle.x_m;dy=float(y)-vehicle.y_m
        vehicle=replace(vehicle,x_m=float(x),y_m=float(y),
            lane_position_m=vehicle.lane_position_m+dx*math.cos(vehicle.heading_rad)+dy*math.sin(vehicle.heading_rad),
            lateral_lane_position_m=vehicle.lateral_lane_position_m-dx*math.sin(vehicle.heading_rad)+dy*math.cos(vehicle.heading_rad))
    else:
        old_v=-float(np.dot(np.array([vehicle.vx_mps-ego.vx_mps,vehicle.vy_mps-ego.vy_mps]),los))
        velocity=np.array([vehicle.vx_mps,vehicle.vy_mps])-(scale-1)*old_v*los
        heading=math.atan2(float(velocity[1]),float(velocity[0]))
        vehicle=replace(vehicle,vx_mps=float(velocity[0]),vy_mps=float(velocity[1]),speed_mps=float(np.linalg.norm(velocity)),
            heading_rad=heading,sumo_angle_deg=(90-math.degrees(heading))%360,velocity_source='controlled_static_radial_velocity_sweep')
    states[selected_id]=vehicle
    assert vehicle.speed_mps<=30.0 and np.linalg.norm([vehicle.x_m-ego.x_m,vehicle.y_m-ego.y_m])>vehicle.length_m+2.5
    return states

def main():
    if (ROOT/'completion.json').exists():raise RuntimeError('Completed experiment protected; copy code, inputs and source_snapshot to a new directory to rerun.')
    frozen,base=baseline();radar=model.RadarConfig(**frozen['radar']);system=model.SystemConfig(**frozen['system'])
    obs=tuple(model.relative_observation(slot,base['ego'],base[original.FIXED_IDS[slot]]) for slot in model.SLOT_ORDER)
    base_risk=model.compute_ttc_weights(obs,epsilon_s=.1)
    selected_index=int(np.argmax(base_risk.omega));selected=obs[selected_index];selected_id=selected.vehicle_id
    sweeps=[('range',np.linspace(.75,1.25,51)),('closing_speed',np.linspace(.5,1.5,51))]
    started=datetime.now(timezone.utc).isoformat();begin=time.perf_counter()
    save(ROOT/'run_config.json',{'started_at_utc':started,'methods':METHODS,'labels':LABELS,'system':asdict(system),'radar':asdict(radar),
        'selected_target':f'k{selected_index+1}','selected_vehicle_id':selected_id,'baseline_time_s':frozen['time_s'],
        'baseline_distance_m':selected.distance_m,'baseline_closing_speed_mps':selected.closing_speed_mps,
        'range_scales':list(sweeps[0][1]),'closing_speed_scales':list(sweeps[1][1]),'points_per_sweep':51,
        'cold_start_at_every_point':True,'warm_start_between_points':False,'same_solver_parameters_for_every_method':True,
        'epsilon_TTC_s':.1,'epsilon_DA_m':EPSILON_D_M,'state_sweep':'One selected target varies; ego and three other targets frozen. Range changes actual state geometry; velocity changes only radial component.',
        'not_new_SUMO_trajectories':True,'road_speed_limit_mps':30.,'no_trajectory_or_FIM_or_evaluator_tuning':True,
        'source_hashes':{n:hashlib.sha256((ROOT/'source_snapshot'/n).read_bytes()).hexdigest() for n in ['main.py','model.py']},
        'allocation_methods_sha256':hashlib.sha256((ROOT/'allocation_methods.py').read_bytes()).hexdigest()})
    metrics={m:{'inner_evaluations':0,'complete_p1_solves':0,'direct_allocations':0,'expected_zero_power_endpoints':0,
        'max_dwell_sum_error_s':0.,'minimum_dwell_margin_s':float('inf'),'maximum_p1_kkt_residual':0.} for m in METHODS}
    results=[];targets=[];physical=[];references=[];baseline_checks=[];wide={s:[] for s,_ in sweeps}
    candidates=Stream(ROOT/'candidate_evaluations.csv');trace=Stream(ROOT/'ao_trace.csv')
    with (ROOT/'run_log.txt').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)):
        print('Start actual controlled five-method sensitivity solves:',started,flush=True)
        try:
            for sweep,scales in sweeps:
                for point,scale in enumerate(scales):
                    states=controlled_states(base,selected_id,sweep,float(scale));ego=states['ego']
                    observations=tuple(model.relative_observation(slot,ego,states[original.FIXED_IDS[slot]]) for slot in model.SLOT_ORDER)
                    risk=model.compute_ttc_weights(observations,epsilon_s=.1)
                    beams=tuple(model.closed_form_beams(t.azimuth_rad,radar) for t in observations)
                    state=CurrentState(observations,beams,radar,tuple(float(w) for w in risk.omega))
                    x=observations[selected_index].distance_m if sweep=='range' else observations[selected_index].closing_speed_mps
                    probe=(800000 if sweep=='range' else 900000)+point
                    state_sha=hashlib.sha256(json.dumps([asdict(s) for s in states.values()],sort_keys=True).encode()).hexdigest()
                    for vehicle in states.values():physical.append({'sweep':sweep,'point_index':point,'probe_id':probe,'scale':float(scale),
                        'source_kind':'controlled_static_sensitivity_state','is_exact_baseline':int(point==25),**asdict(vehicle)})
                    wrow={'sweep':sweep,'point_index':point,'scale':float(scale),'x_value':x,'x_unit':'m' if sweep=='range' else 'm/s','selected_target':f'k{selected_index+1}'}
                    for method in METHODS:
                        weights=optimization_weights(method,state);before=metrics[method]['inner_evaluations']
                        solver=make_ao_solver(method,state,metrics[method])
                        assert solver.__code__ is model.alternating_optimize_system.__code__
                        ao=solver(probe,observations,beams,weights,radar,system)
                        assert ao.p1_calls==metrics[method]['inner_evaluations']-before
                        final=ao.final;inner=final.p1
                        common=evaluate_common_ttc_wcrb(state,inner.dwell_s,final.sensing_power_w,state.physical_ttc_weights)
                        margins=model.constraint_margins(final.total_sensing_s,final.sensing_power_w,4,system)
                        assert min(margins.values())>=-system.feasibility_tolerance and inner.minimum_dwell_margin_s>=-1e-12
                        assert inner.dwell_sum_error_s<1e-12 and np.isfinite(common.total_wcrb)
                        assert ao.converged and ao.power_monotonic_numeric and ao.power_monotonic_theory
                        base_row={'sweep':sweep,'point_index':point,'probe_id':probe,'scale':float(scale),'x_value':x,'method':method,'state_sha256':state_sha}
                        row={**base_row,'total_wcrb':common.total_wcrb,'allocation_objective':final.optimizer_objective,
                            'Ts_s':final.total_sensing_s,'Ps_w':final.sensing_power_w,'ao_rounds':ao.ao_rounds,'ao_converged':int(ao.converged),
                            'ao_relative_change':ao.last_relative_change,'inner_evaluations':ao.p1_calls,
                            'complete_P1_solves':ao.p1_calls if method in METHODS[:3] else 0,
                            'direct_allocations':ao.p1_calls if method in METHODS[3:] else 0,
                            'dwell_sum_error_s':inner.dwell_sum_error_s,'minimum_dwell_margin_s':inner.minimum_dwell_margin_s,
                            'final_feasible':1,**margins}
                        row.update({f'T{k+1}_s':float(v) for k,v in enumerate(inner.dwell_s)})
                        results.append(row);wrow[method]=common.total_wcrb
                        for k,(target,w,optw,tk,crb,C,W) in enumerate(zip(observations,state.physical_ttc_weights,weights,inner.dwell_s,
                            common.crb_results,common.scalar_crb,common.target_wcrb)):
                            targets.append({**base_row,'target':f'k{k+1}','slot':target.slot,'vehicle_id':target.vehicle_id,
                                'distance_m':target.distance_m,'closing_speed_mps':target.closing_speed_mps,'azimuth_rad':target.azimuth_rad,
                                'TTC_s':float(risk.ttc_s[k]),'physical_ttc_weight':w,'optimization_weight':float(optw),
                                'Ts_s':final.total_sensing_s,'Ps_w':final.sensing_power_w,'dwell_s':float(tk),
                                'CRB_d_m2':crb.crb_d_m2,'CRB_v_m2ps2':crb.crb_v_m2ps2,'scalar_crb':float(C),'target_wcrb':float(W),
                                'FIM_dd':float(crb.fim[0,0]),'FIM_dv':float(crb.fim[0,1]),'FIM_vv':float(crb.fim[1,1])})
                        for r in ao.candidate_records:
                            value=r['physical_weighted_crb']
                            value=value if np.isfinite(value) else 'infinity_expected_zero_power' if r['Ps_w']==0 and r['feasible'] else ''
                            candidates.add({**base_row,'candidate_index':r['candidate_index'],'ao_round':r['ao_round'],'stage':r['stage'],
                                'candidate_Ts_s':r['Ts_s'],'candidate_Ps_w':r['Ps_w'],'feasible':r['feasible'],
                                'inner_evaluated':r['p1_called'],'allocation_objective':value,**{k:r[k] for k in margins}})
                        for r in ao.ao_trace:trace.add({**base_row,'ao_round':r['ao_iteration'],'stage':r['stage'],
                            'Ts_s':r['Ts_s'],'Ps_w':r['Ps_w'],'allocation_objective':r['weighted_crb']})
                        reference=outer_reference(method,state,weights,system)
                        references.append({**base_row,**reference,'ao_allocation_objective':final.optimizer_objective,
                            'relative_objective_gap_percent':100*(final.optimizer_objective/reference['reference_allocation_objective']-1)})
                        if point==25:
                            recorded=next(r for r in frozen['baseline_all_methods'] if r['method']==method)
                            baseline_checks.append({'sweep':sweep,'method':method,'cold_start_wcrb':common.total_wcrb,
                                'recorded_warm_start_wcrb':float(recorded['total_wcrb']),
                                'relative_difference_percent':100*(common.total_wcrb/float(recorded['total_wcrb'])-1),
                                'Ts_difference_s':final.total_sensing_s-float(recorded['Ts_s']),
                                'Ps_difference_w':final.sensing_power_w-float(recorded['Ps_w'])})
                    wide[sweep].append(wrow)
                    if point%10==0:
                        print(f'{sweep}: {point+1}/51 points; all five policies solved; x={x:.6g}; inner evaluations={sum(m["inner_evaluations"] for m in metrics.values())}',flush=True)
                        save(ROOT/'progress.json',{'sweep':sweep,'point_index':point,'completed_method_points':len(results),'elapsed_s':time.perf_counter()-begin})
        finally:candidates.close();trace.close()
    assert len(results)==510 and len(targets)==2040 and len(physical)==510
    for name,rows in [('method_points.csv',results),('target_allocations.csv',targets),('controlled_states.csv',physical),
        ('outer_precision_checks.csv',references),('baseline_point_checks.csv',baseline_checks),
        ('fig8b_five_methods.csv',wide['range']),('fig8c_five_methods.csv',wide['closing_speed'])]:csvsave(name,rows)
    save(ROOT/'candidate_metrics.json',metrics)
    save(ROOT/'completion.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'wall_seconds':time.perf_counter()-begin,'actual_AO_solves':510,'method_points':510,'target_rows':2040,
        'points_per_sweep':51,'all_checks_passed':True,'new_SUMO_trajectories_claimed':False,'figure8a_data_modified':False})
    from plot_sensitivity import draw
    draw()
    print('All 510 AO solves, full intermediate data and both figures completed.',flush=True)

if __name__=='__main__':main()
