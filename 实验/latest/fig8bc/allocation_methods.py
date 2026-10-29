"""Five actual policies; one original physical CRB evaluator and AO engine.

The original source files are immutable snapshots. The AO function's code
object and block optimizers are reused; only its evaluator factory is injected.
"""
from dataclasses import dataclass
from math import isfinite, inf
import types
import numpy as np
import model

METHODS=('TTC-AO','RU-AO','DA-AO','TTC-PD','UD')
LABELS={'TTC-AO':'Proposed TTC-AO','RU-AO':'Risk-Unaware AO','DA-AO':'Distance-Aware AO',
        'TTC-PD':'TTC-Proportional','UD':'Uniform-Dwell'}
EPSILON_D_M=.1

@dataclass(frozen=True)
class CurrentState:
    targets:tuple
    beams:tuple
    radar:model.RadarConfig
    physical_ttc_weights:tuple

@dataclass(frozen=True)
class CommonEvaluation:
    crb_results:tuple
    scalar_crb:np.ndarray
    target_wcrb:np.ndarray
    total_wcrb:float

def evaluate_common_ttc_wcrb(state,dwell_times,sensing_power,physical_ttc_weights):
    """The ONLY final comparison metric used by all five methods."""
    weights=np.asarray(physical_ttc_weights,dtype=float)
    assert np.array_equal(weights,np.asarray(state.physical_ttc_weights))
    dwell=np.asarray(dwell_times,dtype=float)
    assert dwell.shape==weights.shape==(len(state.targets),)
    results=tuple(model.fim_and_crb(t,b,float(tk),float(sensing_power),state.radar)
                  for t,b,tk in zip(state.targets,state.beams,dwell))
    scalar=np.array([r.scalar_crb for r in results])
    if sensing_power==0:
        # The original feasible power endpoint is unidentifiable, not a final
        # finite estimate. Avoid undefined 0*infinity for zero-risk targets.
        return CommonEvaluation(results,scalar,np.full(len(dwell),inf),inf)
    assert np.isfinite(scalar).all() and np.all(scalar>=0)
    assert all(np.isfinite(r.fim).all() and np.linalg.eigvalsh(r.fim).min()>0 for r in results)
    contributions=weights*scalar
    return CommonEvaluation(results,scalar,contributions,float(np.sum(contributions)))

def optimization_weights(method,state):
    if method=='RU-AO':return np.ones(len(state.targets))
    if method=='DA-AO':
        # Eligibility agrees with the existing TTC branch; velocity magnitude
        # is NOT used for DA priority once the target qualifies.
        raw=np.array([(1/(t.distance_m+EPSILON_D_M) if t.closing_speed_mps>0 else 0.) for t in state.targets])
        return len(raw)*raw/raw.sum() if raw.sum()>0 else np.ones(len(raw))
    return np.asarray(state.physical_ttc_weights,dtype=float)

def direct_dwell(method,weights,total_s,minimum_s):
    k=len(weights)
    assert total_s>=k*minimum_s-1e-12
    if method=='UD':return np.full(k,total_s/k)
    assert method=='TTC-PD'
    w=np.asarray(weights,dtype=float)
    shares=w/w.sum() if w.sum()>0 else np.full(k,1/k)
    dwell=minimum_s+(total_s-k*minimum_s)*shares
    # Correct only roundoff in the equality, not the allocation rule.
    dwell[int(np.argmax(dwell))]+=total_s-float(dwell.sum())
    return dwell

@dataclass(frozen=True)
class DirectAllocation:
    """AO-compatible inner-result interface; KKT is genuinely not applicable."""
    dwell_s:np.ndarray
    crb_results:tuple
    per_target_scalar_crb:np.ndarray
    ordinary_total_crb:float
    weighted_total_crb:float
    kkt_residual:None
    dwell_sum_error_s:float
    minimum_dwell_margin_s:float
    status:str

def allocate_at(method,state,weights,total_s,power,minimum_s):
    if method in ('TTC-AO','RU-AO','DA-AO'):
        return model.solve_p1(state.targets,state.beams,weights,total_s,power,minimum_s,state.radar)
    dwell=direct_dwell(method,weights,total_s,minimum_s)
    evaluated=evaluate_common_ttc_wcrb(state,dwell,power,state.physical_ttc_weights)
    return DirectAllocation(dwell,evaluated.crb_results,evaluated.scalar_crb,
        float(evaluated.scalar_crb.sum()),evaluated.total_wcrb,None,
        abs(float(dwell.sum())-total_s),float(dwell.min()-minimum_s),
        'zero_power_unidentifiable' if power==0 else ('direct_proportional' if method=='TTC-PD' else 'direct_uniform'))

def make_ao_solver(method,state,metrics):
    """Reuse byte-identical original AO code, same block solvers and settings.

    Native weighted P1 methods delegate to the original evaluator unchanged.
    Direct rules implement its candidate interface, retaining the same checks.
    The original AOResult.p1_calls field counts inner evaluations for adapters;
    output explicitly distinguishes actual P1 solves from direct allocations.
    """
    class PolicyEvaluator(model.CandidateEvaluator):
        def evaluate(self,Ts,Ps,stage,ao_round):
            if method in ('TTC-AO','RU-AO','DA-AO'):
                result=super().evaluate(Ts,Ps,stage,ao_round)
            else:
                margins=model.constraint_margins(Ts,Ps,len(self.targets),self.system)
                feasible=isfinite(Ts) and isfinite(Ps) and min(margins.values())>=-self.system.feasibility_tolerance
                inner=None
                if feasible:
                    inner=allocate_at(method,state,self.weights,Ts,Ps,self.system.min_dwell_s)
                    self.p1_calls+=1
                    physical=inner.weighted_total_crb;value=physical;time_cost=energy_cost=0.
                else:
                    self.infeasible_evaluations+=1
                    violation=sum(max(0.,-v)**2 for v in margins.values() if isfinite(v))
                    value=self.system.infeasible_penalty*(1+violation)
                    physical=float('nan');time_cost=energy_cost=float('nan')
                self.records.append({'cycle_index':self.cycle_index,'candidate_index':len(self.records)+1,
                    'ao_round':ao_round,'stage':stage,'Ts_s':Ts,'Ps_w':Ps,'feasible':int(feasible),
                    'p1_called':int(feasible),'physical_weighted_crb':physical,
                    'time_resource_cost':time_cost,'energy_resource_cost':energy_cost,
                    'optimizer_value':value,**margins})
                result=model.CandidateResult(Ts,Ps,physical,value,time_cost,energy_cost,feasible,margins,inner,stage,ao_round)
            if result.feasible:
                inner=result.p1
                metrics['inner_evaluations']+=1
                metrics['complete_p1_solves' if method in METHODS[:3] else 'direct_allocations']+=1
                error=abs(float(inner.dwell_s.sum())-Ts)
                floor=float(inner.dwell_s.min()-self.system.min_dwell_s)
                metrics['max_dwell_sum_error_s']=max(metrics['max_dwell_sum_error_s'],error)
                metrics['minimum_dwell_margin_s']=min(metrics['minimum_dwell_margin_s'],floor)
                assert error<1e-12 and floor>=-1e-12
                if Ps==0:metrics['expected_zero_power_endpoints']+=1
                else:
                    assert np.isfinite(inner.weighted_total_crb) and np.isfinite(inner.per_target_scalar_crb).all()
                    for crb in inner.crb_results:
                        assert np.isfinite(crb.fim).all() and crb.fim[0,0]>0 and np.linalg.det(crb.fim)>0
                    if inner.kkt_residual is not None:
                        metrics['maximum_p1_kkt_residual']=max(metrics['maximum_p1_kkt_residual'],inner.kkt_residual)
                    if method in METHODS[:3]:
                        assert abs(float(np.dot(self.weights,inner.per_target_scalar_crb))-inner.weighted_total_crb)<1e-12
            return result
    namespace=dict(model.alternating_optimize_system.__globals__)
    namespace['CandidateEvaluator']=PolicyEvaluator
    original=model.alternating_optimize_system
    solver=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    solver.__kwdefaults__=original.__kwdefaults__
    return solver

def outer_reference(method,state,weights,system):
    """Independent precision diagnostic; never replaces the plotted AO result.

    Exact P1 and uniform: maximal QoS time at its allowed power is globally
    optimal for this model (scale dwell up; range term cannot increase, velocity
    term decreases). PD: energy-boundary priority shares vary with Ts, so use
    an explicit grid plus local bounded refinement, not a global certificate.
    """
    lo=len(state.targets)*system.min_dwell_s
    hi=system.sensing_time_qos_max_s-1e-12
    def evaluate(Ts):
        Ps=min(system.sensing_power_max_w,system.sensing_energy_max_j/Ts)
        inner=allocate_at(method,state,weights,float(Ts),float(Ps),system.min_dwell_s)
        return inner.weighted_total_crb,float(Ts),float(Ps),inner
    if method!='TTC-PD':
        value,Ts,Ps,inner=evaluate(hi)
        kind='model_scaling_boundary_certificate'
    else:
        from scipy.optimize import minimize_scalar
        grid=np.linspace(lo,hi,81)
        samples=[evaluate(t) for t in grid]
        index=min(range(len(samples)),key=lambda i:samples[i][0])
        candidates=[samples[index],samples[0],samples[-1]]
        left=grid[max(0,index-1)];right=grid[min(len(grid)-1,index+1)]
        if right>left:
            refined=minimize_scalar(lambda t:evaluate(t)[0],bounds=(left,right),method='bounded',options={'xatol':1e-12})
            candidates.append(evaluate(refined.x))
        value,Ts,Ps,inner=min(candidates,key=lambda r:r[0])
        kind='81_point_grid_plus_bounded_refinement_not_global_certificate'
    common=evaluate_common_ttc_wcrb(state,inner.dwell_s,Ps,state.physical_ttc_weights)
    return {'reference_kind':kind,'reference_Ts_s':Ts,'reference_Ps_w':Ps,
            'reference_allocation_objective':value,'reference_common_ttc_wcrb':common.total_wcrb}
