"""Replay the recorded SUMO route with causal radar tracking and allocation."""
import csv
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = HERE / "inputs"
import model
import replay_scene as original
from ttc_precision import TTCInfeasible, TTCPrecision

MINIMAX = "MinMax-AO"
INIT_COV = np.diag([0.2**2, 0.1**2])
ANGLE_NOISE = np.deg2rad(0.2)
ACCEL_PSD = 0.02  # m²/s³, continuous white relative acceleration


@dataclass(frozen=True)
class CurrentState:
    targets: tuple
    beams: tuple
    radar: model.RadarConfig
    physical_ttc_weights: tuple


def optimization_weights(method, state):
    if method == "RU-AO":
        return np.ones(len(state.targets))
    return np.asarray(state.physical_ttc_weights, dtype=float)


def direct_dwell(method, weights, total_s, minimum_s):
    k = len(weights)
    assert total_s >= k * minimum_s - 1e-12
    if method == "UD":
        return np.full(k, total_s / k)
    assert method == "TTC-PD"
    shares = weights / weights.sum() if weights.sum() > 0 else np.full(k, 1 / k)
    dwell = minimum_s + (total_s - k * minimum_s) * shares
    dwell[int(np.argmax(dwell))] += total_s - float(dwell.sum())
    return dwell


def load_traffic(path):
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            state = model.VehicleState(**{
                f.name: (row[f.name] if f.type in (str, "str") else
                         int(row[f.name]) if f.type in (int, "int") else float(row[f.name]))
                for f in fields(model.VehicleState)
            })
            groups.setdefault(state.time_s, []).append(state)
    return sorted(groups.items())


def predict(mean, covariance, dt):
    transition = np.array([[1.0, -dt], [0.0, 1.0]])  # Positive v means closing.
    process_noise = ACCEL_PSD * np.array([[dt**3 / 3, -dt**2 / 2], [-dt**2 / 2, dt]])
    return mean @ transition.T, transition @ covariance @ transition.T + process_noise


def information(dwell, power, coefficients):
    return power * coefficients * np.column_stack((dwell, dwell**3))


def posterior(prior_information, measurement_information):
    updated = prior_information.copy()
    updated[:, 0, 0] += measurement_information[:, 0]
    updated[:, 1, 1] += measurement_information[:, 1]
    return np.linalg.inv(updated)


def weighted_bound(covariance, weights, references):
    return weights @ (np.diagonal(covariance, axis1=-2, axis2=-1) @ references)


def allocate_dwell(total, power, minimum, coefficients, prior_info, weights, references, recursive, required=None):
    # This KKT solution remains valid for the original single-frame objective.
    instant, _ = model._allocate_convex_dwell(
        references[0] / coefficients[:, 0], references[1] / coefficients[:, 1],
        weights, total, minimum, power,
    )
    if required is None and (not recursive or total <= len(weights) * minimum + 1e-12):
        return instant
    floor = np.full(len(weights), minimum / total) if required is None else required / total
    free = max(0., 1 - floor.sum())
    uniform = floor + free / len(weights)
    if free < 1e-13:
        dwell = uniform * total
        dwell[np.argmax(dwell)] += total - dwell.sum()
        return dwell
    scale = weighted_bound(
        posterior(prior_info, information(uniform * total, power, coefficients)), weights, references
    )

    def objective(shares):
        dwell = shares * total
        covariance = posterior(prior_info, information(dwell, power, coefficients))
        # d tr(W P)/dJ_j = -[P W P]_jj.
        sensitivity = np.einsum("kij,j,kij->ki", covariance, references, covariance)
        diagonal_gradient = power * total * coefficients * np.column_stack((np.ones(len(dwell)), 3 * dwell**2))
        gradient = -weights * np.sum(sensitivity * diagonal_gradient, axis=1)
        return weighted_bound(covariance, weights, references) / scale, gradient / scale

    constraint = {"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones_like(x)}
    vertices = floor + free * np.eye(len(weights))
    instant_share = instant / total
    if required is not None:
        extra = np.maximum(instant_share - floor, 0)
        instant_share = floor + free * extra / extra.sum() if extra.sum() else uniform
    starts = (uniform, instant_share, *vertices)
    best = min(starts, key=lambda x: objective(x)[0])
    successes = 0
    for start in starts:
        result = minimize(objective, start, jac=True, method="SLSQP",
                          bounds=list(zip(floor, np.ones(len(weights)))), constraints=constraint,
                          options={"ftol": 1e-10, "maxiter": 60})
        if result.success and abs(result.x.sum() - 1) < 1e-9 and np.all(result.x >= floor - 1e-10):
            successes += 1
            if objective(result.x)[0] < objective(best)[0]:
                best = result.x
    if successes == 0:
        raise RuntimeError("All posterior P1 starts failed")
    dwell = best * total
    dwell[np.argmax(dwell)] += total - dwell.sum()
    return dwell


def minimax_dwell(total, power, minimum, coefficients, prior_info, references, required=None):
    """Minimize the largest normalized posterior CRB under the dwell budget."""
    count = len(coefficients)
    floor = np.full(count, minimum / total) if required is None else required / total
    free = 1 - floor.sum()
    if free <= 1e-12:
        return (floor + free / count) * total

    alpha = prior_info[:, 0, 0]
    beta = prior_info[:, 1, 1]
    cross_sq = prior_info[:, 0, 1] ** 2
    range_scale = power * coefficients[:, 0] * total
    velocity_scale = power * coefficients[:, 1] * total**3
    ref_d, ref_v = references
    max_share = floor + free

    def bound(k, share):
        dd = alpha[k] + range_scale[k] * share
        vv = beta[k] + velocity_scale[k] * share**3
        return (ref_d * vv + ref_v * dd) / (dd * vv - cross_sq[k])

    floor_bounds = [bound(k, floor[k]) for k in range(count)]
    cap_bounds = [bound(k, max_share[k]) for k in range(count)]

    def required_shares(threshold):
        shares = np.empty(count)
        for k in range(count):
            if threshold >= floor_bounds[k]:
                shares[k] = floor[k]
            elif threshold <= cap_bounds[k]:
                shares[k] = max_share[k]
            else:
                shares[k] = brentq(lambda x: bound(k, x) - threshold, floor[k], max_share[k], xtol=1e-14)
        return shares

    lower, upper = max(cap_bounds), max(floor_bounds)
    if required_shares(lower).sum() <= 1 + 1e-13:
        threshold = lower
    else:
        threshold = upper * brentq(lambda level: required_shares(upper * level).sum() - 1,
                                   lower / upper, 1.0, xtol=1e-13)
    shares = required_shares(threshold)
    shares[np.argmax(shares)] += 1 - shares.sum()
    return shares * total


def evaluate_candidate(ts, ps, system, coefficients, prior_info, weights, references,
                       recursive, carry=None, direct=None, *, minimax=False, precision=None, required=None):
    if precision is not None and required is None:
        required = precision.required_dwell(ts, ps, system.min_dwell_s, coefficients, prior_info)
    effective = required if required is not None and np.any(required > system.min_dwell_s) else None
    if minimax:
        dwell = minimax_dwell(ts, ps, system.min_dwell_s, coefficients, prior_info, references, effective)
    elif direct is None:
        dwell = allocate_dwell(ts, ps, system.min_dwell_s, coefficients, prior_info,
                               weights, references, recursive, effective)
    else:
        dwell = direct_dwell(direct, weights, ts, system.min_dwell_s)
    candidates = (dwell, carry * (ts / carry.sum())) if carry is not None and direct is None else (dwell,)
    best = None
    for candidate in candidates:
        measurement = information(candidate, ps, coefficients)
        post = posterior(prior_info, measurement)
        if precision is not None and np.any(precision.gamma(post)[precision.active] >
                                            precision.gamma_limit * (1 + 1e-9)):
            continue
        target_bounds = np.reciprocal(measurement) @ references
        bound = weighted_bound(post, weights, references)
        if minimax:
            objective = float(np.max(np.diagonal(post, axis1=-2, axis2=-1) @ references))
        else:
            objective = bound if recursive else weights @ target_bounds
        if best is None or objective < best["objective"]:
            best = dict(time=ts, power=ps, dwell=candidate, post=post,
                        bound=bound, objective=objective, required_dwell=required)
    if best is None:
        if direct is not None:
            raise TTCInfeasible(f"{direct}_fixed_allocation_violates_TTC_limit", required)
        raise RuntimeError("P1 failed the posterior TTC precision check after a feasible budget precheck")
    if (abs(best["dwell"].sum() - ts) > 1e-10 or best["dwell"].min() < system.min_dwell_s - 1e-10
            or min(model.constraint_margins(ts, ps, len(weights), system).values()) < -1e-9):
        raise RuntimeError("Resource allocation violated a constraint")
    return best


def allocate(system, coefficients, prior_info, weights, references, recursive, previous, direct=None,
             *, minimax=False, precision=None):
    # Each coordinate is monotone in information. Scaling all dwell times
    # upward while holding total energy preserves Jdd and increases Jvv.
    maximum_time = system.sensing_time_qos_max_s - 1e-12
    maximum_power = min(system.sensing_power_max_w, system.sensing_energy_max_j / maximum_time)
    if precision is not None and not recursive:
        raise ValueError("Posterior TTC precision requires recursive allocation")
    cap_required = None
    if precision is not None:
        # Scaling all dwells to the time cap at fixed energy improves or preserves both
        # information entries. If this corner is infeasible, no P2 allocation can work.
        cap_required = precision.required_dwell(maximum_time, maximum_power, system.min_dwell_s,
                                                coefficients, prior_info)
    time, power = previous or (system.initial_sensing_time_s, system.initial_sensing_power_w)
    time = np.clip(time, len(weights) * system.min_dwell_s, maximum_time)
    power = min(power, maximum_power)
    steps, rejected = [], []
    for stage, ts, ps in (("initial", time, power), ("time", maximum_time, power),
                          ("power", maximum_time, maximum_power)):
        if steps and (ts, ps) == (steps[-1]["time"], steps[-1]["power"]):
            steps.append(dict(steps[-1], stage=stage))
            continue
        carry = steps[-1]["dwell"] if steps else None
        required = None
        if precision is not None:
            try:
                required = cap_required if stage == "power" else precision.required_dwell(
                    ts, ps, system.min_dwell_s, coefficients, prior_info)
                best = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                          references, recursive, carry, direct, minimax=minimax,
                                          precision=precision, required=required)
            except TTCInfeasible as error:
                if stage == "power":
                    raise
                rejected.append(f"{stage}:{error}")
                continue
        else:
            best = evaluate_candidate(ts, ps, system, coefficients, prior_info, weights,
                                      references, recursive, carry, direct, minimax=minimax)
        if steps and best["objective"] > steps[-1]["objective"] * (1 + 1e-10):
            raise RuntimeError("Block update increased its objective")
        steps.append(dict(best, stage=stage, required_dwell=required, rejected_stages=";".join(rejected)))
    return steps


def track(replay, warmup, radar, system, seed, references, methods, *, radar_overrides=None,
          ttc_gamma_limit=None, feedback_enabled=True, constrain_warmup=True):
    radar_overrides = radar_overrides or {}
    rng = np.random.default_rng(seed)
    warm_time, _, warm_targets = warmup
    discovery = np.array([(target.distance_m, target.closing_speed_mps) for target in warm_targets])
    init_mean = discovery + rng.standard_normal(discovery.shape) * np.sqrt(INIT_COV.diagonal())
    init_angle = np.array([target.azimuth_rad for target in warm_targets]) + ANGLE_NOISE * rng.standard_normal(len(discovery))
    histories = {}
    for method in methods:
        histories[method] = (init_mean, np.broadcast_to(INIT_COV, (len(discovery), 2, 2)),
                             init_angle, None, None)
    last_time = warm_time
    for cycle, (time, ego, targets) in enumerate(replay):
        dt = time - last_time
        truth = np.array([(target.distance_m, target.closing_speed_mps) for target in targets])
        truth_weights = model.compute_ttc_weights(targets).omega
        true_angle = np.array([target.azimuth_rad for target in targets])
        # The current angle reading becomes available only after resource selection.
        next_angle = true_angle + ANGLE_NOISE * rng.standard_normal(len(targets))
        noise = rng.standard_normal(truth.shape)
        snr, _ = original.communication_snr_from_rsu(ego)
        cycle_system = replace(system, communication_snr_linear=snr)
        for method, (mean, cov, angle, previous, previous_bound) in histories.items():
            predicted, prior = predict(mean, cov, dt)
            if np.any(predicted[:, 0] <= 0):
                raise RuntimeError("Predicted range outside the observation model")
            # Distance, closing speed and angle are from history. Geometry
            # fields unused by scheduling are inherited from the warm-up snapshot.
            predicted_targets = [
                replace(target, distance_m=x[0], closing_speed_mps=x[1], azimuth_rad=a)
                for target, x, a in zip(warm_targets, predicted, angle)
            ]
            weights = model.compute_ttc_weights(predicted_targets).omega
            beams = [model.closed_form_beams(a, radar) for a in angle]
            state = CurrentState(predicted_targets, beams, radar, tuple(weights))
            allocation_weights = optimization_weights(method, state)
            coefficients = np.array([
                model.information_coefficients_per_watt(target, beam, radar_overrides.get(target.vehicle_id, radar))[:2]
                for target, beam in zip(predicted_targets, beams)
            ])
            direct = method if method in ("TTC-PD", "UD") else None
            definition = TTCPrecision(predicted, ttc_gamma_limit)
            precision = definition if ttc_gamma_limit is not None and (constrain_warmup or time >= 0) else None
            scheduling_prior = prior
            if not feedback_enabled:
                _, scheduling_prior = predict(mean, np.broadcast_to(INIT_COV, prior.shape), dt)
            try:
                steps = allocate(cycle_system, coefficients, np.linalg.inv(scheduling_prior), allocation_weights, references,
                                 True, previous, direct, minimax=method == MINIMAX, precision=precision)
            except TTCInfeasible as error:
                # A failed cycle has no radar measurement. Propagate the predicted state and prior.
                histories[method] = predicted, prior, angle, previous, weighted_bound(prior, weights, references)
                for k, target in enumerate(targets):
                    yield dict(seed=seed, cycle=cycle, time_s=time, method=method, target=target.vehicle_id,
                               predicted_range_m=predicted[k, 0], predicted_closing_mps=predicted[k, 1],
                               truth_weight=truth_weights[k], predicted_weight=weights[k],
                               post_dd=prior[k, 0, 0], post_dv=prior[k, 0, 1], post_vv=prior[k, 1, 1],
                               allocation_status="infeasible", infeasible_reason=str(error),
                               required_dwell_s=error.required[k], max_Ts_s=cycle_system.sensing_time_qos_max_s)
                continue
            best = steps[-1]
            dwell, ts, power, post = best["dwell"], best["time"], best["power"], best["post"]
            assumed_info = information(dwell, power, coefficients)
            scheduling_gamma = definition.gamma(post)
            if not feedback_enabled:
                post = posterior(np.linalg.inv(prior), assumed_info)
            actual_coefficients = []
            for target, beam in zip(targets, beams):
                actual = model.closed_form_beams(target.azimuth_rad, radar)
                pointed = replace(
                    beam,
                    tx_gain_linear=abs(np.vdot(actual.tx_steering, beam.tx_beam))**2,
                    rx_gain_linear=abs(np.vdot(beam.rx_beam, actual.rx_steering))**2,
                )
                target_radar = radar_overrides.get(target.vehicle_id, radar)
                actual_coefficients.append(model.information_coefficients_per_watt(target, pointed, target_radar)[:2])
            realized_info = information(dwell, power, np.array(actual_coefficients))
            measurement = truth + noise / np.sqrt(realized_info)
            gain = post * assumed_info[:, None, :]
            updated = predicted + np.einsum("kij,kj->ki", gain, measurement - predicted)
            if np.linalg.eigvalsh(prior - post).min() < -1e-12 or np.linalg.eigvalsh(post).min() <= 0:
                raise RuntimeError("Posterior failed covariance ordering")
            histories[method] = updated, post, next_angle, (ts, power), weighted_bound(post, weights, references)
            for k, (target, truth_weight, weight, estimate, observation, actual_state, forecast, bearing, prior_k, post_k, old_cov, tk, ji, jr) in enumerate(zip(
                targets, truth_weights, weights, updated, measurement, truth, predicted, angle, prior, post, cov,
                dwell, assumed_info, realized_info,
            )):
                yield dict(seed=seed, cycle=cycle, time_s=time, method=method, target=target.vehicle_id,
                           previous_weighted_pcrb=previous_bound,
                           truth_weight=truth_weight, predicted_weight=weight,
                           predicted_azimuth_rad=bearing,
                           predicted_range_m=forecast[0], predicted_closing_mps=forecast[1],
                           previous_post_dd=old_cov[0, 0], previous_post_dv=old_cov[0, 1], previous_post_vv=old_cov[1, 1],
                           prior_dd=prior_k[0, 0], prior_dv=prior_k[0, 1], prior_vv=prior_k[1, 1],
                           post_dd=post_k[0, 0], post_dv=post_k[0, 1], post_vv=post_k[1, 1],
                           predicted_Jdd=ji[0], predicted_Jvv=ji[1], realized_Jdd=jr[0], realized_Jvv=jr[1],
                           estimated_range_m=estimate[0], estimated_closing_mps=estimate[1],
                           measured_range_m=observation[0], measured_closing_mps=observation[1],
                           true_range_m=actual_state[0], true_closing_mps=actual_state[1],
                           Ts_s=ts, Ps_w=power, dwell_s=tk, communication_rate_bps=model.communication_rate_bps(ts, cycle_system),
                           scheduled_prior_wPCRB=weighted_bound(prior, weights, references),
                           initial_wPCRB=next((s["bound"] for s in steps if s["stage"] == "initial"), None),
                           time_update_wPCRB=next((s["bound"] for s in steps if s["stage"] == "time"), None),
                           optimized_wPCRB=weighted_bound(post, weights, references),
                           initial_objective=steps[0]["objective"],
                           optimized_objective=best["objective"], allocation_status="feasible", infeasible_reason="",
                           gamma_plus=scheduling_gamma[k], gamma_limit=precision.gamma_limit if precision else None,
                           required_dwell_s=(best["required_dwell"][k]
                                             if best["required_dwell"] is not None else None),
                           rejected_stages=best["rejected_stages"], max_Ts_s=cycle_system.sensing_time_qos_max_s,
                           **model.constraint_margins(ts, power, len(targets), cycle_system))
        last_time = time


def extract_snapshots(rows):
    for time, states in rows:
        ego, targets, missing = original.fixed_target_observations(states, "ego")
        if missing:
            raise RuntimeError(f"Incomplete snapshot at {time} s: {missing}")
        yield time, ego, targets

